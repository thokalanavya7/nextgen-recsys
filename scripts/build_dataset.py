"""Data preprocessing pipeline (architecture stage: DATA PREPROCESSING).

Reads the raw Amazon reviews + product metadata, then produces:

  data/processed/products.csv      - cleaned product catalog (asin, title,
                                     category, brand, price, text, image, sentiment)
  data/processed/interactions.csv  - cleaned implicit/explicit interactions
  data/processed/train.csv         - train split (per-user chronological)
  data/processed/test.csv          - held-out last interaction per user

Cleaning steps: missing-value handling, de-duplication, price/brand parsing,
k-core sparsity filtering, review-sentiment scoring, feature extraction
(product text document), and the user-item matrix is materialised later in
train.py (kept sparse - this dataset is too large for a dense matrix).
"""
import gzip
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
CATEGORY = os.environ.get("CATEGORY", "All_Beauty")
# interactions come from the Amazon 2014 "Beauty" 5-core file (SNAP) -
# the dense benchmark subset - and product metadata from the matching
# 2014 metadata dump, stream-filtered to meta_beauty_2014.jsonl by
# download_data.py.
REVIEWS_FILE = os.environ.get("REVIEWS_FILE", "reviews_Beauty_5.json.gz")
META_FILE = os.environ.get("META_FILE", "meta_beauty_2014.jsonl")
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
RATING_POS_THRESHOLD = 4.0
MAX_DESC_CHARS = 600


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def parse_price(value):
    if value is None:
        return np.nan
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value).replace(",", ""))
    return float(m.group(1)) if m else np.nan


def flatten(value, max_chars=None):
    if isinstance(value, list):
        value = " ".join(str(v) for v in value if v)
    value = str(value or "").strip()
    value = re.sub(r"\s+", " ", value)
    if max_chars:
        value = value[:max_chars]
    return value


def iter_meta(path):
    """Yield records from the 2014 metadata dump (some lines are Python-
    literal dicts, not strict JSON - eval safely with a fallback)."""
    import ast
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                try:
                    yield ast.literal_eval(line)
                except (ValueError, SyntaxError):
                    continue


def load_products():
    path = os.path.join(RAW, META_FILE)
    rows = []
    for r in iter_meta(path):
        asin = r.get("asin")
        title = flatten(r.get("title"))
        if not asin or not title:
            continue
        # 2014 meta: categories is a list of lists, e.g. [["Beauty","Hair Care"]]
        cats = r.get("categories") or []
        if cats and isinstance(cats[0], list):
            cats = cats[0]
        cats = [flatten(c) for c in cats if flatten(c)]
        main_cat = cats[1] if len(cats) > 1 else (cats[0] if cats else "Unknown")
        image = flatten(r.get("imUrl"))
        if not image:
            images = r.get("imageURLHighRes") or r.get("imageURL") or []
            image = images[0] if images else ""
        rows.append({
            "asin": asin,
            "title": title,
            "category": main_cat,
            "category_path": " > ".join(cats[:3]),
            "brand": flatten(r.get("brand")),
            "price": r.get("price") if isinstance(r.get("price"), (int, float)) else parse_price(r.get("price")),
            "image": image,
            "description": flatten(r.get("description"), MAX_DESC_CHARS),
        })
    df = pd.DataFrame(rows).drop_duplicates("asin")
    # product document used by content-based filtering
    df["doc"] = (
        df["title"].fillna("") + " " +
        df["brand"].fillna("") + " " +
        df["category_path"].fillna("") + " " +
        df["description"].fillna("")
    ).str.lower()
    df["brand"] = df["brand"].replace("", "Unknown")
    return df


def load_interactions():
    path = os.path.join(RAW, REVIEWS_FILE)
    rows = []
    for r in read_jsonl_gz(path):
        user, asin, rating = r.get("reviewerID"), r.get("asin"), r.get("overall")
        if not user or not asin or rating is None:
            continue
        rows.append({
            "user": user,
            "asin": asin,
            "rating": float(rating),
            "ts": int(r.get("unixReviewTime", 0)),
            "review": flatten(r.get("reviewText"), 2000),
        })
    return pd.DataFrame(rows)


def kcore_filter(df, min_u, min_i, max_rounds=4):
    """Iteratively drop users/items below the interaction threshold."""
    for _ in range(max_rounds):
        before = len(df)
        df = df[df.user.isin(df.user.value_counts()[lambda s: s >= min_u].index)]
        df = df[df.asin.isin(df.asin.value_counts()[lambda s: s >= min_i].index)]
        if len(df) == before:
            break
    return df


def main():
    os.makedirs(OUT, exist_ok=True)
    print("loading product metadata...")
    products = load_products()
    print(f"products: {len(products)}")

    print("loading reviews...")
    inter = load_interactions()
    print(f"raw interactions: {len(inter)}")

    # de-duplicate (user, item): keep the most recent rating
    inter = inter.sort_values("ts").drop_duplicates(["user", "asin"], keep="last")

    # keep only products present in the cleaned catalogue
    inter = inter[inter.asin.isin(set(products.asin))]
    print(f"after join with catalog: {len(inter)}")

    # sparsity control via k-core
    inter = kcore_filter(inter, MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)
    products = products[products.asin.isin(set(inter.asin))].copy()
    print(f"after {MIN_USER_INTERACTIONS}-core filter: {len(inter)} interactions, "
          f"{inter.user.nunique()} users, {inter.asin.nunique()} items")

    # review sentiment -> item-level sentiment prior
    print("scoring review sentiment (VADER)...")
    sia = SentimentIntensityAnalyzer()
    texts = inter["review"].fillna("")
    inter["sentiment"] = [
        sia.polarity_scores(t)["compound"] if t else 0.0 for t in texts
    ]
    item_sent = inter.groupby("asin")["sentiment"].mean().rename("item_sentiment")
    products = products.merge(item_sent, on="asin", how="left")
    products["item_sentiment"] = products["item_sentiment"].fillna(0.0)

    # chronological split: last interaction per user -> test
    inter = inter.sort_values(["user", "ts"])
    last = inter.groupby("user").tail(1)
    train = inter.drop(last.index)
    test = last[["user", "asin", "rating", "ts"]]
    train[["user", "asin", "rating", "ts"]].to_csv(
        os.path.join(OUT, "train.csv"), index=False)
    test.to_csv(os.path.join(OUT, "test.csv"), index=False)
    inter[["user", "asin", "rating", "ts", "sentiment"]].to_csv(
        os.path.join(OUT, "interactions.csv"), index=False)
    products.drop(columns=["doc"]).to_csv(
        os.path.join(OUT, "products.csv"), index=False)
    products[["asin", "doc"]].to_csv(
        os.path.join(OUT, "product_docs.csv"), index=False)

    print(f"train: {len(train)}  test: {len(test)}")
    print(f"wrote processed files to {OUT}")


if __name__ == "__main__":
    sys.exit(main())
