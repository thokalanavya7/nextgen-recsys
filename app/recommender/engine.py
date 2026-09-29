"""Hybrid recommendation engine - orchestrates the component models.

Pipeline (mirrors the project architecture):
  content score (CBF) + collaborative score (CF) + association-rule score
  -> weighted blend with popularity & review-sentiment priors
  -> rank top-N and attach human-readable reasons.
"""
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd

from .models import (AssociationRuleModel, CollaborativeModel,
                     ContentBasedModel, PopularityModel)

PROCESSED = os.environ.get(
    "DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
ARTIFACTS = os.environ.get(
    "ARTIFACT_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "artifacts"))

# hybrid weights, tuned on a held-out user sample (see artifacts/evaluation.csv)
DEFAULT_WEIGHTS = {
    "content": 0.35, "collaborative": 0.30, "association": 0.15,
    "popularity": 0.10, "sentiment": 0.10,
}


class RecommenderEngine:
    def __init__(self):
        self.ready = False

    # ---------------- training / loading ----------------
    def train(self):
        train = pd.read_csv(os.path.join(PROCESSED, "train.csv"))
        docs = pd.read_csv(os.path.join(PROCESSED, "product_docs.csv"))
        prods = pd.read_csv(os.path.join(PROCESSED, "products.csv"))
        self._fit_all(train, docs, prods)
        os.makedirs(ARTIFACTS, exist_ok=True)
        joblib.dump({k: getattr(self, k) for k in
                     ("popularity", "content", "collaborative",
                      "association", "catalog")},
                    os.path.join(ARTIFACTS, "models.joblib"), compress=3)
        self.ready = True
        return self

    def load(self):
        path = os.path.join(ARTIFACTS, "models.joblib")
        state = joblib.load(path)
        for k, v in state.items():
            setattr(self, k, v)
        self.ready = True
        return self

    def _fit_all(self, train, docs, prods):
        print("fitting popularity...")
        self.popularity = PopularityModel().fit(train)
        print("fitting content-based (TF-IDF)...")
        self.content = ContentBasedModel().fit(docs)
        print("fitting collaborative (item-item cosine)...")
        self.collaborative = CollaborativeModel().fit(train, self.content.asins)
        print("mining association rules (FP-Growth)...")
        self.association = AssociationRuleModel().fit(train)
        self.catalog = prods.set_index("asin")
        print(f"rules mined: {len(self.association.rules)}")

    # ---------------- inference ----------------
    def recommend(self, user_items: dict, k=10, exclude_seen=True,
                  weights=None, candidates=None):
        """user_items: {asin: rating-strength} from train + live events."""
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        items = self.content.asins if candidates is None else candidates
        seen = set(user_items)
        scores = {}
        scores["content"] = self.content.score(items, user_items)
        scores["collaborative"] = self.collaborative.score(items, user_items)
        scores["association"] = self.association.score(items, user_items)
        scores["popularity"] = self.popularity.score(items)
        sent = self.catalog.reindex(items)["item_sentiment"].fillna(0)
        scores["sentiment"] = np.clip((sent.to_numpy() + 1.0) / 2.0, 0, 1)

        final = sum(w[k_] * s for k_, s in scores.items())
        order = np.argsort(-final)
        out = []
        for i in order:
            asin = items[i]
            if exclude_seen and asin in seen:
                continue
            comps = {k_: float(scores[k_][i]) for k_ in w}
            out.append({
                "asin": asin,
                "score": round(float(final[i]), 4),
                "components": {k_: round(v, 4) for k_, v in comps.items()},
                "reasons": self._reasons(asin, user_items, comps, w),
            })
            if len(out) >= k:
                break
        return out

    def similar_products(self, asin, n=8):
        return [{"asin": a, "score": round(s, 4)}
                for a, s in self.content.nearest(asin, n)]

    def _reasons(self, asin, user_items, comps, w):
        reasons = []
        contrib = {k_: comps[k_] * w[k_] for k_ in w}
        top = sorted(contrib, key=contrib.get, reverse=True)[:2]
        meta = self.catalog.loc[asin] if asin in self.catalog.index else {}
        title = str(meta.get("title", asin))[:60] if len(meta) else asin

        for c in top:
            if contrib[c] <= 0:
                continue
            if c == "collaborative":
                # find which user item drove the score
                best_src, best_sim = None, 0.0
                for src in user_items:
                    for nbr, sim in self.collaborative.neighbors.get(src, []):
                        if nbr == asin and sim > best_sim:
                            best_src, best_sim = src, sim
                if best_src is not None:
                    st = self._short_title(best_src)
                    reasons.append(f"Customers with similar taste rated "
                                   f"'{st}' - this ranks close to it")
                else:
                    reasons.append("Recommended by collaborative filtering")
            elif c == "association":
                rule = self.association.explain(user_items, asin)
                if rule:
                    ants = ", ".join(self._short_title(a) for a in rule[0][:2])
                    reasons.append(f"Frequently bought with '{ants}' "
                                   f"(confidence {rule[1]:.0%})")
                else:
                    reasons.append("Frequently co-purchased product")
            elif c == "content":
                cat = str(meta.get("category", "this category"))
                reasons.append(f"Matches your interests in {cat}")
            elif c == "popularity":
                reasons.append("Trending with shoppers")
            elif c == "sentiment":
                reasons.append("Loved in customer reviews")
        if not reasons:
            reasons.append(f"Suggested product: {title}")
        return reasons[:2]

    def _short_title(self, asin):
        try:
            return str(self.catalog.loc[asin, "title"])[:50]
        except KeyError:
            return asin
