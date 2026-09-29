"""Download the Amazon Reviews 2018 dataset (All_Beauty category).

Source: Julian McAuley's UCSD dataset repository
https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

Citation (required if you use this data in a publication):
    Jianmo Ni, Jiacheng Li, Julian McAuley.
    "Justifying Recommendations using Distantly-Labeled Reviews and
    Fine-Grained Aspects." EMNLP 2019.
"""
import os
import sys
import requests

BASE = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2"
SNAP = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles"
CATEGORY = os.environ.get("CATEGORY", "All_Beauty")
FILES = [
    (f"{SNAP}/reviews_Beauty_5.json.gz", "reviews_Beauty_5.json.gz"),
    (f"{BASE}/metaFiles2/meta_{CATEGORY}.json.gz",
     f"meta_{CATEGORY}.json.gz"),
]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"exists: {dest}")
        return
    print(f"downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"wrote {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for url, name in FILES:
        download(url, os.path.join(OUT_DIR, name))
    print("done")
