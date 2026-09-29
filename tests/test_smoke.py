"""Smoke test: fit tiny synthetic data through the full engine."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.recommender.engine import RecommenderEngine


def tiny_engine():
    rng = np.random.default_rng(0)
    items = [f"I{i:03d}" for i in range(60)]
    users = [f"U{u:03d}" for u in range(40)]
    rows = []
    for u in users:
        for it in rng.choice(items, 8, replace=False):
            rows.append({"user": u, "asin": it,
                         "rating": float(rng.integers(1, 6)),
                         "ts": int(rng.integers(1, 1000))})
    train = pd.DataFrame(rows)
    docs = pd.DataFrame({
        "asin": items,
        "doc": [f"product {i} beauty skincare lotion cream" for i in range(60)],
    })
    prods = pd.DataFrame({
        "asin": items, "title": [f"Product {i}" for i in range(60)],
        "category": ["Beauty"] * 60, "item_sentiment": np.zeros(60),
    })
    eng = RecommenderEngine()
    eng._fit_all(train, docs, prods)
    return eng, train


def test_recommend_shape():
    eng, train = tiny_engine()
    hist = dict(train[train.user == "U000"][["asin", "rating"]].values)
    recs = eng.recommend(hist, k=10)
    assert len(recs) == 10
    assert all("asin" in r and "score" in r and "reasons" in r for r in recs)
    seen = set(hist)
    assert not any(r["asin"] in seen for r in recs)


def test_cold_start_returns_popular():
    eng, _ = tiny_engine()
    recs = eng.recommend({}, k=5)
    assert len(recs) == 5


def test_component_variants_dont_crash():
    eng, train = tiny_engine()
    hist = dict(train[train.user == "U001"][["asin", "rating"]].values)
    for w in ({"content": 1, "collaborative": 0, "association": 0,
               "popularity": 0, "sentiment": 0},
              {"content": 0, "collaborative": 1, "association": 0,
               "popularity": 0, "sentiment": 0},
              {"content": 0, "collaborative": 0, "association": 1,
               "popularity": 0, "sentiment": 0}):
        assert len(eng.recommend(hist, k=5, weights=w)) >= 1
