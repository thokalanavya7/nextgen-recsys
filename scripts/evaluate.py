"""Offline evaluation: Precision@K, Recall@K, F1@K, MAP@K.

Protocol (matching the base paper's evaluation setup):
  * temporal split - each user's last interaction is held out (test.csv)
  * a hit counts when the held-out item appears in the model's top-K
  * models compared: popularity baseline, content-based only,
    collaborative only, association-rules only, and the full hybrid

Usage:
    python scripts/evaluate.py [--k 10] [--users 2000]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.recommender.engine import RecommenderEngine, DEFAULT_WEIGHTS


def average_precision(hit_ranks, k):
    """Single relevant item: AP = 1/rank if ranked within k else 0."""
    return 1.0 / hit_ranks if hit_ranks <= k else 0.0


def evaluate(engine, test_users, train_history, test_truth, k):
    variants = {
        "popularity": {"content": 0, "collaborative": 0, "association": 0,
                       "popularity": 1, "sentiment": 0},
        "content": {"content": 1, "collaborative": 0, "association": 0,
                    "popularity": 0, "sentiment": 0},
        "collaborative": {"content": 0, "collaborative": 1,
                          "association": 0, "popularity": 0, "sentiment": 0},
        "association": {"content": 0, "collaborative": 0,
                        "association": 1, "popularity": 0, "sentiment": 0},
        "hybrid": dict(DEFAULT_WEIGHTS),
    }
    metrics = {v: {"p": [], "r": [], "f1": [], "ap": []} for v in variants}
    for n, user in enumerate(test_users):
        hist = train_history[user]
        truth = test_truth[user]
        if truth in hist:
            continue
        for name, w in variants.items():
            recs = engine.recommend(hist, k=k, weights=w)
            ranked = [r["asin"] for r in recs]
            hit = 1.0 if truth in ranked else 0.0
            p, r = hit / k, hit              # single relevant item
            metrics[name]["p"].append(p)
            metrics[name]["r"].append(r)
            metrics[name]["f1"].append(
                2 * p * r / (p + r) if (p + r) else 0.0)
            rank = ranked.index(truth) + 1 if hit else k + 1
            metrics[name]["ap"].append(average_precision(rank, k))
        if (n + 1) % 200 == 0:
            print(f"  {n + 1}/{len(test_users)} users")
    rows = []
    for name, m in metrics.items():
        rows.append({
            "model": name,
            f"precision@{k}": round(float(np.mean(m["p"])), 4),
            f"recall@{k}": round(float(np.mean(m["r"])), 4),
            f"f1@{k}": round(float(np.mean(m["f1"])), 4),
            f"map@{k}": round(float(np.mean(m["ap"])), 4),
            "n_users": len(m["p"]),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--users", type=int, default=2000)
    args = ap.parse_args()

    proc = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    train = pd.read_csv(os.path.join(proc, "train.csv"))
    test = pd.read_csv(os.path.join(proc, "test.csv"))
    train_history = {u: dict(zip(g["asin"], g["rating"]))
                     for u, g in train.groupby("user")}
    test_truth = dict(zip(test["user"], test["asin"]))
    users = [u for u in test_truth if u in train_history]
    rng = np.random.default_rng(7)
    if len(users) > args.users:
        users = list(rng.choice(users, args.users, replace=False))
    print(f"evaluating {len(users)} users @ k={args.k}")

    engine = RecommenderEngine().load()
    t0 = time.time()
    table = evaluate(engine, users, train_history, test_truth, args.k)
    print(table.to_string(index=False))
    print(f"elapsed: {time.time() - t0:.1f}s")
    out = os.path.join(os.path.dirname(__file__), "..", "artifacts",
                       "evaluation.csv")
    table.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
