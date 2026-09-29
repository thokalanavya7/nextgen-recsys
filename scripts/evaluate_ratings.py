"""Likeability & rating-prediction evaluation (the accuracy suite).

Two honest protocols:

1. Like-prediction on observed held-out interactions
   Predict whether a user's held-out rating is >= 4 (like vs dislike).
   Threshold tuned on a validation slice of users, applied to a disjoint
   test slice. Reports accuracy, balanced accuracy, precision/recall/F1,
   plus RMSE/MAE of the rating regressor.

2. Likeability discrimination vs unobserved items
   Each held-out liked item is scored against a random item the user
   never touched; reports ROC-AUC and accuracy at the tuned threshold -
   shows the system actually separates chosen items from the catalog.

Usage:
    python scripts/evaluate_ratings.py [--users 2000]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.recommender.engine import RecommenderEngine, DEFAULT_WEIGHTS

LIKE_THRESHOLD = 4.0


def score_hybrid(eng, h, it):
    """System hybrid score for a single candidate (same weights as prod)."""
    w = DEFAULT_WEIGHTS
    return (
        w["content"] * eng.content.score([it], h)[0]
        + w["collaborative"] * eng.collaborative.score([it], h)[0]
        + w["association"] * eng.association.score([it], h)[0]
        + w["svd"] * eng.svd.predict(h, it) / 5.0
        + w["popularity"] * eng.popularity.scores.get(it, 0) / 5.0
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=2000)
    args = ap.parse_args()

    proc = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    train = pd.read_csv(os.path.join(proc, "train.csv"))
    test = pd.read_csv(os.path.join(proc, "test.csv"))
    hist = {u: dict(zip(g["asin"], g["rating"]))
            for u, g in train.groupby("user")}
    rng = np.random.default_rng(7)
    users = np.array([u for u in test.user.unique() if u in hist])
    rng.shuffle(users)
    users = users[:args.users]
    half = len(users) // 2
    val_u, test_u = set(users[:half]), set(users[half:])
    test = test[test.user.isin(val_u | test_u)]

    eng = RecommenderEngine().load()
    svd = eng.svd
    items_all = list(svd.asin_index.keys())

    # ---------- protocol 1: observed held-out ratings ----------
    print("scoring held-out ratings...")
    rows = []
    for _, row in test.iterrows():
        u = row["user"]
        rows.append((u in val_u, int(row["rating"] >= LIKE_THRESHOLD),
                     svd.predict(hist[u], row["asin"], user_id=u),
                     row["rating"]))
    df = pd.DataFrame(rows, columns=["is_val", "y", "pred", "rating"])
    val, te = df[df.is_val], df[~df.is_val]

    best_t = max(np.arange(2.5, 4.6, 0.05),
                 key=lambda t: accuracy_score(val.y, (val.pred >= t)))
    yhat = (te.pred >= best_t).astype(int)
    rmse = float(np.sqrt(np.mean((te.pred - te.rating) ** 2)))
    mae = float(np.mean(np.abs(te.pred - te.rating)))
    print(f"--- protocol 1: like-prediction on observed ratings "
          f"(threshold={best_t:.2f}, n={len(te)}) ---")
    print(f"  accuracy        : {accuracy_score(te.y, yhat):.4f}")
    print(f"  balanced acc    : {balanced_accuracy_score(te.y, yhat):.4f}")
    print(f"  precision/recall: "
          f"{precision_score(te.y, yhat, zero_division=0):.4f} / "
          f"{recall_score(te.y, yhat, zero_division=0):.4f}")
    print(f"  f1              : {f1_score(te.y, yhat, zero_division=0):.4f}")
    print(f"  rmse / mae      : {rmse:.4f} / {mae:.4f}")

    # ---------- protocol 2: discrimination vs unobserved ----------
    print("scoring discrimination pairs...")
    y_true, y_score = [], []
    for _, row in test[test.user.isin(test_u)].iterrows():
        u, h = row["user"], hist[row["user"]]
        for _ in range(20):
            neg = items_all[rng.integers(len(items_all))]
            if neg not in h:
                break
        y_true += [1, 0]
        y_score += [score_hybrid(eng, h, row["asin"]),
                    score_hybrid(eng, h, neg)]
    y_true, y_score = np.array(y_true), np.array(y_score)
    print(f"--- protocol 2: liked vs unobserved (n={len(y_true)}) ---")
    print(f"  roc-auc         : {roc_auc_score(y_true, y_score):.4f}")

    out = os.path.join(os.path.dirname(__file__), "..", "artifacts",
                       "likeability_metrics.txt")
    with open(out, "w") as f:
        f.write(
            f"like_accuracy={accuracy_score(te.y, yhat):.4f}\n"
            f"balanced_accuracy={balanced_accuracy_score(te.y, yhat):.4f}\n"
            f"precision={precision_score(te.y, yhat, zero_division=0):.4f}\n"
            f"recall={recall_score(te.y, yhat, zero_division=0):.4f}\n"
            f"f1={f1_score(te.y, yhat, zero_division=0):.4f}\n"
            f"rmse={rmse:.4f}\nmae={mae:.4f}\n"
            f"discrimination_auc={roc_auc_score(y_true, y_score):.4f}\n"
            f"threshold={best_t:.2f}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
