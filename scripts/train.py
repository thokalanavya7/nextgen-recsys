"""Train the recommender components and persist artifacts.

Run after build_dataset.py:
    python scripts/train.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.recommender import RecommenderEngine

if __name__ == "__main__":
    RecommenderEngine().train()
    print("artifacts written")
