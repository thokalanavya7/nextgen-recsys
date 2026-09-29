# NexaShop — Next-Generation E-Commerce Recommendation System

A hybrid AI recommendation engine for e-commerce, built as a final-year
project. It combines **content-based filtering**, **item-item
collaborative filtering**, and **association-rule mining (FP-Growth)**
into a weighted hybrid scorer, with popularity and review-sentiment
priors for cold-start handling. A FastAPI backend exposes auth,
catalog, interaction tracking, and a recommendation service; a
storefront UI demonstrates real-time personalization.

## Architecture

```
User -> Web UI (browse/search/rate/buy)
      -> FastAPI backend (users, products, interaction tracking,
         recommendation service)
      -> SQLite (live events) + Amazon review corpus (trained model)
      -> preprocessing: cleaning, missing values, feature extraction,
         user-item matrix, review sentiment
      -> hybrid engine: CBF + CF + ARM -> weighted score -> Top-N
         with human-readable reasons
```

## Dataset

- **Interactions**: Amazon Product Data 2014, *Beauty* 5-core subset
  (198,502 reviews · 22,363 users · 12,101 items) — Stanford SNAP.
- **Product metadata**: Amazon metadata for the same ASINs
  (title, category, brand, price, image, description).

If you use this data in a paper, cite both the SNAP Amazon dataset and
Ni et al., EMNLP 2019 (see `docs/references.md`).

## Quickstart

```bash
pip install -r requirements.txt

python scripts/download_data.py     # fetch dataset (~90 MB)
python scripts/build_dataset.py     # clean, filter, split (~5-10 min)
python scripts/train.py             # fit models, write artifacts/ (~5-15 min)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — sign up (cold-start flow) or click
**"Try demo user"** to sign in as a real reviewer with history.

## Evaluation

```bash
python scripts/evaluate.py --k 10 --users 2000
```

Temporal per-user split; reports Precision@K, Recall@K, F1@K, MAP@K for
popularity / CBF / CF / ARM / hybrid, written to
`artifacts/evaluation.csv` for the results table of the paper.

## Deploy

See `Dockerfile` (single-container deployment of backend + UI + model
artifacts) and `docs/deployment.md` for Render/Fly.io/EC2 steps.

## Project layout

```
app/            FastAPI backend + storefront UI (static/)
app/recommender/  engine.py (hybrid), models.py (CBF/CF/ARM/popularity)
app/db.py       SQLite: users, sessions, live interactions
scripts/        download_data.py, build_dataset.py, train.py, evaluate.py
data/           raw + processed (gitignored)
artifacts/      trained model artifacts (gitignored)
docs/           references, deployment, paper notes
```
