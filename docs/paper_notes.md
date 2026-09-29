# Notes for the conference paper

Map your architecture diagram to this implementation:

| Architecture box | Code |
|---|---|
| User Interface | `app/static/` (login, search, browse, product view, feedback) |
| Application/Backend | `app/main.py`, `app/db.py` (users, products, interaction tracking, recommendation service) |
| Data Storage | SQLite (`app.db`) + Amazon corpus in `data/` |
| Data Preprocessing | `scripts/build_dataset.py` (cleaning, missing values, feature extraction, user-item matrix, sentiment) |
| Content-Based Filtering | `ContentBasedModel` in `app/recommender/models.py` (TF-IDF + cosine) |
| Collaborative Filtering | `CollaborativeModel` (mean-centred item-item cosine, top-50 neighbours) |
| Association Rule Mining | `AssociationRuleModel` (FP-Growth via mlxtend, confidence×lift scoring) |
| Hybrid/Weighted Score | `RecommenderEngine.recommend` in `app/recommender/engine.py` |
| Personalized Output | top-N with per-item `reasons` explanations |

## Headline results (artifacts/evaluation.csv)

Leave-one-out temporal split, 2,000 sampled users, K=10:

| Model | P@10 | R@10 | F1@10 | MAP@10 |
|---|---|---|---|---|
| Popularity | 0.0007 | 0.0070 | 0.0013 | 0.0015 |
| Content-based | 0.0070 | 0.0705 | 0.0128 | 0.0287 |
| Collaborative | 0.0009 | 0.0090 | 0.0016 | 0.0026 |
| Association rules | 0.0008 | 0.0075 | 0.0014 | 0.0047 |
| SVD (matrix factorisation) | 0.0004 | 0.0045 | 0.0008 | 0.0010 |
| **Hybrid (ours)** | 0.0067 | 0.0665 | 0.0121 | **0.0358** |

Hybrid achieves the best MAP (+25% over the strongest single component)
— the components are complementary: CF/ARM/SVD alone are weak on sparse
data but add ranking signal that CBF misses.

## Accuracy suite (artifacts/likeability_metrics.txt)

Like-prediction on held-out observed ratings (threshold tuned on a
disjoint validation slice, n=1,000):

- **Accuracy 78.9%**, F1 0.878, precision 0.800, recall 0.974
- Rating regression: RMSE 1.10, MAE 0.82
- Liked-vs-unobserved discrimination: ROC-AUC 0.705 (hybrid score)

Present these as: *"the system predicts whether a user will like a
product with ~79% accuracy (F1 0.88)".* Do NOT inflate further — ~79%
on sparse real-world data is a defensible number; claims of 90%+ on
this benchmark are a known red flag for reviewers.

## Anti-plagiarism checklist

- All code was written for this project; algorithms are standard
  (TF-IDF, cosine similarity, FP-Growth, Bayesian average) and should be
  cited via the base papers / canonical sources, not presented as novel
  algorithms.
- The *contribution* you can claim: the five-signal weighted fusion,
  the explanation layer, the real-time profile update, and the
  deployable end-to-end system.
- In the paper, describe the method in your own words and cite:
  Liu & Zhang (ARM/eval protocol), Yu & Aviles (deep-learning
  comparator), Xu & Gu (CF formulation), Vinodh Kumar et al. (system
  architecture), plus McAuley et al. (dataset). See `docs/references.md`.
- Run your manuscript through your institute's plagiarism checker
  (Turnitin/iThenticate) — code and architecture are original, so
  similarity should only match standard definitions if quoted.

## Suggested ablations / extra experiments for a stronger paper

1. `python scripts/evaluate.py --k 5` and `--k 20` for a metrics-vs-K table.
2. Cold-start analysis: restrict test users to <5 train items and show
   popularity+sentiment+content coverage vs pure CF.
3. Weight sensitivity: edit `DEFAULT_WEIGHTS` — produces a nice figure.
4. Latency: time `/api/recommendations` (~ms) to support the
   "real-time" claim.
