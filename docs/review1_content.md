# NexaShop — Review-1 Content (all 7 parameters)

Paste-ready content for your report/PPT. Sections 5 & 6 are summarized here —
the full version with UML diagrams is in `docs/report_sections.md`.

---

## 1. Abstract

Online shoppers face an overwhelming product catalog, and single-technique
recommender systems struggle with cold start, data sparsity, and
over-specialization. This project implements **NexaShop**, a hybrid
e-commerce recommendation system that fuses six complementary signals —
TF-IDF content-based filtering, item–item collaborative filtering,
FP-Growth association-rule mining, biased SVD matrix factorization,
popularity and review sentiment — into a weighted score that ranks top-N
products for each user. Every recommendation carries a human-readable
reason, and live user feedback (clicks, ratings, purchases) updates
recommendations in real time without retraining. Evaluated on the Amazon
Beauty 5-core benchmark (198K reviews, 22K users, 12K products), the hybrid
achieves the best ranking quality of all models (MAP@10 = 0.0358, +25% over
the strongest single component) and predicts product likeability with 78.9%
accuracy (F1 0.88). A working storefront (FastAPI + web UI) demonstrates the
full pipeline end to end.

## 2. Existing System

- Most deployed recommenders use a **single technique**: pure collaborative
  filtering (Amazon-style "customers also bought"), pure content matching,
  or simple popularity trending lists.
- Weaknesses: CF fails on cold-start/sparse users; CBF over-specializes and
  cannot discover cross-category items; association-rule-only systems need
  dense purchase baskets; deep-learning approaches are accurate but opaque
  and expensive.
- Existing systems rarely **explain** why an item was recommended and rarely
  incorporate real-time feedback without full retraining.

## 3. Proposed System

- **Weighted hybrid** of six complementary signals — content 0.30,
  collaborative 0.30, association rules 0.15, SVD 0.05, popularity 0.10,
  sentiment 0.10 — so each component covers the others' blind spots.
- **Explainable output**: each recommended item shows the top reasons that
  drove its score (e.g. "Customers with similar taste rated X",
  "Frequently bought with Y", "Matches your interests in Hair Care").
- **Real-time personalization**: live events feed the user profile
  immediately — recommendations change on the next request, no retraining.
- Full-stack implementation: storefront UI → FastAPI backend → SQLite/CSV
  storage → preprocessing pipeline → hybrid engine.
- Honest, reproducible evaluation on a public benchmark dataset.

## 4. Literature Survey

| # | Paper | Method | Gap addressed |
|---|---|---|---|
| 1 | E-Commerce Personalized Recommendation System in Big Data Based on Three-Way Concept Lattices | three-way concept lattices over user/item features | no collaborative signal; heavy lattice construction |
| 2 | Enhancing E-Commerce Using Fashion Recommendation System | content + collaborative hybrid for fashion | domain-specific; no association rules or explanations |
| 3 | Association Rule Algorithm based Personalized Recommendation System for E-Commerce Platforms | Apriori/FP-Growth association rules | sparse data → few rules; single technique |
| 4 | Design and Application of Recommendation System Model for E-Commerce Platform Based on Deep Learning | deep-learning model | black-box, costly training, weak explainability |

**Research gap identified:** no existing approach unifies content,
collaborative, association-rule and latent-factor signals with per-item
explanations and a real-time feedback loop — the gap NexaShop fills.
(Dataset citation: McAuley et al., "Image-based recommendations on styles
and substitutes", SIGIR 2015.)

## 5. System Design

Layered architecture: **User → Web UI (login/search/browse/feedback) →
FastAPI Backend (user mgmt, product mgmt, interaction tracking,
recommendation service) → Data Storage (SQLite + CSV stores + model
artifacts) → Preprocessing (cleaning → k-core → feature extraction →
user–item matrix) → Hybrid Engine (CBF ‖ CF ‖ ARM ‖ SVD + priors → weighted
score → top-N) → Personalized Output (products + reasons) → feedback loop.**

UML diagrams (use case, class, sequence, activity, ER) with PlantUML code
ready to render: `docs/report_sections.md` §5.4.

## 6. Implementation

- **Stack:** Python 3.9+, FastAPI/Uvicorn, vanilla JS UI, SQLite, joblib;
  scikit-learn, mlxtend, VADER, scipy.
- **Pipeline:** `download_data.py → filter_meta.py → build_dataset.py →
  train.py`; models served via `RecommenderEngine` (uniform
  `score(user_items, item)` interface, weighted blend, reason generation).
- **API:** /api/signup, /api/login, /api/demo, /api/products,
  /api/interactions, /api/recommendations.
- **Formulas:** TF-IDF cosine; mean-centred item–item cosine (top-50
  neighbors); FP-Growth conf×lift (239 rules); r̂ = μ + b_u + b_i + p_u·q_i.
- Full module details: `docs/report_sections.md` §6.

## 7. Results

Dataset: Amazon Beauty 5-core — 198,215 interactions after preprocessing
(175,883 train / 22,332 test).

Ranking (leave-one-out, 2,000 users, K=10):

| Model | P@10 | R@10 | F1@10 | MAP@10 |
|---|---|---|---|---|
| Popularity | 0.0007 | 0.0070 | 0.0013 | 0.0015 |
| Content-based | 0.0070 | 0.0705 | 0.0128 | 0.0287 |
| Collaborative | 0.0009 | 0.0090 | 0.0016 | 0.0026 |
| Association rules | 0.0008 | 0.0075 | 0.0014 | 0.0047 |
| SVD | 0.0004 | 0.0045 | 0.0008 | 0.0010 |
| **Hybrid (ours)** | 0.0067 | 0.0665 | 0.0121 | **0.0358** |

Likeability prediction (held-out ratings, n=1,000): **accuracy 78.9%**,
F1 0.878, RMSE 1.10, ROC-AUC 0.705.

Takeaways for your viva: the hybrid beats every single model on MAP (+25%
over best single), sparse data explains the low absolute hit rate (normal
for this benchmark), and the ~79% like-prediction accuracy is the honest
headline number.
