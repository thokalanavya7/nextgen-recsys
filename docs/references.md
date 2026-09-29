# References & originality notes

This project is an **original implementation**. It reuses only published
ideas (algorithms), which are cited below — the code, UI, system design,
and the specific hybrid weighting scheme were written from scratch for
this project.

## Base papers (provided)

1. J. Liu and Q. Zhang, "Association Rule Algorithm based Personalized
   Recommendation System for E-Commerce Platforms," *ICDSIS 2025*.
   → association-rule component + evaluation protocol
   (Precision@K / Recall@K / F1@K / MAP).
2. X. Yu and J. Aviles, "Design and Application of Recommendation System
   Model for E-Commerce Platform Based on Deep Learning Algorithm," 2024.
   → motivation for the hybrid direction (their CNN+LSTM+attention is a
   separate approach; ours is non-deep-learning and interpretable).
3. H. Xu and J. Gu, "E-Commerce Personalized Recommendation System in
   Big Data Based on Three-Way Concept Lattices," 2025.
   → collaborative-filtering formulation and sparsity motivation.
4. S. Vinodh Kumar, P. Kumar, S. Saravanan, "Enhancing E-Commerce using
   Fashion Recommendation System," 2024.
   → user/admin module split and real-time suggestion UX.

## Dataset citation (required in any publication)

- J. McAuley, C. Targett, Q. Shi, A. van den Hengel, "Image-Based
  Recommendations on Styles and Substitutes," *SIGIR 2015*.
  (Amazon Product Data 2014, Beauty 5-core)
- J. Ni, J. Li, J. McAuley, "Justifying Recommendations using
  Distantly-Labeled Reviews and Fine-Grained Aspects," *EMNLP 2019*.
  (Amazon Reviews 2018 metadata)

## What is novel in this implementation (talking points for the paper)

- Weighted hybrid score fusing five signals (text similarity,
  item-item CF, FP-Growth rules, Bayesian popularity, review sentiment)
  with per-recommendation **explanation generation** ("frequently
  bought with X", "matches your interest in skincare").
- A **real-time personalization layer**: live click/view/rate/purchase
  events are merged with the trained profile on every request — no
  model retraining needed.
- End-to-end deployable system: API + storefront + offline evaluation
  harness, not only an offline experiment.
