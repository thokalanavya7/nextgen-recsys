# NexaShop — PPT Slide Content (paste-ready)

~14 slides for a final-year review/defense presentation. One block = one slide.
Text is short by design — keep slides sparse, talk the details.

---

## Slide 1 — Title
**Hybrid E-Commerce Recommendation System using Content-Based, Collaborative and Association-Rule Mining**
Your name, roll no, guide name, department, college, year.
(Small subtitle: NexaShop — deployed demo)

## Slide 2 — Introduction
- E-commerce catalogs are huge — users can't find relevant products.
- Recommenders boost engagement & sales by personalizing the catalog.
- Single-technique recommenders suffer: cold start (CF), over-specialization (CBF), sparsity.
- **Goal:** combine complementary techniques → better, explainable recommendations.

## Slide 3 — Objectives
- Build an end-to-end hybrid recommender on real Amazon data.
- Combine CBF + CF + Association Rules + SVD in a weighted model.
- Give each recommendation a human-readable *reason*.
- Real-time personalization from live clicks/ratings — no retraining.
- Evaluate honestly: ranking metrics + like-prediction accuracy.

## Slide 4 — Literature Survey (base papers)
| Paper | Method | Limitation |
|---|---|---|
| Three-Way Concept Lattices (e-commerce recsys) | concept lattices + content | no collaborative signal |
| Fashion Recommendation System | content/collaborative hybrid | domain-specific, no rules |
| Association Rule based Personalized RecSys | Apriori/FP-Growth rules | needs dense baskets |
| DL-based e-commerce recsys model | deep learning | heavy training, black-box |

→ **Gap:** no unified, explainable hybrid with real-time feedback. **Ours fills it.**

## Slide 5 — Existing vs Proposed
| Existing systems | Proposed system |
|---|---|
| Single technique | 6-signal weighted hybrid |
| Black-box output | reason for every item |
| Batch-only | live feedback updates recs instantly |
| Accuracy-only claims | transparent evaluation on real benchmark |

## Slide 6 — System Architecture
*(Paste your architecture diagram here — UI → Backend → Data Storage → Preprocessing → Hybrid Engine → Personalized Output)*

## Slide 7 — Modules
- **User Management** — signup/login, sessions
- **Product Management** — catalog, search, categories
- **Interaction Tracking** — view/click/rate/purchase events
- **Recommendation Service** — hybrid scoring + reasons
- **Preprocessing** — cleaning, k-core, feature extraction, user–item matrix

## Slide 8 — Hybrid Engine (one diagram + bullets)
- CBF: TF-IDF + cosine over product text
- CF: item–item cosine on rating matrix
- ARM: FP-Growth "bought together" rules (conf × lift)
- SVD: 64-factor latent model r̂ = μ + b_u + b_i + p_u·q_i
- Priors: popularity + review sentiment
- **score = Σ wᵢsᵢ → rank top-N**

## Slide 9 — Dataset
- Amazon Beauty 5-core (SNAP, McAuley et al. 2015)
- 198,502 reviews • 22,363 users • 12,101 products
- Metadata: title, category, brand, price, image
- Chronological leave-last-out split: 176k train / 22k test

## Slide 10 — Implementation / Tech Stack
- Python, FastAPI + Uvicorn backend, vanilla JS storefront
- scikit-learn, mlxtend, VADER, scipy sparse
- SQLite runtime store; joblib model artifacts
- Docker-ready; deployed demo

## Slide 11 — Results (chart)
- Hybrid MAP@10 = **0.0358** — best of all 6 models, +25% over best single
- Like-prediction: **78.9% accuracy**, F1 0.88, ROC-AUC 0.71
- *(Insert bar chart of MAP by model — use artifacts/evaluation.csv)*

## Slide 12 — Demo (screenshots)
- Storefront home, recommendation strip with reason chips,
  product modal (stars + buy), demo-user button
- Live URL / QR if deployed

## Slide 13 — Conclusion & Future Work
- Built & evaluated a full hybrid recsys on real data; hybrid beats every single model.
- Explainable recommendations via per-item reasons; real-time feedback loop.
- **Future:** neural CF, session-based recs, A/B testing, larger catalogs.

## Slide 14 — References
- McAuley et al., "Image-based recommendations on styles and substitutes", SIGIR 2015 (dataset)
- The 4 base papers + scikit-learn / mlxtend / FastAPI docs
