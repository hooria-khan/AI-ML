# AI/ML Engineering — Advanced Internship Tasks: Solutions (3 of 5)

DevelopersHub Corporation — submitted tasks: **Task 2, Task 3, Task 5**

## Why these 3, and an important caveat up front

Tasks 1 (BERT fine-tuning) and 4 (LangChain/RAG chatbot) both fundamentally require live network
access to Hugging Face Hub and/or an LLM API to download pretrained weights or generate text.
The environment these notebooks were built and run in has **no internet access** to those
services (only PyPI/npm/GitHub are reachable, no GPU). Tasks 2, 3, and 5 were chosen because they
could be built as **genuine, fully-executed, end-to-end pipelines** rather than pseudocode.

Where a real dataset or a real model wasn't reachable, each notebook:
1. Says so explicitly, up front, in a markdown cell.
2. Substitutes something as close to the real thing as possible (a synthetic dataset with the
   same schema and realistic relationships; a hand-built mini-CNN instead of a pretrained one; a
   classical NLP pipeline plus the actual LLM prompts you'd use once API access exists).
3. Spells out exactly which single cell/line to change to point the same pipeline at the real
   dataset or a real model/API.

Every notebook was actually run top-to-bottom (`jupyter nbconvert --execute`) before being
handed over — the printed numbers are real outputs, not invented ones.

---

## Task 2 — End-to-End ML Pipeline (Customer Churn)

**Objective:** reusable, production-ready pipeline predicting customer churn.

**Approach:** synthetic Telco-schema dataset (2,000 rows) generated so churn probability
genuinely depends on `Contract`, `tenure`, and `MonthlyCharges` — the same signal a real churn
dataset has. `ColumnTransformer` (scaling + one-hot encoding) feeds into a `Pipeline`, tuned with
`GridSearchCV` across both Logistic Regression and Random Forest in a single search. Best pipeline
exported with `joblib`.

**Result:** best model selected by 5-fold CV was Logistic Regression (C=1); test accuracy 0.66,
F1 0.575 (see notebook for full classification report + confusion matrix).

**File:** `Task2_Churn_Pipeline.ipynb`, `churn_pipeline.joblib`, `confusion_matrix.png`

---

## Task 3 — Multimodal ML (Housing Price: Images + Tabular)

**Objective:** predict house prices from both tabular features and images.

**Approach:** synthetic tabular data (sqft, bedrooms, bathrooms, age, location score) drives a
ground-truth price; synthetic images are procedurally generated to correlate with price tier
(brighter/more colorful ≈ pricier), giving the image channel real signal to learn from. A
hand-built 3-filter convolution + pooling step (edge/blur kernels via `scipy.signal.convolve2d`)
stands in for a pretrained CNN. Tabular + image features are concatenated and fed to a
`RandomForestRegressor`.

**Result:** fused model MAE ≈ $5,400 / RMSE ≈ $7,150, vs. a tabular-only baseline of MAE ≈
$21,600 / RMSE ≈ $27,800. (The gap is larger than you'd see on real photos, since the synthetic
images were deliberately built to correlate strongly with price — real house photos carry a
noisier, weaker signal.)

**File:** `Task3_Multimodal_Housing.ipynb`, `sample_houses.png`, `pred_vs_actual.png`

---

## Task 4 (not submitted) — Context-Aware Chatbot

Skipped: RAG generation quality depends on an actual LLM call, which this sandbox can't make.
A TF-IDF retrieval layer could be built without one, but a chatbot with no generation model
behind it wouldn't actually demonstrate the task.

---

## Task 5 — Auto-Tagging Support Tickets

**Objective:** tag free-text tickets by category, comparing zero-shot vs. fine-tuned/few-shot,
outputting top-3 tags per ticket.

**Approach:** 100 synthetic tickets across 5 categories (Billing, Technical, Account, Feature
Request, Cancellation). The actual zero-shot and few-shot **prompts** you'd send to a real LLM
are written out in full. Since no LLM API is reachable here, TF-IDF cosine-similarity against
category descriptions plays the zero-shot role (no labeled examples used), and a
`LogisticRegression` classifier trained on a labeled split plays the fine-tuned/few-shot role.
Both output ranked top-3 predictions.

**Result:** zero-shot top-1 accuracy 0.33 / top-3 accuracy 0.60; trained top-1 accuracy 1.00 /
top-3 accuracy 1.00 — the trained model wins clearly, as expected, since only it ever sees
labeled examples of each category's vocabulary.

**File:** `Task5_Auto_Tagging_Tickets.ipynb`

---

## Repository checklist (per the assignment's submission requirements)

- [x] Problem statement & objective — top markdown cell of each notebook
- [x] Dataset loading/generation & preprocessing — Steps 1–2 of each notebook
- [x] Model development & training — subsequent steps
- [x] Evaluation with relevant metrics — accuracy/F1 (Task 2), MAE/RMSE (Task 3), top-1/top-3
      accuracy (Task 5)
- [x] Visualizations — confusion matrix, sample images, predicted-vs-actual scatter
- [x] Final summary/insights — last markdown cell of each notebook
- [ ] Push to GitHub with a README (this file) and submit the repo link via Google Classroom —
      that upload step is on you, since it needs your own GitHub account.
