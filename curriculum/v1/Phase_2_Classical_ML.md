# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: CLASSICAL ML CRASH COURSE (Weeks 7–9)
# ═══════════════════════════════════════════════════════════════════════
# Zero to AI Engineer — 9-Month Masterclass
# ═══════════════════════════════════════════════════════════════════════

---

## 🎯 Why a Whole Phase on Classical ML?

Because every neural-network debugging technique — train/val/test discipline, bias-variance reasoning, regularization, the right choice of metric, proper cross-validation, error analysis — was invented and refined in the classical-ML era. Jumping from "I know calculus" straight to "here's a Transformer" is how people get stuck with models that train but don't *work*.

Three weeks here will pay you back 10× in every subsequent phase.

## 🔗 The Salesforce Analogy

> Before you debug a 30-node **Flow** in production, you'd get fluent with a `Process Builder` 3-step formula. Classical ML is the `Process Builder` of AI: less powerful, but the concepts transfer perfectly to deep learning. Linear regression is the "Hello World" that every neural net is a generalization of.

## 🗺️ Where This Phase Fits

```
Phase 1 (math)  →  Phase 2 (classical ML: YOU ARE HERE)  →  Phase 3 (neural nets)
    ↑                        ↑                                      ↑
"the tools"           "the workflows"                      "the deeper architectures"
```

## 📖 The Six Concepts You Must Internalize This Phase

1. **Parametric model fitting by gradient descent** — the exact workflow you'll reuse for every neural net.
2. **Bias–Variance tradeoff** — why a model can "memorize training data but fail on test data," and why this happens to neural nets too.
3. **Train / Validation / Test discipline + Cross-Validation** — the single most-cheated-on skill in applied ML.
4. **The right metric for the problem** — accuracy, precision, recall, F1, ROC-AUC, MAE, MAPE. Pick wrong → your model optimizes the wrong thing.
5. **Regularization** (L1, L2, early stopping) — all the same tricks you'll use in NNs, just named differently.
6. **Error analysis** — reading misclassified examples as *data*, not as bugs. This is the skill that separates senior ML engineers from juniors.

---

# WEEK 7 — Linear & Logistic Regression from Scratch

> The goal this week is to *implement* linear and logistic regression **from scratch with NumPy using gradient descent** (you already have the autograd intuition from Phase 1). Then reproduce the exact same fits using `scikit-learn` and confirm numerical agreement. This one-two punch ("build it, then verify it matches the library") is the core pattern of this entire course.

## 📚 Resources

| # | Resource | Duration | Why |
|---|----------|----------|-----|
| 1 | [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo) | 27 min | Crystal-clear intuition |
| 2 | [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8) | 9 min | The sigmoid origin story |
| 3 | [3B1B: Gradient descent, how NNs learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) | 21 min | Review if rusty |
| 4 | Ch. 4 of [Hands-On ML (Géron)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) | ~90 min read | THE classical ML chapter |
| 5 | [sklearn docs: Linear models](https://scikit-learn.org/stable/modules/linear_model.html) | reference | Production API |

## 🧮 The Math in One Page

**Linear regression.** Given `(xᵢ, yᵢ)` with `xᵢ ∈ Rᵈ`, find `w, b` minimizing

```
L(w, b) = (1/n) Σ (yᵢ − (w·xᵢ + b))²
```

Gradients:
```
∂L/∂w = −(2/n) Σ xᵢ (yᵢ − ŷᵢ)
∂L/∂b = −(2/n) Σ     (yᵢ − ŷᵢ)
```

MLE interpretation (Phase 1 W5 callback!): this *is* MLE under `yᵢ ~ N(w·xᵢ + b, σ²)`.

**Logistic regression.** For binary classification, put a `sigmoid` on top:

```
σ(z) = 1 / (1 + e^(−z));   p(y=1 | x) = σ(w·x + b)
```

Loss is **Binary Cross-Entropy** (a.k.a. NLL under Bernoulli — see Phase 1 W6):

```
L = −(1/n) Σ [ yᵢ log ŷᵢ + (1 − yᵢ) log(1 − ŷᵢ) ]
```

Gradients (neat fact you should derive by hand):
```
∂L/∂w = (1/n) Σ xᵢ (ŷᵢ − yᵢ)       ← exactly the linear-regression gradient!
∂L/∂b = (1/n) Σ     (ŷᵢ − yᵢ)
```

The residual `(ŷᵢ − yᵢ)` appearing in both is not a coincidence. It's a property of the **exponential family** — something you'll see again when you study softmax + cross-entropy in Phase 5.

## 💻 Exercise 7.1 — Build `LinearRegressor` From Scratch

```python
# src/classical/linear.py
import numpy as np

class LinearRegressor:
    """Multi-variate linear regression via batch gradient descent."""

    def __init__(self, lr: float = 0.01, n_iter: int = 1000, l2: float = 0.0):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
        self.w = None
        self.b = 0.0
        self.history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        n, d = X.shape
        self.w = np.zeros(d)
        for _ in range(self.n_iter):
            y_hat = X @ self.w + self.b
            residual = y_hat - y
            grad_w = (2 / n) * X.T @ residual + 2 * self.l2 * self.w
            grad_b = (2 / n) * residual.sum()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            self.history.append(float(np.mean(residual ** 2)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b
```

**Test:** generate `y = 2x₁ − 3x₂ + 1 + noise`. Your fit should recover `w ≈ [2, −3]`, `b ≈ 1`. Compare with `sklearn.linear_model.LinearRegression` — the two should agree to 3 decimals.

## 💻 Exercise 7.2 — Build `LogisticRegressor` From Scratch

```python
class LogisticRegressor:
    def __init__(self, lr=0.1, n_iter=2000, l2=0.0):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
        self.w = None; self.b = 0.0

    @staticmethod
    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        for _ in range(self.n_iter):
            p = self._sigmoid(X @ self.w + self.b)
            grad_w = (X.T @ (p - y)) / n + 2 * self.l2 * self.w
            grad_b = (p - y).mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X): return self._sigmoid(X @ self.w + self.b)
    def predict(self, X, threshold=0.5): return (self.predict_proba(X) > threshold).astype(int)
```

**Test on the Breast Cancer dataset:**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X, y = load_breast_cancer(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegressor(lr=0.1, n_iter=3000).fit(X_tr, y_tr)
acc = (model.predict(X_te) == y_te).mean()
print(f"My logreg test acc: {acc:.3f}")   # expect ~0.97
```

Then compare with `sklearn.linear_model.LogisticRegression` — results within ~1%.

## 📅 Week 7 Schedule

| Day | Focus | Deliverable |
|-----|-------|-------------|
| Mon | Theory (videos 1–3) + derive gradients by hand | Journal page with derivation |
| Tue | Build `LinearRegressor` (Ex 7.1) | `src/classical/linear.py` + 5 unit tests |
| Wed | Build `LogisticRegressor` (Ex 7.2) on Breast Cancer | `src/classical/logistic.py` + tests |
| Thu | L1 vs L2 regularization; add to your classes | Plot weight shrinkage as λ varies |
| Fri | Sklearn parity check + blog post "I re-implemented logistic regression" | Blog post committed |
| Sat | **Bias-variance polynomial demo** (high-degree polyfit overfit visualization) | `projects/05-bias-variance-demo/` |
| Sun | Retrospective + paper-of-the-week | Progress bar, retro notes |

## ✅ Week 7 Success Criteria

- [ ] `LinearRegressor` + `LogisticRegressor` numerically match sklearn within 1%
- [ ] Can derive logistic regression gradient by hand on paper
- [ ] L1 (Lasso) and L2 (Ridge) regularization implemented and visualized
- [ ] Understand that BCE = NLL under Bernoulli (callback to Phase 1 W6)

---

# WEEK 8 — Trees, Boosting, Metrics, and Cross-Validation

> This is the most "engineering" week of Phase 2. You'll use XGBoost as-is (no re-implementation — it's a software engineering marvel, not an ML one) and instead spend the time on the *workflows* that make models actually work: cross-validation, metric selection, hyperparameter search, and error analysis.

## 📚 Resources

| # | Resource | Duration | Why |
|---|----------|----------|-----|
| 1 | [StatQuest: Decision Trees](https://www.youtube.com/watch?v=_L39rN6gz7Y) | 17 min | Tree intuition |
| 2 | [StatQuest: Gradient Boost Part 1](https://www.youtube.com/watch?v=3CC4N4z3GJc) | 15 min | The boosting idea |
| 3 | [StatQuest: ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM) | 16 min | The metric you'll see most in interviews |
| 4 | [Kaggle: 30 Days of ML (Intermediate)](https://www.kaggle.com/learn) | ~5 hrs | Hands-on metric + CV practice |
| 5 | [sklearn docs: Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) | reference | Metric encyclopedia |

## 📖 The Metrics Zoo (Memorize This Table)

| Problem | Primary Metric | When It Lies | Better Alternative |
|---------|----------------|--------------|--------------------|
| Balanced binary classification | Accuracy | When classes are imbalanced | F1, ROC-AUC |
| Imbalanced binary classification | **F1** or **PR-AUC** | Accuracy says 99% because it predicts "no" always | — |
| Multi-class classification | Macro-F1 | Class imbalance hidden by "micro" avg | Per-class F1 report |
| Ranking (recsys, search) | NDCG, MRR | Accuracy / F1 don't account for rank | — |
| Regression | MAE, RMSE | R² if target has outliers | MAPE if scale-relative |
| Calibration | ECE (Expected Calibration Error) | Low loss ≠ well-calibrated probabilities | — |

### Confusion Matrix — Know These From Memory

```
                 Predicted
                 POS    NEG
Actual POS      [TP]   [FN]   ← recall = TP/(TP+FN)  "of the positives, how many did I catch?"
Actual NEG      [FP]   [TN]
                 ↑
         precision = TP/(TP+FP)  "of my positive predictions, how many were right?"
```

- **Fraud detection** → optimize recall (don't miss fraud)
- **Spam filter** → optimize precision (don't block real email)
- **Medical screening** → optimize recall (don't miss disease) at the front, precision at confirmatory test
- **General balanced case** → F1 (harmonic mean of P and R)

### Cross-Validation Patterns

| Pattern | Use When |
|---------|----------|
| Hold-out (80/20) | Quick sanity check |
| K-Fold (k=5 or 10) | Default for tabular data |
| Stratified K-Fold | Classification with class imbalance |
| **Time-series split** | ANY temporal data — *leakage trap* otherwise |
| Group K-Fold | Same user/entity shouldn't appear in both train and val |
| Nested CV | When you also tune hyperparameters |

> **The leakage trap:** shuffling a time-series dataset and doing random CV will make your model look magical in validation and terrible in production. Always think: *does my split respect the arrow of time / grouping / source*? This mistake has killed more ML projects than bad models.

## 💻 Exercise 8.1 — Metrics From Scratch

```python
# src/classical/metrics.py
import numpy as np

def confusion_matrix(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

def precision(y_true, y_pred):
    c = confusion_matrix(y_true, y_pred)
    return c["tp"] / max(c["tp"] + c["fp"], 1)

def recall(y_true, y_pred):
    c = confusion_matrix(y_true, y_pred)
    return c["tp"] / max(c["tp"] + c["fn"], 1)

def f1(y_true, y_pred):
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / max(p + r, 1e-12)

def roc_curve(y_true, y_score):
    thresholds = np.sort(np.unique(y_score))[::-1]
    tpr, fpr = [], []
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        c = confusion_matrix(y_true, yp)
        tpr.append(c["tp"] / max(c["tp"] + c["fn"], 1))
        fpr.append(c["fp"] / max(c["fp"] + c["tn"], 1))
    return np.array(fpr), np.array(tpr)

def roc_auc(y_true, y_score):
    fpr, tpr = roc_curve(y_true, y_score)
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))
```

Unit-test against `sklearn.metrics` — should match to 1e-6 on the same inputs.

## 💻 Exercise 8.2 — Titanic Kaggle (Project #4)

```python
# projects/04-titanic-kaggle/run.py — sketch
import pandas as pd, xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("train.csv")
y = df["Survived"]; X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket"])

numeric = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
categorical = ["Sex", "Embarked", "Cabin"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    # leave numerics passthrough; impute missing ages with median upstream
])

model = Pipeline([("pre", pre), ("xgb", xgb.XGBClassifier(n_estimators=300, max_depth=4))])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring="f1")
print(f"F1 across folds: {scores.mean():.3f} ± {scores.std():.3f}")
```

**Target:** top 40% on the Kaggle public leaderboard. Nothing heroic — this is about process discipline, not leaderboard chasing.

## 📅 Week 8 Schedule

| Day | Focus | Deliverable |
|-----|-------|-------------|
| Mon | Videos + the metrics table above | Memorize the confusion matrix |
| Tue | Ex 8.1: metrics from scratch | `src/classical/metrics.py` with unit tests |
| Wed | XGBoost on Titanic (first pass) | Baseline submission on Kaggle |
| Thu | Cross-validation + hyperparameter search | Proper K-fold CV pipeline |
| Fri | Error analysis: read your 20 worst predictions | Blog post: "What my Titanic model got wrong" |
| Sat | **Kaggle refinement** (Project #4) | Top-40% submission |
| Sun | Retro + paper-of-the-week | Progress update |

## ✅ Week 8 Success Criteria

- [ ] Can draw the confusion matrix and all four derived metrics from memory
- [ ] Can pick the right metric for 5 different problem framings
- [ ] Implemented `precision`, `recall`, `f1`, `roc_auc` from scratch, match sklearn
- [ ] Titanic Kaggle submission in top 40%
- [ ] Error analysis notebook committed — identifies ≥3 failure categories

---

# WEEK 9 — Data Hygiene, Leakage, and the "Spam Classifier Ladder"

> This week you'll build **the same spam classifier three times**: once with logistic regression on TF-IDF features, once with a small MLP, and once with a pre-trained embedding model (`sentence-transformers`) + logistic regression. Same data, same metric, three levels of "sophistication." This will make the rest of the course (and the Phase 3 "why bother with NNs" question) land viscerally.

## 📚 Resources

| # | Resource | Duration | Why |
|---|----------|----------|-----|
| 1 | [Chip Huyen: Data Leakage](https://huyenchip.com/2021/11/03/data-leakage.html) | 15 min read | Best single read on leakage |
| 2 | [Google ML Engineering: Data Prep](https://developers.google.com/machine-learning/data-prep) | ~60 min | Production discipline |
| 3 | [scikit-learn: text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) | reference | TF-IDF & CountVectorizer |
| 4 | [Sentence-Transformers quickstart](https://www.sbert.net/docs/quickstart.html) | 15 min | Plug-and-play embeddings |

## 📖 Leakage Patterns to Learn (All of These Have Sunk Real Projects)

1. **Time leakage** — using future information to predict the past
2. **Group leakage** — same user in train and test
3. **Preprocessing leakage** — fitting a scaler/imputer on the full dataset before splitting
4. **Target leakage** — a feature that is computed *from* the target (e.g., "days since last purchase" predicting "made a purchase")
5. **Duplicate leakage** — near-duplicate rows ending up split across train/test

Rule of thumb: **any preprocessing that looks at `y` must be inside the CV loop.**

## 💻 Exercise 9.1 — The Spam Classifier Ladder (Project #5)

```python
# projects/05-spam-classifier/run.py — three models, one dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score

# Use SMS Spam Collection from UCI or Kaggle — load into texts, labels
# (1) TF-IDF + Logistic Regression
vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_tfidf = vec.fit_transform(texts)                 # leakage-safe via pipeline in reality
logreg = LogisticRegression(max_iter=1000).fit(X_tfidf_train, y_train)

# (2) TF-IDF + small MLP (sklearn)
mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=30).fit(X_tfidf_train, y_train)

# (3) Sentence-Transformer embeddings + Logistic Regression
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer("all-MiniLM-L6-v2")
X_emb = enc.encode(texts, show_progress_bar=True)
logreg_emb = LogisticRegression(max_iter=1000).fit(X_emb_train, y_train)

# Compare F1 across all three. Document findings.
```

The punchline you want to *feel* by Friday:

- Model 1 (TF-IDF + logreg): very high F1 — surprisingly hard to beat
- Model 2 (TF-IDF + MLP): marginal lift at best
- Model 3 (Embeddings + logreg): dominates on out-of-distribution variants (new phrasings) because embeddings capture semantics, not words

This *directly motivates* the NLP phase (Phase 4) and the embeddings-for-RAG part of Phase 7. It's the "why" that turns abstract into visceral.

## 📅 Week 9 Schedule

| Day | Focus | Deliverable |
|-----|-------|-------------|
| Mon | Read Chip Huyen on leakage; audit Week 8 Titanic for leakage | Journal: 3 leakages you nearly committed |
| Tue | Implement Model 1 (TF-IDF + logreg) with proper Pipeline | F1 baseline committed |
| Wed | Implement Model 2 (TF-IDF + MLP) | F1 comparison committed |
| Thu | Implement Model 3 (sentence-transformers + logreg) | F1 comparison + OOD test set |
| Fri | Error analysis across all 3 models; write Friday blog post | Blog post: "Same data, three models, one lesson" |
| Sat | **Phase 2 post-exam** (`assessments/phase_2_post_assessment.ipynb`) + polish project | Exam score committed |
| Sun | Retro + paper-of-the-week + plan Phase 3 | Tag `phase-2-complete` in git |

## ✅ Week 9 / Phase 2 Final Success Criteria

- [ ] All 5 leakage patterns explained in journal with a code snippet
- [ ] Spam classifier ladder (3 models) in `projects/05-spam-classifier/` with comparable F1 numbers
- [ ] Can defend your metric choice in 2 sentences
- [ ] Error analysis notebook identifies ≥3 failure categories per model
- [ ] **Phase 2 post-exam** ≥ 80%
- [ ] 5-minute "teach-it-back" Loom: "Bias-variance tradeoff in 5 minutes"

---

## 🏆 PHASE 2 MILESTONE CHECK

You should now be able to answer each of these out loud:

1. Derive the logistic regression gradient from its BCE loss.
2. When is accuracy a bad metric? Give a concrete example.
3. What is data leakage, and how do you prevent it when using `StandardScaler`?
4. What's the difference between macro-F1, micro-F1, and weighted-F1?
5. When should you use a time-series split vs K-fold?
6. What does "bias-variance tradeoff" mean operationally (not just verbally)?
7. Why did Model 3 (embeddings) in your spam classifier beat Model 1 on OOD inputs?

If 5+ answers feel uncertain, add a refresher week before Phase 3.

## 📚 Phase 2 Project Outputs

| # | Project | Repo Folder | Mini-project Ladder # |
|---|---------|-------------|----------------------|
| 1 | Bias-variance polynomial demo | `projects/05-bias-variance-demo/` | — |
| 2 | Titanic Kaggle (top 40%) | `projects/04-titanic-kaggle/` | 4 |
| 3 | Spam classifier ladder (3 models) | `projects/05-spam-classifier/` | 5 |

## ➡️ Next

[Phase 3: Neural Networks Deep Dive (Weeks 10–15)](./Phase_3_Neural_Networks.md)
