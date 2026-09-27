"""
Generate all pre- and post-phase assessment notebooks.

Usage:
    python assessments/_generate_assessments.py

This produces:
    assessments/phase_1_post_assessment.ipynb
    assessments/phase_2_pre_assessment.ipynb
    assessments/phase_2_post_assessment.ipynb
    ...
    assessments/phase_8_post_assessment.ipynb

Each notebook follows the same convention as 00_Foundation_Assessment_Exercises.ipynb:
 - Intro markdown with grading rubric
 - N exercises, each with (prompt markdown, empty code cell for student, validation cell)
 - Final scoring cell that tallies results

The validation cells use plain `assert` so the notebook is self-graded.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Minimal ipynb cell helpers
# ---------------------------------------------------------------------------

def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }

def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }

def nb(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ai-mastery)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def header(
    phase: str,
    kind: str,
    num_q: int,
    rubric: str = "≥ 80% to pass. ≥ 90% = excellent. < 60% = revisit the phase.",
) -> dict:
    return md(
        f"""# {phase} — {kind} Assessment

**Questions:** {num_q}  |  **Rubric:** {rubric}

## How to use this notebook

1. Read each question carefully.
2. Fill in the code cell marked `# YOUR CODE HERE`.
3. Run the validation cell below it — it uses `assert` statements to check your answer.
4. At the end, the final cell tallies your score.

> If a validation cell raises `AssertionError`, your answer is wrong. Fix the implementation and re-run.
> If a validation cell runs without error AND prints `✅`, you got it.

---
"""
    )

def question(q_id: str, title: str, prompt: str, skeleton: str, validation: str) -> list[dict]:
    return [
        md(f"## Q{q_id} — {title}\n\n{prompt}\n"),
        code(skeleton + "\n"),
        code(
            "# ✅ Validation for Q" + q_id + "\n"
            "# Running this cell WITHOUT errors means you passed.\n"
            "try:\n"
            + "\n".join("    " + ln for ln in validation.splitlines())
            + '\n    print("✅ Q' + q_id + ' passed.")\n'
            "    _scores['" + q_id + "'] = 1\n"
            "except AssertionError as e:\n"
            "    print(f'❌ Q" + q_id + " failed: {e}')\n"
            "    _scores['" + q_id + "'] = 0\n"
        ),
    ]

def init_score() -> dict:
    return code("_scores = {}\n")

def final_score(num_q: int) -> dict:
    return code(
        "total = sum(_scores.values())\n"
        f"max_score = {num_q}\n"
        "print(f'Score: {total}/{max_score} ({100*total/max_score:.0f}%)')\n"
        "if total / max_score >= 0.9:\n"
        "    print('🟢 Excellent. Move forward with confidence.')\n"
        "elif total / max_score >= 0.8:\n"
        "    print('🟢 Pass. You are ready for the next phase.')\n"
        "elif total / max_score >= 0.6:\n"
        "    print('🟡 Borderline. Revisit weak topics for a few days, then retry.')\n"
        "else:\n"
        "    print('🔴 Revisit this phase before moving on.')\n"
    )

# ---------------------------------------------------------------------------
# PHASE 1 — MATH ENGINE — POST assessment
# ---------------------------------------------------------------------------

def phase_1_post() -> dict:
    cells: list[dict] = [
        header("Phase 1: The Math Engine", "POST", 6),
        md("**Scope:** vectors, matrices, chain rule, autograd, MLE, information theory.\n\n---"),
        init_score(),
    ]
    cells += question(
        "1",
        "Dot product & cosine similarity",
        "Implement `dot(a, b)` and `cosine(a, b)` from scratch (no numpy). Both should accept plain lists.",
        "def dot(a, b):\n    # YOUR CODE HERE\n    ...\n\n"
        "def cosine(a, b):\n    # YOUR CODE HERE\n    ...\n",
        "import math\n"
        "assert dot([1,2,3], [4,5,6]) == 32\n"
        "assert abs(cosine([1,0,0], [1,0,0]) - 1.0) < 1e-9\n"
        "assert abs(cosine([1,0,0], [0,1,0]) - 0.0) < 1e-9\n"
        "assert abs(cosine([1,2,3], [2,4,6]) - 1.0) < 1e-9\n",
    )
    cells += question(
        "2",
        "Matrix multiply without numpy",
        "Implement `matmul(A, B)` for A shape (m×k) and B shape (k×n), both lists-of-lists. Return (m×n) list-of-lists.",
        "def matmul(A, B):\n    # YOUR CODE HERE\n    ...\n",
        "A = [[1,2],[3,4]]\n"
        "B = [[5,6],[7,8]]\n"
        "assert matmul(A, B) == [[19,22],[43,50]]\n"
        "import numpy as np\n"
        "Ar = np.random.randn(4,3).tolist()\n"
        "Br = np.random.randn(3,5).tolist()\n"
        "expected = (np.array(Ar) @ np.array(Br)).tolist()\n"
        "got = matmul(Ar, Br)\n"
        "for i in range(4):\n"
        "    for j in range(5):\n"
        "        assert abs(got[i][j] - expected[i][j]) < 1e-9\n",
    )
    cells += question(
        "3",
        "Chain rule by hand",
        "For `f(x) = (3x + 2)**4`, implement the analytical derivative. Then verify it against a numerical derivative at x=1.5.",
        "def df_dx(x):\n    # Use the chain rule: d/dx [(3x+2)^4]\n    # YOUR CODE HERE\n    ...\n",
        "def numerical(x, h=1e-6):\n"
        "    return ((3*(x+h)+2)**4 - (3*(x-h)+2)**4) / (2*h)\n"
        "for x in [0.0, 1.5, -0.7, 3.2]:\n"
        "    assert abs(df_dx(x) - numerical(x)) < 1e-3, (x, df_dx(x), numerical(x))\n",
    )
    cells += question(
        "4",
        "Micrograd-style autograd (mini)",
        "Implement a tiny `Value` class supporting `+` and `*` with backward. Gradients should accumulate.",
        "class Value:\n"
        "    def __init__(self, data, _prev=()):\n"
        "        self.data = data\n"
        "        self.grad = 0.0\n"
        "        self._prev = _prev\n"
        "        self._backward = lambda: None\n"
        "    def __add__(self, other):\n"
        "        # YOUR CODE HERE\n"
        "        ...\n"
        "    def __mul__(self, other):\n"
        "        # YOUR CODE HERE\n"
        "        ...\n"
        "    def backward(self):\n"
        "        # Topological sort + set self.grad = 1.0 + run _backward\n"
        "        # YOUR CODE HERE\n"
        "        ...\n",
        "a = Value(2.0); b = Value(3.0)\n"
        "c = a * b + a\n"
        "c.backward()\n"
        "# dc/da = b + 1 = 4 ; dc/db = a = 2\n"
        "assert abs(a.grad - 4.0) < 1e-9, a.grad\n"
        "assert abs(b.grad - 2.0) < 1e-9, b.grad\n",
    )
    cells += question(
        "5",
        "MLE for a Gaussian",
        "Implement `fit_gaussian_mle(data)` that returns `(mu, sigma)` via the closed-form MLE.",
        "import numpy as np\n"
        "def fit_gaussian_mle(data):\n"
        "    # YOUR CODE HERE\n"
        "    ...\n",
        "import numpy as np\n"
        "rng = np.random.default_rng(0)\n"
        "samples = rng.normal(loc=4.0, scale=2.5, size=20000)\n"
        "mu, sigma = fit_gaussian_mle(samples)\n"
        "assert abs(mu - 4.0) < 0.1, mu\n"
        "assert abs(sigma - 2.5) < 0.1, sigma\n",
    )
    cells += question(
        "6",
        "Cross-entropy, KL, softmax",
        "Implement `softmax(z)`, `cross_entropy(p, q)`, `kl_divergence(p, q)` using numpy only.",
        "import numpy as np\n"
        "def softmax(z, T=1.0):\n"
        "    # YOUR CODE HERE\n"
        "    ...\n"
        "def cross_entropy(p, q):\n"
        "    # YOUR CODE HERE\n"
        "    ...\n"
        "def kl_divergence(p, q):\n"
        "    # YOUR CODE HERE\n"
        "    ...\n",
        "import numpy as np\n"
        "z = np.array([2.0, 1.0, 0.1])\n"
        "q = softmax(z)\n"
        "assert abs(q.sum() - 1.0) < 1e-9\n"
        "# temperature 0.01 → nearly one-hot on arg-max\n"
        "qT = softmax(z, T=0.01)\n"
        "assert qT.argmax() == 0 and qT.max() > 0.99\n"
        "p = np.array([1.0, 0.0, 0.0])  # one-hot on class 0\n"
        "# CE should equal -log(q[0])\n"
        "assert abs(cross_entropy(p, q) - (-np.log(q[0]))) < 1e-9\n"
        "# KL(p||q) = CE - H(p) = CE - 0\n"
        "assert abs(kl_divergence(p, q) - cross_entropy(p, q)) < 1e-9\n",
    )
    cells.append(final_score(6))
    return nb(cells)

# ---------------------------------------------------------------------------
# PHASE 2 — CLASSICAL ML — PRE + POST
# ---------------------------------------------------------------------------

def phase_2_pre() -> dict:
    cells: list[dict] = [
        header("Phase 2: Classical ML", "PRE (readiness check)", 3,
               "≥ 66% → ready for Phase 2. Otherwise add a 1-week refresher of Phase 1."),
        init_score(),
    ]
    cells += question(
        "1",
        "Sigmoid & BCE readiness",
        "Implement sigmoid and binary cross-entropy for a single pair (y, p).",
        "import numpy as np\n"
        "def sigmoid(z): \n    # YOUR CODE HERE\n    ...\n"
        "def bce(y, p, eps=1e-12): \n    # y ∈ {0,1}, p ∈ (0,1)\n    # YOUR CODE HERE\n    ...\n",
        "import numpy as np\n"
        "assert abs(sigmoid(0.0) - 0.5) < 1e-9\n"
        "assert sigmoid(100.0) > 0.9999\n"
        "assert abs(bce(1, 0.9) - (-np.log(0.9))) < 1e-9\n"
        "assert abs(bce(0, 0.1) - (-np.log(0.9))) < 1e-9\n",
    )
    cells += question(
        "2",
        "Derivative of sigmoid",
        "Implement the derivative of sigmoid w.r.t. its input.",
        "def dsigmoid(z):\n    # YOUR CODE HERE  (hint: sigmoid(z) * (1 - sigmoid(z)))\n    ...\n",
        "for z in [-2.0, 0.0, 3.0]:\n"
        "    h = 1e-6\n"
        "    num = (sigmoid(z+h) - sigmoid(z-h)) / (2*h)\n"
        "    assert abs(dsigmoid(z) - num) < 1e-4, (z, dsigmoid(z), num)\n",
    )
    cells += question(
        "3",
        "Vectorized dot product",
        "Using only numpy, compute y = X @ w + b for batch of inputs.",
        "import numpy as np\n"
        "def predict(X, w, b):\n    # YOUR CODE HERE\n    ...\n",
        "import numpy as np\n"
        "X = np.array([[1,2],[3,4],[5,6]], dtype=float)\n"
        "w = np.array([0.5, -0.1])\n"
        "b = 0.2\n"
        "expected = X @ w + b\n"
        "out = predict(X, w, b)\n"
        "assert np.allclose(out, expected)\n",
    )
    cells.append(final_score(3))
    return nb(cells)

def phase_2_post() -> dict:
    cells: list[dict] = [
        header("Phase 2: Classical ML", "POST", 5),
        init_score(),
    ]
    cells += question(
        "1",
        "Logistic regression from scratch",
        "Fit logistic regression with gradient descent on the given data and achieve ≥ 95% accuracy on the held-out split.",
        "import numpy as np\n"
        "from sklearn.datasets import load_breast_cancer\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "X, y = load_breast_cancer(return_X_y=True)\n"
        "X = StandardScaler().fit_transform(X)\n"
        "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)\n"
        "\n"
        "class MyLogReg:\n"
        "    def __init__(self, lr=0.1, n_iter=3000):\n"
        "        self.lr, self.n_iter = lr, n_iter\n"
        "    def fit(self, X, y):\n"
        "        # YOUR CODE HERE\n"
        "        return self\n"
        "    def predict(self, X):\n"
        "        # YOUR CODE HERE\n"
        "        ...\n"
        "\n"
        "model = MyLogReg().fit(Xtr, ytr)\n"
        "acc = (model.predict(Xte) == yte).mean()\n"
        "print(f'Test accuracy: {acc:.3f}')\n",
        "assert acc >= 0.95, f'Expected ≥ 0.95 test accuracy, got {acc:.3f}'\n",
    )
    cells += question(
        "2",
        "F1 score from scratch",
        "Compute F1 without using sklearn. Return float.",
        "def f1_score_manual(y_true, y_pred):\n"
        "    # YOUR CODE HERE\n"
        "    ...\n",
        "import numpy as np\n"
        "from sklearn.metrics import f1_score\n"
        "rng = np.random.default_rng(1)\n"
        "for _ in range(5):\n"
        "    y_true = rng.integers(0, 2, 100)\n"
        "    y_pred = rng.integers(0, 2, 100)\n"
        "    assert abs(f1_score_manual(y_true, y_pred) - f1_score(y_true, y_pred)) < 1e-9\n",
    )
    cells += question(
        "3",
        "Metric choice — multiple choice",
        "Assign the correct metric name to each scenario. Valid answers: 'accuracy', 'f1', 'roc_auc', 'mae', 'recall'.",
        "choices = {\n"
        "    'fraud_detection_imbalanced_99_1': None,           # we care about catching fraud; class imbalance extreme\n"
        "    'house_price_regression': None,                    # regression, outliers present\n"
        "    'spam_filter_balanced_classes': None,              # cost of FP and FN similar\n"
        "    'medical_screen_rather_overdiagnose': None,        # want to minimise FN\n"
        "    'credit_card_risk_calibrated_probabilities': None, # ranking quality\n"
        "}\n"
        "# YOUR CODE HERE — fill in choices with strings\n",
        "assert choices['fraud_detection_imbalanced_99_1'] in ('f1', 'recall')\n"
        "assert choices['house_price_regression'] == 'mae'\n"
        "assert choices['spam_filter_balanced_classes'] in ('f1','accuracy')\n"
        "assert choices['medical_screen_rather_overdiagnose'] == 'recall'\n"
        "assert choices['credit_card_risk_calibrated_probabilities'] == 'roc_auc'\n",
    )
    cells += question(
        "4",
        "Detect the leakage",
        "The following function has a data-leakage bug. Identify it by setting `leakage_kind` to one of: "
        "'preprocessing_before_split', 'target_leakage', 'group_leakage', 'time_leakage'.",
        "code_snippet = '''\n"
        "scaler = StandardScaler().fit(X_all)     # fits on full dataset!\n"
        "X_scaled = scaler.transform(X_all)\n"
        "X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y)\n"
        "model.fit(X_tr, y_tr)\n"
        "'''\n"
        "leakage_kind = None  # YOUR CODE HERE\n",
        "assert leakage_kind == 'preprocessing_before_split'\n",
    )
    cells += question(
        "5",
        "Bias-variance decomposition (conceptual)",
        "A model has low train error and high test error. Is this primarily high bias or high variance? "
        "Set `answer` to `'bias'` or `'variance'`. Then implement `bigger_model_helps` (True/False) and `more_data_helps` (True/False).",
        "answer = None                 # 'bias' or 'variance'\n"
        "bigger_model_helps = None     # True or False\n"
        "more_data_helps = None        # True or False\n"
        "# YOUR CODE HERE\n",
        "assert answer == 'variance'\n"
        "assert bigger_model_helps == False\n"
        "assert more_data_helps == True\n",
    )
    cells.append(final_score(5))
    return nb(cells)

# ---------------------------------------------------------------------------
# PHASE 3 — NEURAL NETWORKS — PRE + POST
# ---------------------------------------------------------------------------

def phase_3_pre() -> dict:
    cells: list[dict] = [
        header("Phase 3: Neural Networks", "PRE (readiness check)", 3),
        init_score(),
    ]
    cells += question(
        "1",
        "PyTorch tensor basics",
        "Create a PyTorch tensor of shape (3, 4) filled with 1.0, on MPS if available else CPU.",
        "import torch\n"
        "device = 'mps' if torch.backends.mps.is_available() else 'cpu'\n"
        "def make_ones():\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "t = make_ones()\n"
        "assert t.shape == (3,4)\n"
        "assert torch.allclose(t, torch.ones_like(t))\n",
    )
    cells += question(
        "2",
        "MSE loss for regression",
        "Implement MSE in pure PyTorch without torch.nn.functional.mse_loss.",
        "import torch\n"
        "def mse(y_pred, y_true):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "import torch.nn.functional as F\n"
        "y = torch.randn(16, 3)\n"
        "yhat = torch.randn(16, 3)\n"
        "assert abs(mse(yhat, y).item() - F.mse_loss(yhat, y).item()) < 1e-6\n",
    )
    cells += question(
        "3",
        "One-hot encoder",
        "Implement one-hot encoding without using nn.functional.one_hot.",
        "import torch\n"
        "def one_hot(labels, num_classes):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "import torch.nn.functional as F\n"
        "labels = torch.tensor([0, 2, 1, 3])\n"
        "out = one_hot(labels, 4)\n"
        "exp = F.one_hot(labels, 4).float()\n"
        "assert out.shape == exp.shape\n"
        "assert torch.allclose(out.float(), exp)\n",
    )
    cells.append(final_score(3))
    return nb(cells)

def phase_3_post() -> dict:
    cells: list[dict] = [
        header("Phase 3: Neural Networks Deep Dive", "POST", 5),
        init_score(),
    ]
    cells += question(
        "1",
        "Manual forward pass of a 2-layer MLP",
        "Given weights, compute `ReLU(x @ W1 + b1) @ W2 + b2` manually.",
        "import torch\n"
        "def mlp_forward(x, W1, b1, W2, b2):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.randn(4, 3)\n"
        "W1 = torch.randn(3, 8); b1 = torch.zeros(8)\n"
        "W2 = torch.randn(8, 2); b2 = torch.zeros(2)\n"
        "ref = torch.relu(x @ W1 + b1) @ W2 + b2\n"
        "out = mlp_forward(x, W1, b1, W2, b2)\n"
        "assert torch.allclose(out, ref, atol=1e-6)\n",
    )
    cells += question(
        "2",
        "Manual backprop through a linear layer",
        "For y = x @ W + b, given dL/dy, return dL/dx, dL/dW, dL/db.",
        "import torch\n"
        "def linear_backward(dL_dy, x, W):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.randn(5, 3, requires_grad=True)\n"
        "W = torch.randn(3, 4, requires_grad=True)\n"
        "b = torch.zeros(4, requires_grad=True)\n"
        "y = x @ W + b\n"
        "dL_dy = torch.randn_like(y)\n"
        "y.backward(dL_dy)\n"
        "dLdx, dLdW, dLdb = linear_backward(dL_dy, x, W)\n"
        "assert torch.allclose(dLdx, x.grad, atol=1e-6)\n"
        "assert torch.allclose(dLdW, W.grad, atol=1e-6)\n"
        "assert torch.allclose(dLdb, dL_dy.sum(0), atol=1e-6)\n",
    )
    cells += question(
        "3",
        "BatchNorm forward (train mode)",
        "Implement BatchNorm forward over the batch dim (dim=0) with learnable gamma/beta.",
        "import torch\n"
        "def batchnorm_forward(x, gamma, beta, eps=1e-5):\n    # YOUR CODE HERE\n    ...\n",
        "import torch, torch.nn as nn\n"
        "torch.manual_seed(0)\n"
        "x = torch.randn(32, 16)\n"
        "gamma = torch.ones(16); beta = torch.zeros(16)\n"
        "out = batchnorm_forward(x, gamma, beta)\n"
        "bn = nn.BatchNorm1d(16, affine=False, track_running_stats=False)\n"
        "ref = bn(x)\n"
        "assert torch.allclose(out, ref, atol=1e-4)\n",
    )
    cells += question(
        "4",
        "Loss-curve pathology diagnosis",
        "Match each pathology to its cause. Valid causes: 'lr_too_high', 'lr_too_low', 'no_normalization', 'overfit', 'dead_relu'.",
        "diagnoses = {\n"
        "    'loss_diverges_to_nan_within_50_steps': None,\n"
        "    'loss_plateaus_high_from_step_1': None,\n"
        "    'train_loss_drops_val_loss_rises': None,\n"
        "    'many_neurons_output_zero_and_stay_zero': None,\n"
        "    'activations_saturate_and_training_stalls_in_deep_net': None,\n"
        "}\n"
        "# YOUR CODE HERE — fill in diagnoses\n",
        "assert diagnoses['loss_diverges_to_nan_within_50_steps'] == 'lr_too_high'\n"
        "assert diagnoses['loss_plateaus_high_from_step_1'] == 'lr_too_low'\n"
        "assert diagnoses['train_loss_drops_val_loss_rises'] == 'overfit'\n"
        "assert diagnoses['many_neurons_output_zero_and_stay_zero'] == 'dead_relu'\n"
        "assert diagnoses['activations_saturate_and_training_stalls_in_deep_net'] == 'no_normalization'\n",
    )
    cells += question(
        "5",
        "Train a tiny net on XOR",
        "Train a 2-layer MLP on the XOR problem until train accuracy is 100%.",
        "import torch, torch.nn as nn\n"
        "X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)\n"
        "y = torch.tensor([0,1,1,0])\n"
        "# Build and train the model\n"
        "# YOUR CODE HERE — define `model`, train it, and produce `acc`\n"
        "# acc = ...\n",
        "assert acc >= 0.99, f'Expected 100%% XOR accuracy, got {acc}'\n",
    )
    cells.append(final_score(5))
    return nb(cells)

# ---------------------------------------------------------------------------
# PHASE 4 — NLP & SEQUENCES — PRE + POST
# ---------------------------------------------------------------------------

def phase_4_pre() -> dict:
    cells: list[dict] = [
        header("Phase 4: NLP & Sequences", "PRE", 3),
        init_score(),
    ]
    cells += question(
        "1",
        "Cosine similarity on torch tensors",
        "Implement cosine similarity for two 1D torch tensors.",
        "import torch\n"
        "def cos(a, b):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "import torch.nn.functional as F\n"
        "a = torch.randn(128); b = torch.randn(128)\n"
        "assert abs(cos(a,b).item() - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()) < 1e-5\n",
    )
    cells += question(
        "2",
        "Softmax along last dim",
        "Implement numerically stable softmax across the last dim of a 2D tensor.",
        "import torch\n"
        "def softmax_last(x):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "x = torch.tensor([[1000.0, 1001.0, 1002.0]])\n"
        "out = softmax_last(x)\n"
        "assert abs(out.sum().item() - 1.0) < 1e-6\n"
        "assert not torch.isnan(out).any()\n",
    )
    cells += question(
        "3",
        "Embedding lookup",
        "Given an embedding matrix E of shape (V, D) and a tensor of token IDs shape (B, T), return the embeddings shape (B, T, D).",
        "import torch\n"
        "def lookup(E, ids):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "E = torch.randn(50, 8)\n"
        "ids = torch.randint(0, 50, (4, 6))\n"
        "out = lookup(E, ids)\n"
        "assert out.shape == (4, 6, 8)\n"
        "assert torch.allclose(out[0,0], E[ids[0,0]])\n",
    )
    cells.append(final_score(3))
    return nb(cells)

def phase_4_post() -> dict:
    cells: list[dict] = [
        header("Phase 4: NLP & Sequence Models", "POST", 5),
        init_score(),
    ]
    cells += question(
        "1",
        "Byte-pair encoding: merge step",
        "Given a counter of symbol-pair frequencies from your corpus, return the most frequent pair.",
        "from collections import Counter\n"
        "def best_pair(pair_counts: Counter):\n    # YOUR CODE HERE\n    ...\n",
        "from collections import Counter\n"
        "c = Counter({('l','o'): 3, ('o','w'): 5, ('w','e'): 2})\n"
        "assert best_pair(c) == ('o','w')\n",
    )
    cells += question(
        "2",
        "Perplexity from cross-entropy",
        "Given cross-entropy loss in nats, return perplexity.",
        "import math\n"
        "def perplexity(ce_nats):\n    # YOUR CODE HERE\n    ...\n",
        "import math\n"
        "assert abs(perplexity(0.0) - 1.0) < 1e-9\n"
        "assert abs(perplexity(math.log(4)) - 4.0) < 1e-6\n",
    )
    cells += question(
        "3",
        "RNN single-step forward",
        "Implement a vanilla RNN step: h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h).",
        "import torch\n"
        "def rnn_step(x_t, h_prev, W_xh, W_hh, b_h):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "torch.manual_seed(0)\n"
        "x = torch.randn(3, 4); h = torch.randn(3, 5)\n"
        "W_xh = torch.randn(4, 5); W_hh = torch.randn(5, 5); b = torch.zeros(5)\n"
        "out = rnn_step(x, h, W_xh, W_hh, b)\n"
        "expected = torch.tanh(x @ W_xh + h @ W_hh + b)\n"
        "assert torch.allclose(out, expected, atol=1e-6)\n",
    )
    cells += question(
        "4",
        "Attention weights row sum",
        "Given raw attention scores (B, T, T), apply softmax along last dim and verify rows sum to 1.",
        "import torch\n"
        "def attn_weights(scores):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "s = torch.randn(2, 7, 7)\n"
        "w = attn_weights(s)\n"
        "assert torch.allclose(w.sum(dim=-1), torch.ones(2, 7), atol=1e-5)\n",
    )
    cells += question(
        "5",
        "Why RNNs struggle with long contexts (conceptual)",
        "Pick ONE answer by setting `answer` to 'a','b','c', or 'd'.\n\n"
        "(a) RNNs can only process sequences of length ≤ 512.\n"
        "(b) Gradients tend to vanish or explode as they propagate back through many time steps.\n"
        "(c) Transformers are faster because they use GPUs.\n"
        "(d) RNNs ignore word embeddings.",
        "answer = None  # 'a', 'b', 'c', or 'd'\n",
        "assert answer == 'b'\n",
    )
    cells.append(final_score(5))
    return nb(cells)

# ---------------------------------------------------------------------------
# PHASE 5 — TRANSFORMER — PRE + POST
# ---------------------------------------------------------------------------

def phase_5_pre() -> dict:
    cells: list[dict] = [
        header("Phase 5: The Transformer", "PRE", 3),
        init_score(),
    ]
    cells += question(
        "1",
        "Reshape (B, T, C) ↔ (B*T, C)",
        "Flatten batch and time dims, then restore.",
        "import torch\n"
        "def flatten_bt(x): ...\n"
        "def unflatten_bt(x_flat, B, T): ...\n",
        "import torch\n"
        "x = torch.randn(4, 7, 16)\n"
        "x_flat = flatten_bt(x)\n"
        "assert x_flat.shape == (28, 16)\n"
        "x_rec = unflatten_bt(x_flat, 4, 7)\n"
        "assert torch.allclose(x, x_rec)\n",
    )
    cells += question(
        "2",
        "Dot product attention (tiny)",
        "Implement `attn = softmax(Q @ K.T / sqrt(d)) @ V` for single-head without masking.",
        "import torch, math\n"
        "def attn(Q, K, V):\n    # YOUR CODE HERE\n    ...\n",
        "import torch, math\n"
        "torch.manual_seed(0)\n"
        "T, d = 5, 8\n"
        "Q = torch.randn(T, d); K = torch.randn(T, d); V = torch.randn(T, d)\n"
        "scores = Q @ K.T / math.sqrt(d)\n"
        "w = torch.softmax(scores, dim=-1)\n"
        "ref = w @ V\n"
        "out = attn(Q, K, V)\n"
        "assert torch.allclose(out, ref, atol=1e-6)\n",
    )
    cells += question(
        "3",
        "Parameter count of a Linear layer",
        "Return number of parameters of nn.Linear(in, out) with bias=True.",
        "def nparam_linear(in_, out_):\n    # YOUR CODE HERE\n    ...\n",
        "assert nparam_linear(768, 768) == 768*768 + 768\n"
        "assert nparam_linear(512, 2048) == 512*2048 + 2048\n",
    )
    cells.append(final_score(3))
    return nb(cells)

def phase_5_post() -> dict:
    cells: list[dict] = [
        header("Phase 5: The Transformer", "POST", 5),
        init_score(),
    ]
    cells += question(
        "1",
        "Scaled dot-product attention with causal mask",
        "Implement causal self-attention for a single head.",
        "import torch, math\n"
        "def causal_attn(Q, K, V):\n"
        "    # Q/K/V shape: (B, T, d)\n"
        "    # Return output (B, T, d) where each position only attends to <= itself\n"
        "    # YOUR CODE HERE\n    ...\n",
        "import torch, math\n"
        "torch.manual_seed(0)\n"
        "B, T, d = 2, 4, 8\n"
        "Q = torch.randn(B, T, d); K = torch.randn(B, T, d); V = torch.randn(B, T, d)\n"
        "out = causal_attn(Q, K, V)\n"
        "assert out.shape == (B, T, d)\n"
        "# Position 0 should only depend on V[..., 0]\n"
        "scores = (Q @ K.transpose(-2,-1)) / math.sqrt(d)\n"
        "mask = torch.triu(torch.ones(T, T), diagonal=1).bool()\n"
        "scores = scores.masked_fill(mask, float('-inf'))\n"
        "ref = torch.softmax(scores, dim=-1) @ V\n"
        "assert torch.allclose(out, ref, atol=1e-5)\n",
    )
    cells += question(
        "2",
        "Multi-head split and merge",
        "Split (B, T, d_model) into (B, nh, T, d_head) and back.",
        "import torch\n"
        "def split_heads(x, n_heads):\n    # YOUR CODE HERE\n    ...\n"
        "def merge_heads(x):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "x = torch.randn(2, 6, 32)\n"
        "h = split_heads(x, 4)\n"
        "assert h.shape == (2, 4, 6, 8)\n"
        "m = merge_heads(h)\n"
        "assert m.shape == (2, 6, 32)\n"
        "assert torch.allclose(x, m)\n",
    )
    cells += question(
        "3",
        "LayerNorm forward",
        "Implement LayerNorm over the last dim.",
        "import torch\n"
        "def layer_norm(x, gamma, beta, eps=1e-5):\n    # YOUR CODE HERE\n    ...\n",
        "import torch, torch.nn as nn\n"
        "x = torch.randn(4, 16)\n"
        "gamma = torch.ones(16); beta = torch.zeros(16)\n"
        "ref = nn.LayerNorm(16, eps=1e-5, elementwise_affine=False)(x)\n"
        "out = layer_norm(x, gamma, beta)\n"
        "assert torch.allclose(out, ref, atol=1e-5)\n",
    )
    cells += question(
        "4",
        "Sinusoidal positional encoding",
        "Build the (max_len, d_model) positional-encoding matrix from Vaswani et al.",
        "import torch, math\n"
        "def pos_encoding(max_len, d_model):\n    # YOUR CODE HERE\n    ...\n",
        "import torch, math\n"
        "PE = pos_encoding(10, 16)\n"
        "assert PE.shape == (10, 16)\n"
        "# PE[pos, 2i] = sin(pos / 10000^(2i/d))\n"
        "pos, i = 3, 0\n"
        "expected = math.sin(pos / 10000 ** (2*i / 16))\n"
        "assert abs(PE[pos, 2*i].item() - expected) < 1e-5\n",
    )
    cells += question(
        "5",
        "FFN parameter count (GPT-style)",
        "Return parameter count of an FFN block `Linear(d, 4d) → GELU → Linear(4d, d)` with biases.",
        "def ffn_params(d):\n    # YOUR CODE HERE\n    ...\n",
        "assert ffn_params(512) == 512*(4*512) + 4*512 + (4*512)*512 + 512\n",
    )
    cells.append(final_score(5))
    return nb(cells)

# ---------------------------------------------------------------------------
# PHASE 6 — mini-GPT — PRE + POST
# ---------------------------------------------------------------------------

def phase_6_pre() -> dict:
    cells: list[dict] = [
        header("Phase 6: Build Your mini-GPT", "PRE", 3),
        init_score(),
    ]
    cells += question(
        "1",
        "Count total parameters of a nn.Module",
        "Given an nn.Module, sum .numel() of its parameters.",
        "import torch.nn as nn\n"
        "def count_params(m):\n    # YOUR CODE HERE\n    ...\n",
        "import torch.nn as nn\n"
        "m = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))\n"
        "assert count_params(m) == 10*20 + 20 + 20*5 + 5\n",
    )
    cells += question(
        "2",
        "Cross-entropy over a sequence (B, T, V)",
        "Given logits (B, T, V) and targets (B, T), compute mean CE loss.",
        "import torch\n"
        "import torch.nn.functional as F\n"
        "def seq_ce(logits, targets):\n    # YOUR CODE HERE\n    ...\n",
        "import torch, torch.nn.functional as F\n"
        "B, T, V = 4, 6, 10\n"
        "logits = torch.randn(B, T, V); targets = torch.randint(0, V, (B, T))\n"
        "ref = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))\n"
        "assert abs(seq_ce(logits, targets).item() - ref.item()) < 1e-6\n",
    )
    cells += question(
        "3",
        "Top-k masking",
        "Given logits (B, V), zero out all but top-k and renormalize via softmax.",
        "import torch\n"
        "def top_k_softmax(logits, k):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])\n"
        "p = top_k_softmax(logits, k=2)\n"
        "assert p.shape == logits.shape\n"
        "assert abs(p.sum().item() - 1.0) < 1e-6\n"
        "assert (p[0, 2:] == 0).all()\n",
    )
    cells.append(final_score(3))
    return nb(cells)

def phase_6_post() -> dict:
    cells: list[dict] = [
        header("Phase 6: Build Your mini-GPT", "POST", 5),
        init_score(),
    ]
    cells += question(
        "1",
        "KV-cache shape reasoning",
        "For a decoder with n_layers=12, n_heads=12, d_head=64, seq_len=T (generated so far), batch=1, "
        "return number of float values in the KV cache (both K and V across all layers).",
        "def kv_cache_size(n_layers, n_heads, d_head, T, batch=1):\n    # YOUR CODE HERE\n    ...\n",
        "# K and V per layer: each has shape (batch, n_heads, T, d_head)\n"
        "assert kv_cache_size(12, 12, 64, 1024) == 2 * 12 * 1 * 12 * 1024 * 64\n",
    )
    cells += question(
        "2",
        "Top-p (nucleus) sampling",
        "Given sorted logits, keep smallest set whose cumulative prob ≥ p; renormalize; return new probs.",
        "import torch\n"
        "def top_p(logits, p=0.9):\n    # YOUR CODE HERE\n    ...\n",
        "import torch\n"
        "# build distribution where top-3 tokens sum to 0.95\n"
        "logits = torch.tensor([5.0, 4.0, 3.0, -5.0, -10.0])\n"
        "probs = top_p(logits, p=0.9)\n"
        "assert abs(probs.sum().item() - 1.0) < 1e-6\n"
        "assert probs[-1].item() == 0.0 and probs[-2].item() == 0.0\n",
    )
    cells += question(
        "3",
        "Scaling-laws napkin math",
        "Chinchilla says tokens ≈ 20 * params for compute-optimal training. "
        "For a 125M parameter model, how many tokens?",
        "def chinchilla_tokens(params):\n    # YOUR CODE HERE\n    ...\n",
        "assert chinchilla_tokens(125_000_000) == 20 * 125_000_000\n",
    )
    cells += question(
        "4",
        "Loss → perplexity conversion",
        "Given validation cross-entropy (nats), return perplexity rounded to 2 decimals.",
        "import math\n"
        "def loss_to_ppl(ce):\n    # YOUR CODE HERE\n    ...\n",
        "import math\n"
        "assert abs(loss_to_ppl(2.3) - round(math.exp(2.3), 2)) < 1e-6\n",
    )
    cells += question(
        "5",
        "Temperature and sampling (conceptual)",
        "Match each description to 'greedy', 'temp=0.8', or 'top_p=0.95'. "
        "Set each key in `answers` dict.",
        "answers = {\n"
        "    'deterministic_least_creative': None,\n"
        "    'often_best_balance_of_quality_and_diversity': None,\n"
        "    'uses_the_nucleus_of_cumulative_probability': None,\n"
        "}\n"
        "# YOUR CODE HERE\n",
        "assert answers['deterministic_least_creative'] == 'greedy'\n"
        "assert answers['often_best_balance_of_quality_and_diversity'] == 'temp=0.8'\n"
        "assert answers['uses_the_nucleus_of_cumulative_probability'] == 'top_p=0.95'\n",
    )
    cells.append(final_score(5))
    return nb(cells)

# ---------------------------------------------------------------------------
# PHASE 7 — AI ENGINEER TOOLKIT — PRE + POST
# ---------------------------------------------------------------------------

def phase_7_pre() -> dict:
    cells: list[dict] = [
        header("Phase 7: AI Engineer Toolkit", "PRE", 3),
        init_score(),
    ]
    cells += question(
        "1",
        "Cosine similarity in numpy",
        "Given matrix X (N, D) and query q (D,), return top-k indices of cosine sim.",
        "import numpy as np\n"
        "def top_k_cos(X, q, k):\n    # YOUR CODE HERE\n    ...\n",
        "import numpy as np\n"
        "rng = np.random.default_rng(0)\n"
        "X = rng.normal(size=(100, 16))\n"
        "q = X[42].copy()\n"
        "idx = top_k_cos(X, q, 3)\n"
        "assert idx[0] == 42\n"
        "assert len(idx) == 3\n",
    )
    cells += question(
        "2",
        "JSON mode / Pydantic readiness",
        "Using Pydantic, define a model `SOQLAction` with fields `query: str` and `limit: int` (default 200). Construct and validate.",
        "from pydantic import BaseModel\n"
        "class SOQLAction(BaseModel):\n    # YOUR CODE HERE\n    ...\n",
        "from pydantic import ValidationError\n"
        "a = SOQLAction(query='SELECT Id FROM Case')\n"
        "assert a.limit == 200\n"
        "try:\n"
        "    SOQLAction(limit=10)  # missing required 'query'\n"
        "    raise AssertionError('Should have raised ValidationError')\n"
        "except ValidationError:\n"
        "    pass\n",
    )
    cells += question(
        "3",
        "Chunking a long document",
        "Split text into chunks of at most N characters with overlap O.",
        "def chunk(text, size=500, overlap=50):\n    # YOUR CODE HERE\n    ...\n",
        "txt = 'a' * 1200\n"
        "chunks = chunk(txt, size=500, overlap=50)\n"
        "assert all(len(c) <= 500 for c in chunks)\n"
        "assert chunks[0][-50:] == chunks[1][:50]\n",
    )
    cells.append(final_score(3))
    return nb(cells)

def phase_7_post() -> dict:
    cells: list[dict] = [
        header("Phase 7: AI Engineer Toolkit", "POST", 5),
        init_score(),
    ]
    cells += question(
        "1",
        "LoRA parameter-count math",
        "For a weight W of shape (d, k)=(768, 768) with LoRA rank r=8, return (full_params, lora_params).",
        "def lora_params(d, k, r):\n    # YOUR CODE HERE — return (d*k, r*(d+k))\n    ...\n",
        "full, lora = lora_params(768, 768, 8)\n"
        "assert full == 768*768\n"
        "assert lora == 8*(768+768)\n"
        "assert lora / full < 0.03  # <3%\n",
    )
    cells += question(
        "2",
        "Reciprocal Rank Fusion (RRF)",
        "Given two ranked lists of doc IDs, fuse via RRF (k=60) and return top-3.",
        "def rrf_fuse(list_a, list_b, k=60, top=3):\n    # YOUR CODE HERE\n    ...\n",
        "a = ['d1','d2','d3','d4']\n"
        "b = ['d3','d1','d5','d6']\n"
        "out = rrf_fuse(a, b, top=3)\n"
        "assert set(out) <= {'d1','d2','d3','d4','d5','d6'}\n"
        "assert len(out) == 3\n"
        "# d1 and d3 should dominate (appear in both lists)\n"
        "assert 'd1' in out[:2] and 'd3' in out[:2]\n",
    )
    cells += question(
        "3",
        "When to RAG vs fine-tune — decision",
        "Set `decision` to 'rag', 'fine_tune', 'prompt', or 'agent' for each scenario.",
        "decision = {\n"
        "    'always_up_to_date_company_policy_answers': None,\n"
        "    'model_should_write_in_our_brand_voice': None,\n"
        "    'model_should_call_Jira_to_create_tickets': None,\n"
        "    'one_off_demo_for_internal_hackathon': None,\n"
        "}\n"
        "# YOUR CODE HERE\n",
        "assert decision['always_up_to_date_company_policy_answers'] == 'rag'\n"
        "assert decision['model_should_write_in_our_brand_voice'] == 'fine_tune'\n"
        "assert decision['model_should_call_Jira_to_create_tickets'] == 'agent'\n"
        "assert decision['one_off_demo_for_internal_hackathon'] == 'prompt'\n",
    )
    cells += question(
        "4",
        "Agent step budget",
        "Given a max step budget, implement a guard that returns `'budget_exceeded'` on overrun.",
        "def run_agent_stub(steps_needed, max_steps=5):\n    # YOUR CODE HERE\n    # Simulate taking steps_needed steps; stop at max_steps.\n    ...\n",
        "assert run_agent_stub(3, max_steps=5) != 'budget_exceeded'\n"
        "assert run_agent_stub(8, max_steps=5) == 'budget_exceeded'\n",
    )
    cells += question(
        "5",
        "DPO intuition (conceptual)",
        "DPO trains on pairs (chosen, rejected). Its loss encourages the model to... pick 'a','b','c', or 'd'.\n\n"
        "(a) Match ChatGPT token-for-token on chosen.\n"
        "(b) Increase log-prob of chosen relative to reference, decrease log-prob of rejected relative to reference.\n"
        "(c) Use a reward model trained via reinforcement learning.\n"
        "(d) Compute BLEU against a golden answer.",
        "answer = None  # 'a','b','c','d'\n",
        "assert answer == 'b'\n",
    )
    cells.append(final_score(5))
    return nb(cells)

# ---------------------------------------------------------------------------
# PHASE 8 — CAPSTONE — PRE + POST (reflection-only)
# ---------------------------------------------------------------------------

def phase_8_pre() -> dict:
    cells: list[dict] = [
        header("Phase 8: Capstone & Launch", "PRE (readiness)", 3),
        init_score(),
    ]
    cells += question(
        "1",
        "Architecture recall",
        "In one Python list `components`, list (as strings) the six major components in your CodeSage design. "
        "Must include: 'retriever', 'llm', 'ui', 'cache', 'observability', 'evals'.",
        "components = [ ]  # YOUR CODE HERE\n",
        "needed = {'retriever','llm','ui','cache','observability','evals'}\n"
        "assert needed.issubset(set(components))\n",
    )
    cells += question(
        "2",
        "Launch checklist",
        "Set `ready` boolean flags.",
        "ready = {\n"
        "    'eval_harness_green': None,   # True if golden set pass rate ≥ 85%\n"
        "    'docker_reproducible': None,  # True if another dev can `docker compose up` clean\n"
        "    'demo_video_recorded': None,  # True if 3-min demo video ready\n"
        "}\n"
        "# YOUR CODE HERE — set each to True before starting Week 35\n",
        "assert all(v is True for v in ready.values())\n",
    )
    cells += question(
        "3",
        "Time budget",
        "Your two weeks split. Assign integer hours to each activity. Total MUST equal 80 hrs (2 weeks × 40hrs).",
        "budget = {\n"
        "    'integration_coding': None,\n"
        "    'ui_polish': None,\n"
        "    'evals_and_fixes': None,\n"
        "    'docker_deploy': None,\n"
        "    'blog_post_writing': None,\n"
        "    'demo_video_production': None,\n"
        "    'mock_interview_prep': None,\n"
        "}\n"
        "# YOUR CODE HERE — pick any positive ints that sum to 80\n",
        "vals = list(budget.values())\n"
        "assert all(isinstance(v, int) and v > 0 for v in vals)\n"
        "assert sum(vals) == 80\n",
    )
    cells.append(final_score(3))
    return nb(cells)

def phase_8_post() -> dict:
    cells: list[dict] = [
        header("Phase 8: Capstone & Launch", "POST (graduation)", 5,
               "This is a reflection + design exam. Written answers below are graded by YOU — be honest."),
        init_score(),
    ]
    cells += question(
        "1",
        "Design: 10K queries/day RAG with P95 < 2s",
        "Write an architectural sketch (comment-style) as a Python docstring assigned to `DESIGN_1`. "
        "Must mention: caching, async retrieval, streaming, token budget, eval harness, and observability.",
        "DESIGN_1 = ''' \n# YOUR DESIGN HERE\n'''\n",
        "must = ['cache', 'async', 'stream', 'token', 'eval', 'observ']\n"
        "dl = DESIGN_1.lower()\n"
        "for kw in must: assert kw in dl, f'missing concept: {kw}'\n",
    )
    cells += question(
        "2",
        "Debug: 'My RAG hallucinates 20% of the time'",
        "List at least 4 causes in `CAUSES: list[str]`.",
        "CAUSES = [\n"
        "    # YOUR CODE HERE\n"
        "]\n",
        "assert len(CAUSES) >= 4\n",
    )
    cells += question(
        "3",
        "Evaluate a new prompt version",
        "Given the `prompt_v1_scores` and `prompt_v2_scores` lists of per-case scores, "
        "write `should_ship` True/False rule: ship only if v2 mean > v1 mean AND no case regressed by more than 0.2.",
        "def should_ship(v1, v2):\n    # YOUR CODE HERE\n    ...\n",
        "import numpy as np\n"
        "v1 = [0.7, 0.8, 0.9, 0.6]\n"
        "v2_good = [0.75, 0.85, 0.95, 0.62]\n"
        "v2_regress = [0.8, 0.9, 0.95, 0.35]   # last case regressed by 0.25\n"
        "assert should_ship(v1, v2_good) is True\n"
        "assert should_ship(v1, v2_regress) is False\n",
    )
    cells += question(
        "4",
        "Resume line rewrite",
        "Set `LINE` to a resume bullet that reframes 'built an Apex trigger' using AI-engineering-friendly language. "
        "Must include AT LEAST ONE of: 'pipeline', 'evaluation', 'observability', 'batch', 'rate'.",
        "LINE = ''  # YOUR CODE HERE\n",
        "vocab = ['pipeline','evaluation','observability','batch','rate']\n"
        "assert any(v in LINE.lower() for v in vocab)\n"
        "assert len(LINE) > 40\n",
    )
    cells += question(
        "5",
        "Self-assessment",
        "Set `CAN_EXPLAIN` dict of booleans for each competency. All must be True to graduate.",
        "CAN_EXPLAIN = {\n"
        "    'attention_from_scratch': None,\n"
        "    'lora_math': None,\n"
        "    'why_my_rag_fails_and_how_to_fix': None,\n"
        "    'when_to_fine_tune_vs_rag_vs_prompt': None,\n"
        "    'eval_harness_with_llm_as_judge': None,\n"
        "    'streaming_fastapi_with_cache': None,\n"
        "    'quantization_tradeoffs': None,\n"
        "    'dpo_vs_ppo': None,\n"
        "}\n"
        "# YOUR CODE HERE — be honest. If False, go back and revisit.\n",
        "assert all(v is True for v in CAN_EXPLAIN.values()), (\n"
        "    'At least one competency is not confidently true. '\n"
        "    'Revisit the corresponding phase content before claiming graduation.'\n"
        ")\n",
    )
    cells.append(final_score(5))
    return nb(cells)

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

NOTEBOOKS: dict[str, Callable[[], dict]] = {
    "phase_1_post_assessment.ipynb": phase_1_post,
    "phase_2_pre_assessment.ipynb":  phase_2_pre,
    "phase_2_post_assessment.ipynb": phase_2_post,
    "phase_3_pre_assessment.ipynb":  phase_3_pre,
    "phase_3_post_assessment.ipynb": phase_3_post,
    "phase_4_pre_assessment.ipynb":  phase_4_pre,
    "phase_4_post_assessment.ipynb": phase_4_post,
    "phase_5_pre_assessment.ipynb":  phase_5_pre,
    "phase_5_post_assessment.ipynb": phase_5_post,
    "phase_6_pre_assessment.ipynb":  phase_6_pre,
    "phase_6_post_assessment.ipynb": phase_6_post,
    "phase_7_pre_assessment.ipynb":  phase_7_pre,
    "phase_7_post_assessment.ipynb": phase_7_post,
    "phase_8_pre_assessment.ipynb":  phase_8_pre,
    "phase_8_post_assessment.ipynb": phase_8_post,
}

def main() -> None:
    for filename, fn in NOTEBOOKS.items():
        path = HERE / filename
        path.write_text(json.dumps(fn(), indent=1))
        print(f"Wrote {path}")

if __name__ == "__main__":
    main()
