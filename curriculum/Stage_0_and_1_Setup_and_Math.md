# STAGE 0 & STAGE 1
## Setup, Baseline, and Math You Can Run

> ~210 hours · ~12 weeks · This is the document you work from for the next three months.

---

# STAGE 0 — Setup, Baseline & the Contract
**~20 hrs · 1 week**

## 0.1 — Environment (~4 hrs)

Your original `curriculum/v1/Phase_0_Environment_and_Mindset_Setup.md` covers this well. Follow it. The one-line verification that matters:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# must print: True
```

If that prints `False`, stop and fix it before anything else. Everything downstream assumes GPU acceleration on your M3.

Beyond the original list, install: `pytest` (you'll test primitives from module 1.2 onward), `matplotlib` (you'll plot constantly), `jupyterlab`.

## 0.2 — The repo (~3 hrs)

Public repo, `zero-to-ai-engineer`. Structure from your v1 `LEARNING_SYSTEM.md` §5 — it's good. Two changes:

- `notebooks/stage_N_*/` not `phase_N_*/` — match the new numbering so nothing drifts again
- Add `drills/` for Re-Implementation Drill attempts

Public matters. Not because anyone will read it, but because "I'll clean it up before pushing" is how repos die.

## 0.3 — Anki (~2 hrs)

One deck per stage: `S1-math`, `S2-classical`, `S3-nn`, and so on.

Card rules from v1 hold — one concept per card, cloze for equations, no "explain attention" soup cards. Add one rule: **every card you write must be answerable in under 15 seconds.** If it isn't, it's two cards.

## 0.4 — Journal + paper workflow (~2 hrs)

Read Karpathy's *Software 2.0*. Write a one-page summary in `papers/week-00/`. This is the smallest possible rep of the workflow you'll run weekly for fifteen months — do it once now so the machinery exists.

## 0.5 — Baseline diagnostic (~4 hrs)

Run `assessments/00_Foundation_Assessment_Exercises.ipynb`. **Score yourself honestly.** Nobody sees this but you and Claude, and an inflated score costs you months later when a stage assumes something you don't have.

Log the result in `PROGRESS.md`.

## 0.6 — The contract (~2 hrs, with Claude)

A session where you and Claude agree explicitly:

- Your pace, and what a Minimum Viable Week is for you specifically
- That Claude enforces the five gates and will refuse to advance you
- That Claude does not write your project code
- What you want Claude to do when you're frustrated and want to skip ahead
- How often you'll check in (weekly Sunday session is the default)

Write the agreed terms into `PROGRESS.md`. This matters because in month four you will want to renegotiate it, and a written contract is harder to quietly abandon than an implicit one.

**Stage 0 gate:** MPS verified · repo public · diagnostic logged · contract written.

---

# STAGE 1 — Math You Can Run
**~190 hrs · ~11 weeks**

## How this stage works, and why it's different

You told me your math is mostly gone. That's a normal starting point for a senior engineer eight years out of a B.Tech, and it is completely recoverable — but only if we don't teach it the way it was taught to you the first time.

**The rule for this entire stage: nothing arrives as notation first.**

Every concept comes in this order:

```
1. A concrete problem you can feel
2. Code you write that solves it
3. Output — a plot, a printed number — that makes the behaviour visible
4. "Here's how mathematicians write what you just built"
5. The notation, the general form, the proof sketch
```

That order is not a concession or a simplification. For someone with eight years of programming intuition and rusty math, it is genuinely the faster path. Notation is a compression format for an idea you already hold. Handed to you before the idea, it's noise — and it's what makes capable people conclude they're "not math people."

You are not bad at math. You are out of practice, and you were taught symbol-manipulation instead of meaning.

### What you'll have at the end

A NumPy library you wrote, with tests. An autograd engine you built that matches PyTorch's gradients to 1e-6. And the ability to read `∂L/∂W` without flinching.

### Pace

11 weeks at ~17 hrs. Some modules will take longer than estimated. **That is fine and expected.** The estimates are a budget, not a schedule. Module 1.5 (chain rule) and 1.7 (micrograd) are the two most likely to run long — they're also the two that matter most. Let them.

---

## Module 1.1 — Functions, graphs, and change
**~18 hrs**

### The problem you can feel

You have a function that predicts something. It's wrong. You want to know: *if I nudge this parameter, does the error get better or worse, and by how much?*

That question is the whole of calculus, and it's the whole of training a neural network. Everything else is machinery for answering it fast.

### What you build

1. **A plotting habit.** Every function in this module gets plotted. `f(x) = x²`, `f(x) = 3x + 2`, `f(x) = (3x+2)⁴`, sigmoid, ReLU. Look at them. Get a feel for which ones are steep where.

2. **A numerical derivative — before any symbolic rules.**

```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
```

That's it. That's a derivative. Run it on `f(x) = x²` at `x = 3` and get `6.0000`. Run it at `x = 5` and get `10.0000`. Notice the pattern yourself before anyone tells you the rule is `2x`.

3. **A slope visualizer.** Plot a curve, pick a point, draw the tangent line using your numerical derivative. Animate the point moving along the curve and watch the tangent tilt. When the curve is flat, the slope is zero — *see* it.

4. **Limits, but only intuitively.** Make `h` smaller and smaller in your numerical derivative. Watch the answer converge. Then make it `1e-15` and watch it get *worse* — your first taste of numerical instability, which comes back properly in module 1.10.

### Then the notation

Only now: `f'(x)`, `df/dx`, and why there are two notations for the same thing. The power rule, the sum rule, the product rule — presented as *shortcuts for what you've been computing numerically*, verified against your own function each time.

### Gate targets

- **G1:** sketch `f(x) = x²` and its derivative on the same axes, from memory
- **G2:** derive the power rule for `x²` from the limit definition
- **G3:** your `numerical_derivative` matches symbolic answers to 1e-4 on five functions
- **G5:** explain to Claude why the derivative of a constant is zero, in terms of a graph

### Anki: ~10 cards

---

## Module 1.2 — Vectors
**~20 hrs**

### The problem you can feel

You have two customers, described by numbers — orders placed, revenue, days since signup, support tickets. Which two customers are most similar?

You already have the instinct for this from Salesforce: two records, some fields, how alike are they. A vector is a record. A dataset is a list of records. That's genuinely all it is at the start.

### What you build

1. **From scratch, in plain Python first** — no NumPy:

```python
def vector_add(a, b):     return [x + y for x, y in zip(a, b)]
def scalar_mul(s, a):     return [s * x for x in a]
def dot(a, b):            return sum(x * y for x, y in zip(a, b))
def magnitude(a):         return dot(a, a) ** 0.5
def cosine_similarity(a, b):
    return dot(a, b) / (magnitude(a) * magnitude(b))
```

Write tests for every one. `pytest` from here on.

2. **Then the same in NumPy**, and time both. Your loop is ~100× slower. Understand *why* — that's the vectorization lesson, and it's the reason GPUs matter.

3. **Vectors as arrows.** Plot 2-D vectors with `matplotlib.quiver`. Add them, see the parallelogram. Scale them, see them stretch. Take the dot product of perpendicular vectors and get zero — then *see* why on the plot.

4. **The payoff project: `king − man + woman ≈ queen`.** Download GloVe embeddings. Do the arithmetic. Find the nearest vector by cosine similarity. When `queen` comes out, you'll have felt what an embedding is — and you'll understand what your Data Cloud retrievers are doing at a level most people who use them never reach.

### The connection to your day job

Every semantic search you've built in Agentforce is cosine similarity over embeddings, at scale, with an index in front of it. You're now building the thing underneath.

### Then the notation

`**a** · **b**`, `‖**a**‖`, the geometric definition `**a**·**b** = ‖**a**‖‖**b**‖cos θ` — and why that's the *same* as the sum-of-products you coded. Prove they're equal for 2-D. That proof is the moment vectors click.

### Gate targets

- **G1:** draw two vectors and their dot product's geometric meaning
- **G2:** show `**a**·**b** = ‖**a**‖‖**b**‖cos θ` for the 2-D case
- **G3:** all five functions, from scratch, tested
- **G4:** NumPy versions, `assert np.allclose` against yours
- **G5:** defend — "why is cosine similarity better than Euclidean distance for text embeddings?"

### Project P1 (starts here)

**NumPy-only linear algebra library**, `src/mymath/`. Begins with vectors, grows through modules 1.3–1.4. Every function tested. This becomes a thing you actually reference later.

### Anki: ~15 cards

---

## Module 1.3 — Matrices as transformations
**~25 hrs**

### The problem you can feel

You have 10,000 customers, each with 50 numbers. You want to apply the same operation to all of them at once. That's a matrix.

But the deeper idea — the one that makes neural networks make sense — is different: **a matrix is a function that moves space.**

### What you build

1. **`matmul` by hand, on paper, for a 2×3 times 3×2.** Actually do this. Twice. The mechanical repetition is what makes shapes intuitive later.

2. **`matmul` from scratch, triple loop:**

```python
def matmul(A, B):
    n, k = len(A), len(A[0])
    k2, m = len(B), len(B[0])
    assert k == k2, f"shape mismatch: ({n},{k}) @ ({k2},{m})"
    return [[sum(A[i][p] * B[p][j] for p in range(k))
             for j in range(m)] for i in range(n)]
```

3. **Shape discipline — the most transferable skill in this program.** From now on, every array in every file gets a shape comment:

```python
X = np.random.randn(32, 784)   # (B, D_in)
W = np.random.randn(784, 128)  # (D_in, D_hidden)
H = X @ W                      # (B, D_hidden)
```

Do this religiously. In Stage 5 you'll be tracking `(B, T, H, D)` through multi-head attention, and the people who fail there are the ones who never built this habit.

4. **The transformation visualizer.** Draw a grid of points. Apply a 2×2 matrix. Plot before and after. Try: rotation, scaling, shear, reflection, and a singular matrix that collapses the plane onto a line. *Watch space move.* This is the single most valuable 3Blue1Brown insight and it's worth building yourself rather than only watching.

5. **A neural network layer is a matmul.** Build a "layer" that takes `(B, D_in)` and produces `(B, D_out)`. It's `X @ W + b`. That's a layer. You've now built the atom of every network in this program.

6. **Broadcasting.** How does `(32, 128) + (128,)` work? Get this wrong once deliberately and read the error. Broadcasting bugs are the most common silent failure in ML code — they don't crash, they just give you the wrong answer.

### Then the notation

`A ∈ ℝ^{m×n}`, `A^T`, `AB ≠ BA`, identity, inverse. Transpose specifically: you'll meet `Q @ K^T` in Stage 5, and it should feel like something you already know.

### Gate targets

- **G1:** draw what a shear matrix does to a unit square
- **G2:** show by hand why `(AB)^T = B^T A^T`
- **G3:** `matmul`, `transpose`, `identity` from scratch, tested
- **G4:** NumPy equivalence asserted
- **G5:** defend — "why can't you multiply a (3,4) by a (3,4)?" then "what does it mean geometrically that `det(A) = 0`?"

### Anki: ~20 cards

---

## Module 1.4 — Eigenvectors, SVD, and PCA
**~22 hrs**

> **This module is weighted more heavily than most curricula weight it, deliberately.** LoRA — the fine-tuning technique you'll use in Stage 7 and the most employable single skill in this program — is *Low-Rank Adaptation*. The low-rank idea is SVD. Learn it properly here and Stage 7.2 costs you a day instead of a week.

### The problem you can feel

You have customer data with 50 columns. Most of them are redundant — revenue and order-count move together. Can you describe the same customers with 5 numbers instead of 50, losing almost nothing?

### What you build

1. **Eigenvectors, visually first.** Using your transformation visualizer from 1.3: apply a matrix to many vectors and find the ones whose *direction* doesn't change — they only stretch. Those are eigenvectors; the stretch factor is the eigenvalue. Find them by brute-force search before you ever see the characteristic polynomial.

2. **SVD as "any matrix is three simple operations."** `A = UΣV^T` — rotate, stretch, rotate. Verify with `np.linalg.svd` and reconstruct. Then:

3. **Low-rank approximation — the LoRA preview.** Take an image as a matrix. Keep only the top `k` singular values. Reconstruct. Watch the image degrade gracefully as `k` shrinks from 200 to 50 to 10 to 3. **This is exactly what LoRA does to a weight matrix.** When you get to Stage 7.2, you will already understand it; you'll just be applying it to `ΔW` instead of a photograph.

4. **PCA from scratch**, then 2-D PCA of MNIST. Digit clusters appear. You've done dimensionality reduction with no library.

### Gate targets

- **G1:** sketch what `U`, `Σ`, `V^T` each do to a unit circle
- **G2:** show the relationship between SVD of `A` and eigendecomposition of `A^T A`
- **G3:** PCA from scratch, matching sklearn's components up to sign
- **G5:** defend — "if I keep the top 10 singular values of a 1000×1000 matrix, how many numbers am I storing, and what's the compression ratio?"

### Anki: ~15 cards

---

## Module 1.5 — Derivatives and the chain rule
**~25 hrs · budget more if needed**

> One of the two hardest modules in this stage. It's also the one Stage 3.3 (manual backpropagation) sits directly on top of. Do not rush it.

### The problem you can feel

Your network has 100,000 parameters. Changing any one of them changes the loss. You need all 100,000 sensitivities. Computing them one at a time numerically would take forever. The chain rule is how you get all of them in one backward pass.

### What you build

1. **Partial derivatives.** `f(x, y) = x²y + 3y`. Hold `y` fixed, vary `x`, get `∂f/∂x`. Verify numerically. Plot the 3-D surface and the two slice-curves so "holding one variable fixed" is a picture, not a phrase.

2. **The gradient as a vector.** Stack the partials. Plot it as an arrow on a contour map. It points uphill, steepest. Every training step you ever run walks the other way.

3. **The chain rule, built as a composition of functions you already understand:**

```python
# f(x) = (3x + 2)^4
def inner(x): return 3*x + 2
def outer(u): return u**4

# d_inner/dx = 3 ; d_outer/du = 4u^3
# chain: df/dx = 4*(3x+2)^3 * 3
```

Verify against your numerical derivative at five points. Then do it for a three-deep composition. Then five-deep. The mechanical repetition is what makes backprop feel obvious later instead of magical.

4. **Matrix calculus layout conventions — the thing nobody tells you.**

There are two conventions for how to lay out `∂(vector)/∂(vector)`: **numerator layout** and **denominator layout**. They are transposes of each other. Textbooks and papers switch between them silently, sometimes mid-derivation.

This is genuinely the reason backprop derivations feel confusing. It's not you. Learn both, pick denominator layout (what most ML code uses, where `∂L/∂W` has the same shape as `W`), and from then on **always sanity-check a gradient by its shape**: if `W` is `(784, 128)` then `∂L/∂W` must be `(784, 128)`. That single check will catch most of your errors for the next fifteen months.

### Gate targets

- **G1:** draw a computation graph for `(3x+2)⁴` with forward values and backward gradients labelled
- **G2:** derive `∂L/∂W` for `L = ½(Xw − y)²` by hand, correct shapes
- **G3:** symbolic chain-rule result matches numerical to 1e-4 on a five-deep composition
- **G5:** defend — "why does backprop go backwards? What would forward-mode cost for 100,000 parameters?"

### Anki: ~20 cards

---

## Module 1.6 — Gradient descent
**~16 hrs**

### What you build

1. **Vanilla gradient descent** on `f(x) = x²`. Watch it converge. Set the learning rate to 1.5 and watch it *diverge*. Set it to 0.001 and watch it crawl. Feel the tradeoff rather than reading about it.

2. **P2 — the 2-D loss-landscape visualizer.** Contour plot, optimizer path animated on top. Then run the same landscape with: vanilla GD, GD + momentum, and Adam. Watch momentum roll through a shallow valley that vanilla GD crawls along.

3. **Adam, derived rather than quoted.** Momentum is an exponentially weighted average of gradients. RMSprop is an EWA of squared gradients. Adam is both, plus a bias correction for the fact that the averages start at zero. Implement each step; don't import it.

4. **Why AdamW ≠ Adam + L2.** A short but real distinction that comes up constantly in practice and occasionally in interviews.

### Gate targets

- **G1:** sketch a loss landscape where momentum helps and vanilla GD struggles
- **G2:** write Adam's update rule including bias correction
- **G3:** all three optimizers from scratch on a 2-D landscape
- **G5:** defend — "your loss is oscillating wildly. Three possible causes, in order of likelihood?"

### Anki: ~15 cards

---

## Module 1.7 — micrograd
**~22 hrs · the payoff**

Karpathy's first *Zero to Hero* lecture. You build a working automatic differentiation engine.

This is where modules 1.1–1.6 collapse into one thing. The `Value` class holds a number and knows what produced it. `backward()` walks the graph in reverse applying the chain rule. You will have built, in about 150 lines, the core of what PyTorch does.

### What you build

- `Value` class: `__add__`, `__mul__`, `__pow__`, `tanh`, `exp`
- Topological sort for the backward pass
- A small MLP on top of it
- Train it on the `make_moons` dataset to >90% accuracy
- **Assert your gradients match PyTorch's to 1e-6**

### Gate targets

- **G1:** draw the computation graph for `a*b + c` with gradients flowing back
- **G2:** derive the local gradient for the `tanh` node
- **G3:** the whole engine, from scratch
- **G4:** gradients match `torch.autograd` to 1e-6 — this assertion *is* the gate
- **G5:** defend — "what does `.backward()` actually do, step by step?" and "why topological sort?"

### Project P3

`src/micrograd/` — yours, tested, in the repo. This is the first thing in your portfolio that will impress an interviewer.

### Anki: ~15 cards

---

## Module 1.8 — Probability for AI
**~18 hrs**

### The problem you can feel

Your model outputs `[0.7, 0.2, 0.1]` for three classes. What *is* that, exactly? Not "confidence" — that's a hand-wave. It's a probability distribution, and the loss function you train it with is a statement about distributions.

### What you build

- Bernoulli, Categorical, Gaussian — PDFs and samplers, from scratch
- Expectation and variance, computed both analytically and by sampling — watch them converge
- **Maximum Likelihood Estimation**: `fit_gaussian(data)` returning μ and σ, and the discovery that they equal `data.mean()` and `data.std()`. Derive *why*.

MLE is the most important idea in this module. Every loss function you'll meet is "find the parameters that make the observed data most likely." Once you see that, cross-entropy stops being an arbitrary formula.

### Gate targets

- **G2:** derive the MLE for a Gaussian's mean from the log-likelihood
- **G3:** three distributions plus MLE fitting, from scratch
- **G5:** defend — "why do we maximize *log*-likelihood instead of likelihood?"

### Anki: ~15 cards

---

## Module 1.9 — Information theory
**~16 hrs**

### What you build

- `entropy(p)` — average surprise. Plot it for a two-outcome distribution as `p` goes 0 → 1. Maximum at 0.5. *See* why.
- `cross_entropy(p, q)` and `kl_divergence(p, q)`; verify `KL = CE − H`
- `softmax_with_temperature(logits, T)`; sweep `T` from 0.1 to 10 and watch peaked → uniform
- **Prove cross-entropy = negative log-likelihood.** This is the keystone. Do it on paper.
- **Assert your `cross_entropy` matches `F.cross_entropy` to 1e-6**

### Why this module earns its place

Every loss function in deep learning is a probability statement in disguise. Curricula that skip this leave you able to use `nn.CrossEntropyLoss` but unable to say what it means. Your v1 plan was right to include it; it's kept intact.

And temperature — which you tune in Agentforce today — you will now understand as an actual operation on a distribution rather than a dial.

### Gate targets

- **G1:** sketch entropy vs `p` for a coin
- **G2:** prove CE = NLL for the categorical case
- **G3:** all four functions from scratch
- **G4:** matches `F.cross_entropy` to 1e-6
- **G5:** defend — "KL divergence isn't symmetric. Why does that matter in practice?"

### Project P4

Entropy / KL / temperature-sampler toolkit, tested.

### Anki: ~20 cards

---

## Module 1.10 — Numerical stability
**~8 hrs · rarely taught, disproportionately valuable**

### The problem you can feel

Your softmax works fine in testing. In training it produces `nan` at step 4,000 and you lose six hours of compute.

### What you build

1. **Overflow.** `np.exp(800)` is `inf`. Real logits reach that. Fix:

```python
def softmax(x):
    x = x - np.max(x)      # this line is the entire lesson
    e = np.exp(x)
    return e / e.sum()
```

Prove to yourself it's mathematically identical and numerically survivable.

2. **The log-sum-exp trick**, and why `log_softmax` exists as a separate function.

3. **Float formats.** fp32 vs fp16 vs bf16 — what each can represent, why fp16 overflows during training where bf16 doesn't, and why mixed precision is the default in modern training. Print the ranges yourself.

4. **Catastrophic cancellation.** `(1e16 + 1) - 1e16` in floating point. Try it. Be unsettled.

5. **Epsilons.** Why `x / (y + 1e-8)` appears in every normalization layer you'll ever write.

### Gate targets

- **G2:** show algebraically that max-subtraction leaves softmax unchanged
- **G3:** a naive softmax that overflows, and a stable one that doesn't, side by side
- **G5:** defend — "your loss went to NaN at step 4000. Walk me through your debugging."

### Project P5

`drills/numerical_stability_lab.py` — a file of deliberate numerical failures and their fixes. You'll return to this.

### Anki: ~10 cards

---

# Stage 1 exit gate

You do not proceed to Stage 2 until all of these are true:

- [ ] `src/mymath/` — vectors, matrices, PCA. All tested. All from scratch.
- [ ] `src/micrograd/` — gradients match PyTorch to 1e-6.
- [ ] You can derive the chain rule on paper for a 3-layer MLP, cold.
- [ ] You can explain cross-entropy as negative log-likelihood under a categorical distribution, without notes.
- [ ] You can explain why `x - max(x)` appears in every softmax implementation.
- [ ] `cross_entropy` matches `F.cross_entropy` to 1e-6.
- [ ] ~150 Anki cards, reviewed, retention above 85%.
- [ ] Five Gate-5 defences passed with Claude — one each on: the dot product, the chain rule, SVD, MLE, and numerical stability.
- [ ] **One Re-Implementation Drill passed** — rebuild `cosine_similarity` and `numerical_derivative` from a blank file in 45 minutes, no reference.

---

# A note for month three

Around week 10–12, roughly in module 1.5 or 1.7, this will stop feeling like progress. The novelty will be gone, the math will be genuinely hard, and there will be no visible payoff — you won't have built anything that looks like AI yet.

That feeling is not information about your ability. It's the standard shape of the curve, and it's the point at which most people quit and restart with a different course, which resets them to zero.

Two things that help, both concrete:

1. **Re-read your week-1 journal entry.** You will not be able to believe you didn't know what a dot product was.
2. **Shrink, don't stop.** A three-hour week is a week. A zero week starts a habit of zero weeks.

Then module 1.7 arrives, you build a working autograd engine in 150 lines, and it clicks. Everyone who got through says the same thing about that moment.

---

> Stage 2 detail gets written when you're within a week of finishing this stage — so it's current, and so it reflects what you actually struggled with here.
