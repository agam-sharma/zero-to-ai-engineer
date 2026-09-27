# ZERO TO AI ENGINEER: A 9-Month Masterclass

## Personalized Curriculum: Senior Salesforce Engineer → AI Engineer

> **Student Profile:** 7 years Salesforce Engineering (Apex/Java/LWC), solid Python, strong backend/SDLC instincts
>
> **Hardware:** Mac M3 Pro (18 GB unified memory, MPS GPU)
>
> **Target Pace:** 25–30 hrs/week (≈3 hrs weekdays + ~10 hrs weekend). Sustainable over 9 months.
>
> **Two Capstones:**
>
> 1. **Mini-GPT** — a ~50–100M parameter Transformer pretrained from scratch on your M3 Pro (end of Phase 6).
> 2. **CodeSage** — a domain-specialist AI Engineer project combining fine-tuning + RAG + an agent + a deployed web UI, published publicly (end of Phase 8).

> **Why this plan is different from "learn AI in X months" lists:**
> Most plans teach you to *use* models. A few plans teach you to *build* GPT from scratch. This plan does **both** — and adds the production layer (fine-tuning, RAG, agents, evals, deployment) that the job market actually pays for. You graduate as an AI Engineer who understands internals, not a prompt jockey who doesn't, and not a researcher who can't ship.

---

## THE BIG PICTURE — What You Will Actually Understand

By the end of this course you will be able to answer, from first principles, any of these questions an interviewer can throw at you:

1. **Bottom-up (the minute details):**
  - "Derive the gradient of cross-entropy loss w.r.t. the pre-softmax logits."
  - "What does the attention matrix look like shape-wise, and why `Q @ K^T / √d`?"
  - "Why do Transformers use LayerNorm but CNNs use BatchNorm?"
  - "Walk me through what happens in memory during a single forward+backward pass."
2. **Top-down (the bigger picture):**
  - "When would you pick fine-tuning over RAG over prompting?"
  - "How do you evaluate whether a prompt change actually improved the system?"
  - "Design a production system that answers questions over 100K internal documents with <2s latency and no hallucinations."
  - "How does the compute-optimal frontier (Chinchilla) constrain my architecture choices?"

Both ends of the stack are covered. On purpose.

---

## COURSE OVERVIEW — The 9 Phases


| Phase | Weeks       | Title                                                            | What You'll Build                                                                                |
| ----- | ----------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **0** | Week 0      | Environment, Mindset & Learning System                           | M3 Pro AI dev setup + `zero-to-ai-engineer` repo + daily ritual                                  |
| **1** | Weeks 1–6   | The Math Engine: LinAlg, Calculus, **Probability & Info Theory** | A NumPy math library + from-scratch `micrograd` + entropy/KL/cross-entropy from first principles |
| **2** | Weeks 7–9   | Classical ML Crash Course                                        | Titanic Kaggle + spam classifier (logistic→MLP comparison)                                       |
| **3** | Weeks 10–15 | Neural Networks Deep Dive                                        | MLP → CNN → CIFAR-10 classifier with W&B tracking                                                |
| **4** | Weeks 16–20 | NLP & Sequence Models                                            | BPE tokenizer + RNN/LSTM vs attention comparison on Shakespeare                                  |
| **5** | Weeks 21–25 | The Transformer                                                  | Full Transformer from scratch + "Attention Is All You Need" fully understood                     |
| **6** | Weeks 26–29 | Build Your mini-GPT                                              | ~50–100M parameter GPT pretrained on TinyStories/OpenWebText subset                              |
| **7** | Weeks 30–34 | **AI Engineer Toolkit**: Fine-tuning, RAG, Agents, Evals, Deploy | LoRA fine-tune + production RAG app + tool-using agent + observability + deployed API            |
| **8** | Weeks 35–36 | Capstone & Public Launch                                         | **CodeSage** — polished portfolio project + blog post + demo video + interview prep              |


---

## WHY THIS SEQUENCE (And Not the Usual One)

Most curriculums jump from "install PyTorch" to "build a Transformer" and skip the two bridges that make everything click:

- **The probability bridge (Phase 1, Weeks 5–6):** Cross-entropy, KL divergence, MLE, softmax-with-temperature — every loss function in deep learning is a probability statement in disguise. If you don't see that, you half-understand everything downstream.
- **The classical-ML bridge (Phase 2):** Linear regression, logistic regression, bias-variance, proper train/val/test hygiene. Skipping this is why so many beginners can train a model but can't diagnose why it's underperforming.

We don't skip them.

---

## DAILY RHYTHM (THE "THREE-PASS" METHOD)

The full learning system is in `[LEARNING_SYSTEM.md](./LEARNING_SYSTEM.md)`. Summary:

```
╔═══════════════════════════════════════════════════════════════════╗
║  WEEKDAY (Mon–Fri) — ~3 hrs/day                                  ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Morning (7:00–8:00 AM) — PASS 1: WHY (Big Picture)              ║
║    • Watch the conceptual video / read the blog post             ║
║    • No code. Just absorb.                                       ║
║    • Output: 3–5 bullets in your journal + 1 open question       ║
║                                                                   ║
║  Evening (7:30–9:30 PM) — PASS 2: HOW (Minute Details)           ║
║    • Code along with tutorial                                    ║
║    • Derive the key equation on paper                            ║
║    • Commit WIP to git                                           ║
║                                                                   ║
║  Before bed (9:30–10:00 PM) — PASS 3: WHY-IT-WORKED               ║
║    • Close everything. Explain today's concept in 5 sentences    ║
║      to your past self (journal entry). Feynman technique.       ║
║                                                                   ║
╠═══════════════════════════════════════════════════════════════════╣
║  SATURDAY — ~6 hrs                                                ║
║    • AM: build the weekly mini-project                           ║
║    • PM: Anki review + tidy code + push                          ║
║                                                                   ║
║  SUNDAY — ~4 hrs                                                  ║
║    • AM: read the paper-of-the-week + write 1-page summary       ║
║    • Retrospective + Friday blog post                            ║
║    • PM: REST (non-negotiable)                                    ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## THE PROJECT LADDER — A Small Project Per Major Topic

These are **in addition to** the weekly capstones. Each is a focused 1–3 day build that lives in `projects/NN-<slug>/` in your repo.


| #   | Phase | Project                                                        | Skill Proved                           |
| --- | ----- | -------------------------------------------------------------- | -------------------------------------- |
| 1   | 1     | NumPy-only linear algebra library with unit tests              | Compose primitives correctly           |
| 2   | 1     | 2D gradient-descent visualizer (animated loss landscape)       | Optimization intuition                 |
| 3   | 1     | Cross-entropy + KL + temperature sampler from scratch          | Probability / info theory fluency      |
| 4   | 2     | Titanic (Kaggle) with proper cross-validation (top 40%)        | Classical ML hygiene                   |
| 5   | 2     | Spam classifier: logistic reg → MLP same data, side-by-side    | "Why bother with NN?" felt, not told   |
| 6   | 3     | MNIST: MLP vs CNN, benchmarked on MPS vs CPU                   | PyTorch + hardware + baseline thinking |
| 7   | 3     | "Loss-curve pathology zoo" (dead ReLU, LR too high, no norm…)  | Debugging muscle                       |
| 8   | 4     | BPE tokenizer benchmarked against `tiktoken`                   | Data plumbing                          |
| 9   | 4     | Shakespeare: bigram → MLP → RNN → LSTM side-by-side            | Feeling the architecture evolution     |
| 10  | 5     | Annotated "Attention Is All You Need" with links to your code  | Paper literacy                         |
| 11  | 5     | Attention-head interpretability dashboard (BertViz / custom)   | Mechanistic intuition                  |
| 12  | 6     | mini-GPT on TinyStories                                        | Pretraining end-to-end                 |
| 13  | 6     | Scaling-laws calculator for your M3 Pro compute budget         | Think like a pretraining engineer      |
| 14  | 7     | LoRA fine-tune of Phi-3-mini (or Qwen2-0.5B) on your domain    | Production fine-tuning                 |
| 15  | 7     | "Chat with Salesforce Docs" RAG app (Chroma + reranker)        | Retrieval engineering                  |
| 16  | 7     | Agent that writes + runs SOQL against a Salesforce scratch org | Tool-use + your domain moat            |
| 17  | 7     | Prompt regression-test harness (golden set + LLM-as-judge)     | Evals discipline                       |
| 18  | 8     | **CodeSage** — capstone (mini-GPT + RAG + agent + UI) deployed | Hireable portfolio piece               |


---

## PHASE 0 — Environment, Mindset & Learning System (Week 0)

Full details: `[Phase_0_Environment_and_Mindset_Setup.md](./Phase_0_Environment_and_Mindset_Setup.md)` and `[LEARNING_SYSTEM.md](./LEARNING_SYSTEM.md)`.

### Week 0 Deliverables


| #   | Topic                                                         | Build                                                                              | Success Metric                                                                     |
| --- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 0.1 | Python + conda environment (`ai-mastery`, Python 3.11)        | Install miniconda, create env, install PyTorch+MPS+Jupyter+HF+W&B                  | `python -c "import torch; print(torch.backends.mps.is_available())"` prints `True` |
| 0.2 | `zero-to-ai-engineer` GitHub repo scaffolded                  | Create the folder structure from `LEARNING_SYSTEM.md` §5, push to GitHub           | Repo exists publicly; `README.md` shows phase progress bar                         |
| 0.3 | Anki installed + first deck (`ai-foundations.apkg`)           | Install Anki, import the starter deck, review 10 cards                             | You can explain "why cross-entropy is negative log-likelihood" from a flashcard    |
| 0.4 | Paper-of-the-week workflow                                    | Read the Karpathy "Software 2.0" blog; write a 1-page summary in `papers/week-00/` | Summary file committed with tag `week-00-complete`                                 |
| 0.5 | Run the `00_Foundation_Assessment_Exercises.ipynb` diagnostic | Score yourself honestly                                                            | Score logged in `journal/2026-W00.md`                                              |


**Phase 0 Milestone:** environment verified, repo live, journal rhythm started, diagnostic score known.

---

## PHASE 1 — The Math Engine (Weeks 1–6)

Full details: `[Phase_1_The_Math_Engine.md](./Phase_1_The_Math_Engine.md)`.

> **Salesforce Analogy:** math is the "query optimizer" of AI. You *can* ship without knowing it, but you'll hit a wall when the framework stops guessing what you meant. You do NOT need to become a mathematician — you need *intuition* for 6 tools: vectors, matrices, derivatives, probability, entropy, and autograd.


| Week | Theme                                     | Headline Build                                                                                       | Success Metric                                                                                          |
| ---- | ----------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 1    | Linear Algebra — Vectors                  | `vector_add`, `dot`, `cosine_similarity`, king−man+woman≈queen analogy                               | Analogy returns vector closest to "queen"                                                               |
| 2    | Linear Algebra — Matrices, Eigen, PCA     | `matmul` from scratch; 2D PCA of MNIST showing digit clusters                                        | Clusters visible; your math library has 10+ passing unit tests                                          |
| 3    | Calculus — Derivatives & Chain Rule       | Numerical derivative; manually compute ∂ of `(3x+2)⁴` and match                                      | Manual chain rule ≈ numerical within 1e-4                                                               |
| 4    | Calculus — Gradient Descent & `micrograd` | Build Karpathy's micrograd end-to-end; train MLP on moons dataset                                    | MLP hits >90% on moons; gradients match PyTorch autograd                                                |
| 5    | **Probability for AI** (NEW)              | Implement Bernoulli/Categorical/Gaussian PDFs; MLE for Gaussian from data                            | `fit_gaussian(data)` returns μ,σ equal to `data.mean()`, `data.std()`                                   |
| 6    | **Information Theory for AI** (NEW)       | From-scratch `entropy`, `cross_entropy`, `kl_divergence`, `softmax_with_temperature`; prove CE = NLL | Your CE matches `F.cross_entropy` to 1e-6; temperature sweep produces correct uniform→peaked transition |


### Phase 1 Mini-Projects (see ladder): #1, #2, #3

### Phase 1 Milestone Check

> - ✅ Explain dot product, matmul, transpose, and WHERE they appear in a neural net
> - ✅ Derive chain rule and backprop on paper for a 3-layer MLP
> - ✅ Explain cross-entropy as "negative log-likelihood under a categorical distribution"
> - ✅ Explain KL divergence geometrically and algebraically
> - ✅ Build an autograd engine from scratch and know what PyTorch is doing under the hood
> - ✅ Use PyTorch tensors on M3 Pro MPS GPU

**Phase 1 Exam:** `assessments/phase_1_post_assessment.ipynb`

---

## PHASE 2 — Classical ML Crash Course (Weeks 7–9) *(NEW)*

Full details: `[Phase_2_Classical_ML.md](./Phase_2_Classical_ML.md)`.

> **Salesforce Analogy:** Before you debug a complex **Flow** with 30 nodes, you'd master the simple `IF/ELSE` logic. Classical ML is the "simple logic" version of neural networks. Every neural net debugging technique (bias-variance, train/val/test, regularization, metrics) was invented and refined here first.


| Week | Theme                                      | Headline Build                                                                           | Success Metric                                                               |
| ---- | ------------------------------------------ | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 7    | Linear & Logistic Regression from Scratch  | Implement both with gradient descent in NumPy; L1/L2 regularization                      | On a toy dataset, your logreg matches sklearn's to within 1% accuracy        |
| 8    | Trees, Boosting, Metrics, Cross-Validation | XGBoost on Titanic; confusion matrix, P/R/F1, ROC/AUC; K-fold CV                         | Proper CV pipeline; you can explain when F1 > accuracy                       |
| 9    | Data Hygiene, Leakage, Error Analysis      | Spam classifier (logistic → MLP) with careful data splits and an error-analysis notebook | You identify at least 3 categories of misclassifications by reading the data |


### Phase 2 Mini-Projects: #4 Titanic, #5 Spam Classifier

### Phase 2 Milestone Check

> - ✅ Fit linear and logistic regression *without* sklearn, matching sklearn numerically
> - ✅ Explain bias-variance tradeoff with a code demo (high-degree polynomial overfit)
> - ✅ Pick the correct metric for a problem (and justify it) — accuracy, F1, PR-AUC, ROC-AUC, MAE, MAPE
> - ✅ Design a data split that avoids leakage, including time-series splits
> - ✅ Do systematic error analysis on a model you trained

**Phase 2 Exam:** `assessments/phase_2_post_assessment.ipynb`

---

## PHASE 3 — Neural Networks Deep Dive (Weeks 10–15)

Full details: `[Phase_3_Neural_Networks.md](./Phase_3_Neural_Networks.md)`.

> **Salesforce Analogy:** a neural net is a learned **Flow** — same structure of data flowing through transformation nodes, but each node's logic is trained rather than configured.


| Week | Theme                                                         | Headline Build                                                              | Success Metric                                                 |
| ---- | ------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 10   | Perceptron, Activations, Loss Functions                       | AND/OR/XOR demos; plot 6 activations + derivatives; CE vs MSE comparison    | XOR fails with 1 layer, works with 2 — you explain why         |
| 11   | MLP, Backpropagation in Detail                                | Code along Karpathy makemore Part 2; manually backprop through MLP (Part 4) | Your manual gradients == PyTorch autograd to 1e-6              |
| 12   | BatchNorm, LayerNorm, Dropout, Regularization                 | makemore Part 3 (BatchNorm); dropout train vs test comparison               | Activation histograms are Gaussian post-BN; dropout gap closes |
| 13   | CNNs (brief, 3 days) + **Training Diagnostics** (2 days, NEW) | CNN on MNIST on MPS; build the "loss-curve pathology zoo" (project #7)      | You diagnose 6 failure modes by eye                            |
| 14   | Hyperparameter Tuning + Experiment Tracking                   | Learning rate finder; W&B dashboard with 5 sweeps                           | LR finder shows clear valley; optimal LR trains 2–3× faster    |
| 15   | **CIFAR-10 Capstone**                                         | Custom CNN, data aug, LR scheduling, tracked in W&B                         | ≥85% test accuracy; GPU utilization >80%                       |


### Phase 3 Mini-Projects: #6 MNIST, #7 Pathology zoo

### Phase 3 Milestone Check

> - ✅ Build any feed-forward network in PyTorch
> - ✅ Manually backprop through any layer (no `.backward()`)
> - ✅ Diagnose 6+ training failure modes from the loss curve alone
> - ✅ Tune hyperparameters systematically (LR finder, W&B sweeps)
> - ✅ Train on M3 Pro MPS at >80% GPU utilization

**Phase 3 Exam:** `assessments/phase_3_post_assessment.ipynb`

---

## PHASE 4 — NLP & Sequence Models (Weeks 16–20)

Full details: `[Phase_4_NLP_and_Sequences.md](./Phase_4_NLP_and_Sequences.md)`.

> **Salesforce Analogy:** sequence models are **Process Builder** — each step's output depends on previous steps. Language is inherently sequential.


| Week | Theme                                                                   | Headline Build                                                               | Success Metric                                                    |
| ---- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 16   | Tokenization (BPE)                                                      | Karpathy tokenizer video; your BPE vs `tiktoken`                             | Token counts within 10% of GPT-2 tokenizer                        |
| 17   | Word Embeddings + **Dense Retrieval primer** (NEW)                      | Word2Vec skip-gram; use Sentence-Transformers to build a toy semantic search | t-SNE shows clusters; semantic search returns relevant docs       |
| 18   | Vanilla RNN, LSTM, GRU                                                  | Shakespeare char-level generation with all three                             | LSTM > vanilla RNN qualitatively; you explain vanishing gradients |
| 19   | Seq2Seq + Bahdanau Attention (Origin Story)                             | LSTM + attention; visualize attention heatmap                                | Heatmap shows interpretable patterns                              |
| 20   | **Phase 4 Capstone: Shakespeare bigram → MLP → RNN → LSTM → Attention** | Same data, same budget, all 5 side-by-side                                   | Written analysis of quality vs capability progression             |


### Phase 4 Mini-Projects: #8 BPE, #9 Shakespeare ladder

### Phase 4 Milestone Check

> - ✅ Implement BPE from scratch
> - ✅ Train RNN and LSTM from scratch in PyTorch
> - ✅ Explain exactly why attention beats recurrence for long-range dependencies
> - ✅ Build a working embedding-based retrieval system (foundation for RAG in Phase 7)

**Phase 4 Exam:** `assessments/phase_4_post_assessment.ipynb`

---

## PHASE 5 — The Transformer (Weeks 21–25)

Full details: `[Phase_5_The_Transformer.md](./Phase_5_The_Transformer.md)`.

> **Salesforce Analogy:** this is the custom **Lightning Web Component** where you wire together all the child components (attention, FFN, LayerNorm, positional encoding) you built in Phase 4 into the parent.


| Week | Theme                                                                                            | Headline Build                                                                       | Success Metric                                                                       |
| ---- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| 21   | Self-Attention + Multi-Head Attention                                                            | Single-head, then multi-head from scratch; causal mask                               | Output matches `torch.nn.MultiheadAttention` for same inputs                         |
| 22   | Positional Encoding, LayerNorm, FFN                                                              | Sinusoidal PE heatmap; GELU FFN; LayerNorm from scratch                              | All three match PyTorch reference impls within 1e-5                                  |
| 23   | Full Transformer Block + Decoder-Only Stack                                                      | Build `TransformerBlock` and `GPT` class; causal masking verified                    | Shapes correct; gradients flow; no NaNs                                              |
| 24   | Training Loop: LR Schedule, Grad Clip, Grad Accumulation                                         | Full training on Shakespeare                                                         | Loss decreases smoothly; sample text progresses from random → English-ish            |
| 25   | **"Attention Is All You Need" Paper + Mechanistic Interpretability primer** (NEW) + **Capstone** | Annotated paper with code links; TransformerLens on your model; compare with nanoGPT | You can explain every figure/equation; you identify ≥2 "specialized" attention heads |


### Phase 5 Mini-Projects: #10 Annotated paper, #11 Attention dashboard

### Phase 5 Milestone Check

> - ✅ Implement a complete Transformer from scratch
> - ✅ Explain every line of nanoGPT
> - ✅ Read and explain any figure/equation in "Attention Is All You Need"
> - ✅ Identify what specific attention heads learn (syntax, position, copying, …)

**Phase 5 Exam:** `assessments/phase_5_post_assessment.ipynb`

---

## PHASE 6 — Build Your mini-GPT (Weeks 26–29)

Full details: `[Phase_6_Build_Your_GPT.md](./Phase_6_Build_Your_GPT.md)`.

> **Salesforce Analogy:** this is your **managed package deployment** — all components built and tested, now assembled and optimized for real use.


| Week | Theme                                                       | Headline Build                                                                        | Success Metric                                                          |
| ---- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 26   | **Scaling Laws + Architecture Decisions** (NEW) + Data Prep | Chinchilla-style calculation for your compute budget; TinyStories tokenized to binary | Config document with justified choices fits in ~12 GB                   |
| 27   | Pretraining Run                                             | Full training of ~50–100M parameter mini-GPT                                          | Val loss competitive with nanoGPT baseline; training < 4 hrs            |
| 28   | Generation Strategies + KV-Cache                            | Greedy / temperature / top-k / top-p; implement KV-cache                              | Cached generation 3–10× faster; sampling comparisons documented         |
| 29   | RoPE + Flash Attention + Final Capstone                     | Swap sinusoidal for RoPE; use `F.scaled_dot_product_attention`; 20 curated samples    | Longer-context generation improves; `sdpa` faster than manual attention |


### Phase 6 Mini-Projects: #12 mini-GPT, #13 Scaling-laws calculator

### Phase 6 Milestone Check

> - ✅ Pretrain a mini-GPT from scratch on your M3 Pro
> - ✅ Explain and implement all common sampling strategies
> - ✅ Implement KV-cache for O(n) generation
> - ✅ Compute-optimal sizing: you can justify why your model is *this* size

**Phase 6 Exam:** `assessments/phase_6_post_assessment.ipynb`

---

## PHASE 7 — AI Engineer Toolkit (Weeks 30–34) *(MAJOR EXPANSION)*

Full details: `[Phase_7_AI_Engineer_Toolkit.md](./Phase_7_AI_Engineer_Toolkit.md)`.

> **This is the phase that separates "understands AI" from "gets hired as an AI Engineer."** You'll take off-the-shelf models and build real systems with them.


| Week | Theme                                      | Headline Build                                                                                                                | Success Metric                                                             |
| ---- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 30   | Fine-tuning: SFT, LoRA, QLoRA              | Full FT on GPT-2; LoRA fine-tune of Phi-3-mini (or Qwen2-0.5B) on your domain data                                            | Before/after eval shows measurable domain improvement; LoRA < full FT VRAM |
| 31   | **RAG & Vector Databases** (NEW)           | "Chat with Salesforce Docs": chunking + embeddings + Chroma + hybrid search + reranker + Ragas eval                           | Retrieval@5 > 0.8 on golden set; answers grounded in retrieved chunks      |
| 32   | **Agents, Tools, Structured Output** (NEW) | SOQL-writing agent against a scratch org: function calling, ReAct, JSON mode with Instructor/Pydantic, error recovery         | Agent completes 8/10 tasks; malformed output rate < 2%                     |
| 33   | **Evals & Observability** (NEW)            | Prompt regression-test harness (golden set + LLM-as-judge + PromptFoo); Langfuse tracing                                      | Every prompt change runs through the harness; traces visible in Langfuse   |
| 34   | **Deployment & Productionization** (NEW)   | FastAPI endpoint with streaming, Redis cache, token accounting; quantized 7B model via `llama.cpp`; DPO conceptually with TRL | Deployed API handles concurrent requests; 7B runs >20 tok/s Q4 on M3 Pro   |


### Phase 7 Mini-Projects: #14 LoRA, #15 RAG, #16 SOQL agent, #17 Eval harness

### Phase 7 Milestone Check

> - ✅ Fine-tune a small LLM with LoRA on your own data
> - ✅ Build a production-grade RAG system with reranking and grounded answers
> - ✅ Build a tool-using agent with structured output and error handling
> - ✅ Write regression tests for prompts; trace requests in an observability tool
> - ✅ Deploy a model behind a streaming FastAPI endpoint with caching and quantization

**Phase 7 Exam:** `assessments/phase_7_post_assessment.ipynb` (this is the big one — includes a system design question)

---

## PHASE 8 — Capstone & Public Launch (Weeks 35–36)

Full details: `[Phase_8_Capstone_and_Launch.md](./Phase_8_Capstone_and_Launch.md)`.

> Everything you've built becomes a single hireable portfolio piece.

### Week 35 — Build `CodeSage`

A polished project combining:

- A **fine-tuned mini-model** (LoRA on a domain corpus — e.g., Apex code + Salesforce docs)
- A **RAG system** over Salesforce developer docs
- A **tool-using agent** that can query a scratch org (SOQL) and open/read files
- A **streaming FastAPI backend** + **Gradio or Next.js UI**
- **Observability** via Langfuse
- **An evals harness** with a golden set of 30 Q/A pairs

### Week 36 — Public Launch + Interview Prep

- Friday launch: blog post (dev.to), Loom demo video, HF Space or Vercel deploy, Twitter/LinkedIn announcement
- Portfolio README makeover: badges, live demo link, architecture diagram, metrics
- Mock interview prep: 10 system-design questions + 10 fundamentals questions walked through out loud (recorded)

### Phase 8 Milestone / Graduation Criteria

> - ✅ `CodeSage` is publicly accessible and reliably answers 8/10 curated questions
> - ✅ Blog post published and shared
> - ✅ You can give a 20-minute talk walking someone from attention equations → your deployed UI
> - ✅ Portfolio ready for AI Engineer applications

**Phase 8 Exam:** `assessments/phase_8_post_assessment.ipynb` (architecture + mock interview prompt set)

---

## THE LEARNING SYSTEM (See `LEARNING_SYSTEM.md`)

Daily/weekly rituals, Git workflow, Anki deck structure, paper-of-the-week list, weekly retrospective template, flashcard-writing conventions, "when to skip, refresh, or deep-dive" decision tree.

---

## HOW YOU KNOW YOU'RE ON TRACK

Three independent signals (any two green = you're fine):

1. **The exam notebooks pass** (pre-phase diagnostic suggests you're ready; post-phase exam proves you are).
2. **Your Friday blog post writes itself** — if you sit down Friday and have nothing to say, the week was consumption, not learning.
3. **You could explain this week's topic to your past (pre-AI) self in 5 sentences without jargon.**

Red flags:

- 🚨 Skipping Pass 3 ("explain to past self") more than twice in a week
- 🚨 Watching videos at 2× without coding — you're consuming, not learning
- 🚨 Not writing any unit tests for your math/NN primitives — you'll discover bugs *during* training, which is too late
- 🚨 Not committing to git daily

---

## MASTER RESOURCE LIST

### Video Courses (Priority Order)


| #   | Resource                                                                                                           | When                                    |
| --- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------- |
| 1   | [Andrej Karpathy: Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)                            | Weeks 4, 11–13, 16, 21–25 — the spine   |
| 2   | [3Blue1Brown: Neural Networks (full playlist)](https://www.3blue1brown.com/topics/neural-networks)                 | Weeks 1–6, 21–22 — visual intuition     |
| 3   | [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Weeks 1–2                               |
| 4   | [3Blue1Brown: Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)       | Weeks 3–4                               |
| 5   | [StatQuest (Josh Starmer)](https://www.youtube.com/c/joshstarmer)                                                  | Throughout — crystal-clear explanations |
| 6   | [Harvard Stats 110 (Blitzstein)](https://projects.iq.harvard.edu/stat110)                                          | Weeks 5–6 — probability                 |
| 7   | [HuggingFace LLM Course](https://huggingface.co/learn/llm-course)                                                  | Phase 7                                 |
| 8   | [Full Stack Deep Learning (FSDL)](https://fullstackdeeplearning.com/)                                              | Phase 7 — production ML                 |
| 9   | Stanford CS231n / CS224n                                                                                           | Optional deep-dives in Phases 3–4       |


### Books (Free Online)


| Book                                                                                                                              | When                           |
| --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| [Mathematics for Machine Learning](https://mml-book.github.io/)                                                                   | Weeks 1–6                      |
| [Deep Learning Book (Goodfellow)](https://www.deeplearningbook.org/)                                                              | Weeks 10–25                    |
| [Dive into Deep Learning](https://d2l.ai/)                                                                                        | Throughout — alt. explanations |
| [Speech and Language Processing (Jurafsky)](https://web.stanford.edu/~jurafsky/slp3/)                                             | Weeks 16–20                    |
| [Hands-On Machine Learning (Géron) — 3rd ed.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)      | Phase 2 (classical ML)         |
| [Designing Machine Learning Systems (Chip Huyen)](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) | Phase 7                        |
| [AI Engineering (Chip Huyen, 2024)](https://www.oreilly.com/library/view/ai-engineering/9781098166298/)                           | Phase 7                        |


### Blogs / Papers (full reading list in `LEARNING_SYSTEM.md` §7 "Paper of the Week")

### GitHub Repos to Study


| Repo                                                                    | When          |
| ----------------------------------------------------------------------- | ------------- |
| [karpathy/micrograd](https://github.com/karpathy/micrograd)             | Week 4        |
| [karpathy/makemore](https://github.com/karpathy/makemore)               | Weeks 11, 16  |
| [karpathy/minGPT](https://github.com/karpathy/minGPT)                   | Weeks 21–23   |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)                 | Weeks 24–29   |
| [huggingface/transformers](https://github.com/huggingface/transformers) | Phase 7       |
| [huggingface/peft](https://github.com/huggingface/peft)                 | Week 30       |
| [huggingface/trl](https://github.com/huggingface/trl)                   | Week 34 (DPO) |
| [chroma-core/chroma](https://github.com/chroma-core/chroma)             | Week 31       |
| [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph)     | Week 32       |
| [langfuse/langfuse](https://github.com/langfuse/langfuse)               | Week 33       |
| [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)           | Week 34       |


---

## PROGRESS TRACKER

```
Phase 0: Setup & Learning System        [░░░░░░░░░░] Week  0
Phase 1: Math Engine                     [░░░░░░░░░░] Weeks 1–6
Phase 2: Classical ML                    [░░░░░░░░░░] Weeks 7–9
Phase 3: Neural Networks                 [░░░░░░░░░░] Weeks 10–15
Phase 4: NLP & Sequences                 [░░░░░░░░░░] Weeks 16–20
Phase 5: The Transformer                 [░░░░░░░░░░] Weeks 21–25
Phase 6: Build Your mini-GPT             [░░░░░░░░░░] Weeks 26–29
Phase 7: AI Engineer Toolkit             [░░░░░░░░░░] Weeks 30–34
Phase 8: Capstone & Public Launch        [░░░░░░░░░░] Weeks 35–36

CAPSTONE 1: mini-GPT trained locally     [░░░░░░░░░░]
CAPSTONE 2: CodeSage deployed + launched [░░░░░░░░░░]
```

Fill in the bars as you complete each week: `░` → `█`. This block should also live in your repo `README.md` and be updated every Sunday night.

---

## FINAL ADVICE

1. **Code > reading.** If you have 30 minutes, write code, not notes.
2. **Don't skip Karpathy.** Every other resource supports him; none replaces him.
3. **Build in public from day 1.** Your Friday blog posts, not your certificates, will get you interviews.
4. **The "Valley of Despair" is around Week 12–18.** It's normal. Your metric in that valley is *consistency*, not *breakthroughs*.
5. **Your Salesforce background is an advantage, not a handicap.** You already know SDLC, integration patterns, API design, scale. Most AI researchers can't ship. You can. Lean into it — the `CodeSage` capstone is literally designed around that moat.
6. **Two models you will never be:** the "tutorial watcher" who never writes code, and the "ML Twitter influencer" who never understands internals. You will be the third thing: the engineer who built GPT from scratch, deployed it, and can talk to both scientists and product managers.

---

> *"The best time to plant a tree was 20 years ago. The second best time is now."*
>
> Let's build.

