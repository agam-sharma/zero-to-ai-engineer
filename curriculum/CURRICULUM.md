# ZERO TO AI ENGINEER — Rebuilt Curriculum
### Agam Sharma · Senior Salesforce/Agentforce Engineer → AI Engineer
### Version 2.0 · September 2026

---

## Your parameters

| | |
|---|---|
| **Sustainable pace** | 15–20 hrs/week (~17 average) |
| **Deadline** | None. Built properly, not fast. |
| **Math starting point** | Rebuilt from clean |
| **Hardware** | Mac M3 Pro, 18 GB unified memory, MPS |
| **Existing advantage** | 8 yrs engineering · Agentforce/Data Cloud in production · AI Specialist certified |
| **Capstone domain** | Salesforce — deliberately, as a moat |
| **Total** | ~1,100 hours · ~64 weeks · **~15 months** |

That 15-month number is honest. Your previous plan said 9 months at 25–30 hrs/week. At the pace you can actually sustain, the same material plus the gaps it was missing takes about 15 months. A 15-month plan you finish beats a 9-month plan you abandon in month four.

---

## The one insight that shapes this program

**Your day job already teaches you the applied layer.**

You build Agentforce agents, Data Cloud retrieval, prompt frameworks and grounding in production, right now, for real users. That is Stage 8 material. Most people doing this transition have to *learn* RAG and agents; you ship them.

So this program does not front-load what you already do. It goes after what your job cannot teach you: **what is actually happening inside the model.** Foundations-first isn't the slow path for you — it's the only path that adds anything.

The corollary: Stage 8 will feel easy. That's correct. Use the time there to convert tacit knowledge into explainable knowledge, because "I've built RAG systems" and "I can explain why my reranker improved retrieval@5" are different sentences in an interview.

---

## What was wrong with v1 (and what I kept)

**Kept — this was good work:**
- The Three-Pass daily method (WHY → HOW → WHY-IT-WORKED)
- The Four-Gate Test — the best idea in the original document
- Phase sequencing, including the refusal to skip probability/information theory and classical ML
- Paper-of-the-week, Friday blog, Anki, public repo, build-in-public
- Salesforce analogies as scaffolding
- Pre/post assessments per stage
- The entire `reference_code/` tree — 100+ working files, still valid

**Fixed:**
- *Phase 5 and 6 had collided.* The Phase 5 document taught GPT training, which is Phase 6's job. Now strictly separated: Stage 5 is the transformer's parts, Stage 6 is training one.
- *Week numbering drift* across several documents.
- *Phase 3 was under-built* — about half its promised content wasn't written.
- *Math assumed a starting point you don't have.* Rewritten from clean, code-first.

**Added — Gate 5, the Re-Implementation Drill, and nine missing topics:**

| Added | Why it matters |
|---|---|
| Reasoning models & test-time compute | The largest shift in the field since the transformer. v1 had nothing. |
| GQA / MQA | What makes modern inference affordable. v1 had RoPE but not this. |
| Mixture of Experts | Standard in frontier models now. |
| Inference optimization beyond KV-cache | Speculative decoding, continuous batching, paged attention. |
| Multimodality | CLIP, vision encoders, VLMs. Zero coverage in v1. |
| Distributed training concepts | You can't run it on an M3; you will be asked about it. |
| **MCP** | Now the universal standard for agent tooling — and a direct bridge from your Agentforce work. |
| Numerical stability | log-sum-exp, bf16 vs fp16, catastrophic cancellation. Rarely taught, causes real bugs. |
| SVD, properly | It's the low-rank decomposition LoRA is named after. v1 mentioned it once. |

---

## The Five Gates

A concept is learned when all five pass.

```
1. DRAW IT        — sketch shapes/tensors/arrows from memory
2. DERIVE IT      — write the key equation, including one gradient
3. CODE IT        — NumPy only, no autograd, correct on a toy input
4. PRODUCTION IT  — PyTorch, batched, asserts equivalence with (3), runs on MPS
5. DEFEND IT      — explain it to Claude, survive three follow-ups
```

Gates 1–4 are from your original system. Gate 5 is what a mentor adds over a document.

---

## Weekly rhythm at 15–20 hrs

| | |
|---|---|
| **Mon–Fri** | ~2 hrs/day. Three-Pass method. Commit daily. |
| **Sat** | ~4 hrs. Project work + Anki review. |
| **Sun AM** | ~2 hrs. Paper-of-the-week + retrospective + Claude session. |
| **Sun PM** | Rest. Non-negotiable. |

### The Minimum Viable Week

Bad weeks happen — a release, illness, family. On those weeks the target is **3 hours, not zero**:

- One Pass-2 session on one concept (90 min)
- Anki review (30 min)
- One commit, however small
- Tell Claude it's an MVW so it's logged and not treated as a stall

The streak matters more than the volume. Programs die at zero, not at three.

---

## THE ELEVEN STAGES

| # | Stage | Hours | ~Weeks | Ends with |
|---|---|---|---|---|
| 0 | Setup, Baseline & the Contract | 20 | 1 | Environment verified, honest baseline known |
| 1 | **Math You Can Run** | 190 | 11 | Your own autograd engine |
| 2 | Classical ML | 70 | 4 | Two Kaggle-grade models with clean hygiene |
| 3 | Neural Networks | 140 | 8 | CIFAR-10 at ≥85%, backprop by hand |
| 4 | Language & Sequences | 110 | 7 | BPE tokenizer + the Shakespeare ladder |
| 5 | The Transformer | 120 | 7 | Attention matching PyTorch to 1e-6 |
| 6 | Build Your GPT | 110 | 7 | A pretrained mini-GPT you made |
| 7 | Post-Training & Alignment | 90 | 5 | LoRA + DPO + a reasoning experiment |
| 8 | AI Engineering in Production | 110 | 6 | Deployed, evaluated, observable system |
| 9 | Breadth: Vision, Diffusion, Alternatives | 60 | 4 | You know what else exists and why |
| 10 | Capstone & Launch | 80 | 4 | **CodeSage**, public |
| | **Total** | **1,100** | **~64** | |

**Exit ramp:** after Stage 6 (~month 9) you have built a GPT from scratch and can explain every line. That alone is a portfolio most applicants don't have. If life intervenes, stopping there is a real and defensible outcome.

---

## STAGE 0 — Setup, Baseline & the Contract
**~20 hrs · 1 week**

Your original Phase 0 was solid. Keep it, plus:

| Module | What |
|---|---|
| 0.1 | Environment: miniconda `ai-mastery`, Python 3.11, PyTorch + MPS verified |
| 0.2 | Public repo `zero-to-ai-engineer`, folder structure, first commit |
| 0.3 | Anki installed, deck-per-stage created, card-writing conventions read |
| 0.4 | Journal + paper-of-the-week workflow; read Karpathy *Software 2.0* |
| 0.5 | **Baseline diagnostic** — run `00_Foundation_Assessment_Exercises.ipynb`, score honestly, log it |
| 0.6 | **The contract** — a session with Claude agreeing the pace, the gates, and what happens on a bad week |

**Gate:** `torch.backends.mps.is_available()` is `True`; repo is public; diagnostic score logged in `PROGRESS.md`.

---

## STAGE 1 — Math You Can Run
**~190 hrs · 11 weeks · the foundation everything rests on**

> This stage is completely rewritten. Your v1 Phase 1 assumed you could start at dot products and reach autograd in four weeks. With math rebuilt from clean, that pace guarantees a collapse in Stage 3.

**The method:** every concept arrives as *code you write and output you can see*, then as notation. Never notation first. You have eight years of programming intuition — we use it as the on-ramp to math, not the other way round.

| Module | Topic | Hrs | Headline build |
|---|---|---|---|
| 1.1 | Functions, graphs, change | 18 | Plot everything; numerical derivative before symbolic |
| 1.2 | Vectors | 20 | `dot`, `norm`, `cosine_similarity`; king − man + woman ≈ queen |
| 1.3 | Matrices as transformations | 25 | `matmul` by hand → NumPy; shape discipline drilled hard |
| 1.4 | Eigenvectors, **SVD**, PCA | 22 | SVD from scratch; 2-D PCA of MNIST showing digit clusters |
| 1.5 | Derivatives & the chain rule | 25 | Partial derivatives; gradients as vectors; **matrix calculus layout conventions** |
| 1.6 | Gradient descent | 16 | 2-D loss-landscape visualizer; momentum; Adam derived, not quoted |
| 1.7 | **micrograd** | 22 | Karpathy's autograd engine, built by you, matching PyTorch |
| 1.8 | Probability for AI | 18 | Bernoulli/Categorical/Gaussian; MLE from data |
| 1.9 | Information theory | 16 | `entropy`, `cross_entropy`, `kl_divergence`, softmax-with-temperature; prove CE = NLL |
| 1.10 | **Numerical stability** | 8 | log-sum-exp; fp32/fp16/bf16; catastrophic cancellation |

**Why SVD gets its own weight (1.4):** LoRA — which you'll use in Stage 7 and which is the single most employable fine-tuning skill — is *low-rank adaptation*. The low-rank idea is SVD. Learn it properly here and Stage 7 costs you a day instead of a week.

**Why matrix calculus conventions (1.5):** backprop derivations feel confusing mostly because nobody mentions there are two layout conventions (numerator and denominator) and textbooks silently switch. Naming this removes a month of low-grade confusion.

**Why numerical stability (1.10):** your softmax will overflow at logits above ~710. Your loss will go NaN. Almost no curriculum teaches this, and it's a favourite interview probe because it separates people who've trained things from people who've read about training things.

**Projects:** P1 NumPy linear-algebra library with tests · P2 gradient-descent visualizer · P3 micrograd · P4 entropy/KL/temperature sampler · P5 numerical-stability lab

**Stage gate:** you can derive the chain rule on paper for a 3-layer MLP, explain cross-entropy as negative log-likelihood under a categorical distribution, and your micrograd's gradients match PyTorch's to 1e-6.

---

## STAGE 2 — Classical ML
**~70 hrs · 4 weeks**

Kept largely from v1 — it was right. Every neural-network debugging technique was invented here first.

| Module | Topic | Hrs |
|---|---|---|
| 2.1 | Linear & logistic regression from scratch; L1/L2 | 20 |
| 2.2 | Trees, boosting, metrics, cross-validation | 22 |
| 2.3 | Bias–variance, leakage, data hygiene | 14 |
| 2.4 | Error analysis as a discipline | 14 |

**Projects:** P6 Titanic with proper CV · P7 spam classifier, logistic vs MLP side-by-side

**Stage gate:** your logistic regression matches sklearn's to within 1%; you can pick and justify the right metric for a given problem; you can design a split that avoids leakage, including time-series.

---

## STAGE 3 — Neural Networks
**~140 hrs · 8 weeks**

Rebuilt — v1 promised six weeks of content and wrote about half of it.

| Module | Topic | Hrs |
|---|---|---|
| 3.1 | Perceptron, activations, loss functions | 16 |
| 3.2 | MLP + the training loop in detail | 20 |
| 3.3 | **Backprop by hand** — Karpathy makemore Part 4 | 24 |
| 3.4 | Normalization: BatchNorm, LayerNorm, RMSNorm | 18 |
| 3.5 | Regularization: dropout, weight decay, early stopping | 12 |
| 3.6 | Optimizers in practice: SGD → momentum → Adam → AdamW | 12 |
| 3.7 | CNNs | 16 |
| 3.8 | **Training diagnostics** — the loss-pathology zoo | 12 |
| 3.9 | Hyperparameter tuning, LR finder, W&B sweeps | 10 |

**Module 3.3 is the most important module in the program.** Manually backpropagating through an MLP with no `.backward()` is the thing that converts "I use PyTorch" into "I know what PyTorch does." Budget three weeks if it takes three weeks.

**Module 3.8** builds deliberate failures — dead ReLU, LR too high, no normalization, vanishing gradients, overfitting, data leakage — and trains you to name each from the loss curve alone. This is the skill that makes you useful on day one of an AI job.

**Projects:** P8 MNIST MLP vs CNN benchmarked MPS vs CPU · P9 loss-pathology zoo · P10 backprop ninja · P11 CIFAR-10 ≥85%

**Stage gate:** manual gradients match autograd to 1e-6; you diagnose six failure modes from the curve alone.

---

## STAGE 4 — Language & Sequences
**~110 hrs · 7 weeks**

| Module | Topic | Hrs |
|---|---|---|
| 4.1 | Tokenization & BPE from scratch | 20 |
| 4.2 | Word embeddings; word2vec skip-gram | 18 |
| 4.3 | Dense retrieval primer — the foundation under your day job | 12 |
| 4.4 | Vanilla RNN | 16 |
| 4.5 | Vanishing gradients; LSTM and GRU | 22 |
| 4.6 | Seq2seq + Bahdanau attention — the origin story | 22 |

**Module 4.3 note:** you already build retrieval in production. Here you build it from scratch so you understand what your Data Cloud retrievers are doing underneath.

**Projects:** P12 BPE vs `tiktoken` · P13 the Shakespeare ladder (bigram → MLP → RNN → LSTM → attention, same data, same budget) · P14 toy semantic search

**Stage gate:** you can explain precisely why attention beats recurrence for long-range dependencies, having felt it on the ladder rather than read it.

---

## STAGE 5 — The Transformer
**~120 hrs · 7 weeks · no GPT training here — that's Stage 6**

| Module | Topic | Hrs |
|---|---|---|
| 5.1 | Self-attention from scratch | 20 |
| 5.2 | Multi-head attention; causal masking | 18 |
| 5.3 | **GQA / MQA** — what modern models actually use | 10 |
| 5.4 | Positional encoding: sinusoidal → **RoPE** | 16 |
| 5.5 | LayerNorm, RMSNorm, residuals, pre-norm vs post-norm | 14 |
| 5.6 | FFN: GELU, **SwiGLU** | 10 |
| 5.7 | The full block; encoder / decoder / decoder-only | 16 |
| 5.8 | *Attention Is All You Need*, annotated line by line | 8 |
| 5.9 | **Mechanistic interpretability primer** | 8 |

**Projects:** P15 attention matching `torch.nn.MultiheadAttention` to 1e-6 · P16 annotated paper with links to your own code · P17 attention-head interpretability dashboard · P18 modern-architecture comparison write-up (GQA vs MHA, MoE, RoPE vs sinusoidal)

**Stage gate:** you can explain every line of nanoGPT and every figure in the paper, and identify at least two specialized attention heads in a trained model.

---

## STAGE 6 — Build Your GPT
**~110 hrs · 7 weeks · Capstone 1**

| Module | Topic | Hrs |
|---|---|---|
| 6.1 | Scaling laws; Chinchilla; sizing for *your* compute budget | 16 |
| 6.2 | Data preparation and tokenized binary formats | 14 |
| 6.3 | **The pretraining run** — 50–100M parameters on your M3 | 26 |
| 6.4 | Sampling: greedy, temperature, top-k, top-p | 14 |
| 6.5 | KV-cache | 12 |
| 6.6 | **Speculative decoding** | 10 |
| 6.7 | Flash Attention / `F.scaled_dot_product_attention` | 8 |
| 6.8 | **Mixture of Experts primer** | 10 |

**Projects:** P19 scaling-laws calculator for the M3 Pro · P20 mini-GPT on TinyStories · P21 KV-cache + speculative-decoding benchmark

**Stage gate:** a trained model that generates coherent text, a val loss competitive with the nanoGPT baseline, and a written justification for why your model is exactly the size it is.

**This is the exit ramp.** Reaching here means you built a language model from first principles. That is not a common credential.

---

## STAGE 7 — Post-Training & Alignment
**~90 hrs · 5 weeks · split out from v1's overloaded Phase 7**

| Module | Topic | Hrs |
|---|---|---|
| 7.1 | SFT and instruction tuning | 14 |
| 7.2 | **LoRA / QLoRA** — where Stage 1.4's SVD pays off | 20 |
| 7.3 | RLHF: reward models, PPO conceptually | 14 |
| 7.4 | **DPO** — and why it displaced PPO for most teams | 12 |
| 7.5 | **Reasoning models & test-time compute** | 20 |
| 7.6 | Safety: red-teaming, jailbreaks, constitutional methods | 10 |

**Module 7.5 is the most important addition to the whole curriculum.** Chain-of-thought as a trained behaviour, process supervision, RL for reasoning, and the compute-at-inference tradeoff — this is the axis the field moved along while v1 was being written, and its absence would have dated you badly.

**Module 7.6 connects to your AI Specialist certification** — you already have to reason about safe agent behaviour in Agentforce. Here you learn the attack side.

**Projects:** P22 LoRA fine-tune with before/after eval · P23 DPO on preference pairs · P24 a reasoning/test-time-compute experiment · P25 red-team your own model

---

## STAGE 8 — AI Engineering in Production
**~110 hrs · 6 weeks · your home turf — go deeper than you already are**

| Module | Topic | Hrs |
|---|---|---|
| 8.1 | RAG properly: chunking, hybrid search, reranking, Ragas | 22 |
| 8.2 | **Context engineering** — what replaced prompt engineering | 12 |
| 8.3 | Agents: ReAct, function calling, structured output | 18 |
| 8.4 | **MCP — build a server, not just consume one** | 16 |
| 8.5 | Evals & observability | 16 |
| 8.6 | Deployment: FastAPI streaming, quantization, caching | 16 |
| 8.7 | **Inference serving**: continuous batching, paged attention, vLLM | 10 |

**Module 8.4 is your highest-leverage single module.** MCP became the universal standard for agent tool integration in 2026. You already understand Salesforce's data model, security model and API surface better than almost anyone entering AI. A well-built MCP server exposing Salesforce to any model is a portfolio piece nobody else applying for these roles can produce.

**Approach this stage differently from the others.** You've shipped most of this. The work here isn't learning it — it's converting what you know tacitly into something you can explain, measure and defend. Expect Claude to push hard on *why* your production choices were right, not just what they were.

**Projects:** P26 production RAG over Salesforce docs · **P27 Salesforce MCP server** · P28 SOQL agent · P29 prompt regression harness · P30 deployed quantized API

---

## STAGE 9 — Breadth
**~60 hrs · 4 weeks**

Not depth — orientation. You should know what else exists, roughly how it works, and when it's the right tool.

| Module | Topic | Hrs |
|---|---|---|
| 9.1 | Vision: CNNs → ViT → CLIP | 20 |
| 9.2 | Multimodal models: how a VLM actually fuses modalities | 14 |
| 9.3 | Diffusion models: forward/reverse process, one tiny implementation | 16 |
| 9.4 | Alternatives: state-space models, Mamba, and why they exist | 10 |

**Projects:** P31 CLIP-style contrastive model from scratch · P32 tiny diffusion model on MNIST

---

## STAGE 10 — Capstone & Launch
**~80 hrs · 4 weeks · Capstone 2**

### CodeSage

Everything you've built, as one hireable artifact:

- A **LoRA fine-tuned model** on an Apex/Salesforce-docs corpus
- A **production RAG system** over Salesforce developer documentation
- An **MCP server** exposing a scratch org — SOQL, metadata, file access
- A **tool-using agent** on top of it
- **Streaming FastAPI backend** + a real UI
- **Observability** and an **evals harness** with a 30-pair golden set

### Launch

- Public deploy, blog post, demo video, architecture diagram
- Portfolio README with metrics, not adjectives
- Ten mock interviews: five fundamentals, five system design, recorded
- A 20-minute talk taking someone from the attention equation to your deployed UI

**Graduation:** CodeSage answers 8/10 curated questions reliably in public, and you can whiteboard the cross-entropy gradient without hesitating.

---

## On the Salesforce capstone — the honest tradeoff

You chose to keep the Salesforce domain as your moat. I think that's right, and here's the reasoning plus the risk.

**Why it's right:** the Salesforce + AI intersection is real, well-paid, and thin on people who understand both sides. Most AI engineers can't navigate an enterprise CRM data model; most Salesforce engineers can't explain attention. You'll be able to do both, and the MCP server project makes that concrete rather than claimed.

**The risk, stated plainly:** a portfolio that is entirely Salesforce can read as "Salesforce person who learned some AI" rather than "AI engineer." That costs you at generalist AI companies.

**The mitigation, which costs you nothing extra:** your Stage 6 mini-GPT and Stage 9 diffusion/CLIP projects are already domain-neutral. Present the portfolio as *two* things — "I built a language model from scratch" and "I build production AI systems in a domain I know deeply" — rather than one. Lead with whichever fits the role. The work is the same; the framing is free.

---

## Resources

**Video — priority order**

| Resource | When |
|---|---|
| [3Blue1Brown — Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) | Stage 1.2–1.4 |
| [3Blue1Brown — Essence of Calculus](https://www.3blue1brown.com/topics/calculus) | Stage 1.1, 1.5 |
| [3Blue1Brown — Neural Networks](https://www.3blue1brown.com/topics/neural-networks) | Stages 1, 3, 5 |
| [Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) (8 lectures) | The spine: 1.7, 3.2–3.4, 4.1, 5.1–5.7 |
| [StatQuest](https://www.youtube.com/c/joshstarmer) | Throughout, especially Stage 2 |
| [HuggingFace LLM Course](https://huggingface.co/learn/llm-course) | Stages 7–8 |

> Note: your v1 plan mapped Karpathy to six lectures. There are **eight** — you were missing *makemore Part 5: WaveNet* and under-using *Part 4: Becoming a Backprop Ninja*, which is the single most valuable one for Stage 3.

**Reading**

| Resource | When |
|---|---|
| [Mathematics for Machine Learning](https://mml-book.github.io/) (free) | Stage 1 reference — not a primary text for you |
| [Math Academy — Mathematics for ML](https://www.mathacademy.com/courses/mathematics-for-machine-learning) (paid) | Optional Stage 1 spine; adaptive and mastery-based, good fit for rebuilding from clean |
| [Dive into Deep Learning](https://d2l.ai/) | Throughout, for alternative explanations |
| [Sebastian Raschka — The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) | Stage 5.3–5.6, the modern-architecture modules |
| [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) (Jurafsky, free) | Stage 4 |
| [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) (Chip Huyen) | Stage 8 |
| [Model Context Protocol docs](https://modelcontextprotocol.io) | Stage 8.4 |

**Code to study:** `karpathy/micrograd` (1.7) · `karpathy/makemore` (3.2–3.4, 4.1) · `karpathy/nanoGPT` (5–6) · `huggingface/peft` (7.2) · `huggingface/trl` (7.4) · `vllm-project/vllm` (8.7)

**Paper-of-the-week:** keep your v1 list — it's well-chosen. Add, in Stage 5–7: *GQA* (Ainslie et al.) · *Switch Transformers* (Fedus et al., MoE) · *RoFormer* (RoPE) · *Chinchilla* · *DPO* (Rafailov et al.) · a current reasoning-models paper, chosen when you reach 7.5 so it's not stale.

---

## Cloud GPU — the thing v1 didn't budget for

Your M3 Pro handles Stages 1–6 entirely, and most of 7. It will not handle: full fine-tunes above ~1B parameters, any real MoE experiment, or Stage 9's diffusion training at reasonable speed.

Budget roughly **$150–400 total across the program** for rented GPU time (Colab Pro, Lambda, RunPod, Modal), concentrated in Stages 7 and 9. Not having a plan for this is how people stall at module 7.2 and conclude they need different hardware. You don't — you need about twenty hours of rented A100.

---

## Tracking

| File | Purpose |
|---|---|
| `PROGRESS.md` | Living state. Claude maintains it. Read at the start of every session. |
| `MENTOR_PROTOCOL.md` | How Claude teaches. Read before `PROGRESS.md`. |
| `Stage_N_*.md` | Detailed guides, written just-in-time — one stage ahead, so they're current |

Stage detail files are deliberately **not** all written now. A Stage 8 guide written in September 2026 would be stale by the time you reach it in late 2027. Stage 0 and Stage 1 exist now; each next one gets written as you approach it.

---

> You are not starting from zero. You are starting from eight years of shipping software and three years of building AI systems in production, with a math gap.
>
> The math gap is the smallest part of that sentence. It's also the only part this program can't skip.
