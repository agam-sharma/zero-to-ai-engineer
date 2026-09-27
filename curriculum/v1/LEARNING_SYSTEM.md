# The Learning System

> This is the *how* of the 9-month masterclass. The `01_ZERO_TO_GPT_6_Month_Masterclass.md` tells you *what* to learn each week; this document tells you how to learn it, how to track it, and how to know when you've actually learned it.

---

## 1. Why This Document Exists

A senior engineer doesn't fail at AI because the math is too hard. They fail because:

1. **They consume without producing.** 20 hours of YouTube, zero hours of working code.
2. **They learn everything once and nothing deeply.** No spaced repetition → forgetting curve wins.
3. **They never externalize understanding.** No blog post, no diagram, no "teach-it-back." So there's no test of whether they know it.
4. **They can't track their own progress**, so motivation dies around week 10.

Every ritual in this document is designed to attack one of those four failure modes.

---

## 2. The Three-Pass Daily Method

This is the single most important habit in this program. **It is how you convert hours into learning.**

```
PASS 1 — WHY (Big Picture)          ~30 min, morning
  • Watch video / read blog / read paper abstract
  • No code, no notes yet, no pausing every 10 seconds
  • At the end: write 3–5 bullets in your journal titled "what is this for?"
  • Write ONE open question you couldn't answer

PASS 2 — HOW (Minute Details)       ~90–120 min, main session
  • Code along with the tutorial at your own pace
  • Pause liberally. Derive one key equation by hand.
  • Break the code on purpose once (wrong shape, wrong loss, wrong init) and
    observe the failure. This is how you build debugging intuition.
  • End with a git commit (even WIP) and update TODO for tomorrow.

PASS 3 — WHY-IT-WORKED (Consolidation) ~20–30 min, evening
  • Close every tab
  • Open your journal and explain today's topic in 5 sentences to your
    past (pre-AI) self. Use NO jargon without defining it.
  • If you can't do it in 5 sentences → you didn't learn it. Mark it "revisit"
    and plan a second Pass 2 tomorrow.
```

**Why three passes work:** Pass 1 activates the schema. Pass 2 fills in details (cognitively expensive, done in your peak focus window). Pass 3 consolidates into long-term memory via explanation-to-self (Feynman technique + testing effect).

**Total weekday time: ~3 hours.** That's the medium-pace target.

---

## 3. Weekly Rituals

| Day | Ritual | Time | Deliverable |
|-----|--------|------|-------------|
| **Mon** | Paper-of-the-week opened; skim abstract + figures during Pass 1 | (within daily 3 hrs) | `papers/week-XX/notes.md` started |
| **Tue–Thu** | Regular Three-Pass daily | 3 hrs/day | Daily journal entries + commits |
| **Fri** | **Blog post** (300+ words) — "What I learned this week" | 45 min at end of day | Posted to dev.to or GitHub Pages; tweet/LinkedIn optional |
| **Sat** | Mini-project AM (see project ladder) + Anki review | ~6 hrs | Project pushed + Anki reviewed |
| **Sun AM** | Read paper-of-the-week fully + 1-page summary | 2 hrs | `papers/week-XX/summary.md` |
| **Sun PM** | **Weekly retrospective** | 1 hr | `journal/retros/2026-WXX.md` (template below) + progress-bar update |
| **Sun rest-of-day** | REST | — | Non-negotiable for burnout prevention |

### Weekly Retrospective Template (`journal/retros/TEMPLATE.md`)

```markdown
# Week XX — YYYY-MM-DD

## Phase progress
- Current phase: Phase N — "..."
- Weeks into phase: W of T
- Overall: XX / 36 (XX%)

## What I actually understood this week (top 3)
1.
2.
3.

## What I struggled with (top 3)
1.
2.
3.

## What I shipped (commits, projects, blog posts)
- [ ] Daily commits: X/5 weekdays
- [ ] Mini-project: ...
- [ ] Blog post: <link>
- [ ] Paper summary: <link>

## Assessment check
- Next pre-phase diagnostic attempted? y/n, score: ___/___
- Current post-phase exam attempted? y/n, score: ___/___

## Energy / pace
- Felt overstretched / on pace / could do more?
- Any days skipped? Why? How to recover?

## Next week's top 3 priorities
1.
2.
3.

## Open questions (carry over next week)
-
```

### Friday Blog Post Template

Keep it short. 300 words is the minimum bar; 600 is the sweet spot.

```markdown
# Week XX: <one-line takeaway>

## What I learned
(2–3 paragraphs — the "aha" moment)

## The concept I finally got
(1 paragraph with a code snippet or diagram)

## What I built
(1 paragraph with a GitHub link)

## What's still fuzzy
(1–2 sentences — honest, not performative)

## Next week
(1 sentence)
```

Even if nobody reads it, **writing forces you to test your understanding**. The first time you sit down to write and realize you can't explain something — that's when actual learning starts.

---

## 4. Spaced Repetition with Anki

Install [Anki](https://apps.ankiweb.net/). Create one deck per phase: `P0-setup`, `P1-math`, `P2-classicalML`, `P3-nn`, `P4-nlp`, `P5-transformer`, `P6-gpt`, `P7-toolkit`, `P8-capstone`.

**Rules for cards:**
- **One concept per card.** If your answer has "and", it's two cards.
- **Cloze deletions for equations** are excellent: `softmax(x_i) = exp(x_i) / {{c1::sum_j exp(x_j)}}`.
- **Code-behavior cards:** "Given a `(B, T, C)` tensor and `torch.nn.Linear(C, D)`, what is the output shape?" → `(B, T, D)`.
- **Diagrams via images** (use your own sketches — phone photo is fine).
- **No "soup" cards** like "Explain attention." Break it: one card for Q/K/V, one for the scaling factor, one for the mask, one for the softmax, one for the final `@V`.

**Reviewing:**
- Saturday is dedicated Anki day. 20–30 minutes.
- Do NOT do new cards every day — the FSRS algorithm handles scheduling. Trust it.
- If a card is wrong 3+ times → rewrite it; it's a bad card, not a bad you.

**Target deck size (rough):**
- Phase 1: ~60 cards
- Phase 2: ~40 cards
- Phase 3: ~80 cards
- Phase 4: ~70 cards
- Phase 5: ~100 cards (Transformer is dense)
- Phase 6: ~40 cards
- Phase 7: ~80 cards
- Phase 8: ~20 cards

Total ~500 cards over 9 months. That's the shape of mastery.

---

## 5. The Git Repository: Your Public Progress Tracker

Create a **public** repo called `zero-to-ai-engineer` (or pick your own name — but keep it public for accountability).

### Folder structure

```
zero-to-ai-engineer/
├── README.md                    # Progress bars, current week, portfolio links
├── LEARNING_SYSTEM.md           # Symlink or copy of this file
├── journal/
│   ├── 2026-W00.md              # One file per week
│   ├── 2026-W01.md
│   ├── ...
│   └── retros/
│       ├── TEMPLATE.md
│       └── 2026-W01.md
├── notebooks/
│   ├── phase_1_math/
│   │   ├── 01_linalg_vectors.ipynb
│   │   └── ...
│   ├── phase_2_classical_ml/
│   └── ...
├── src/                         # Production-style, importable modules
│   ├── mymath/                  # Your NumPy library (Phase 1)
│   ├── micrograd/               # Your autograd (Phase 1)
│   ├── classical/               # sklearn-style wrappers (Phase 2)
│   ├── mytokenizer/             # BPE (Phase 4)
│   ├── mygpt/                   # Your Transformer (Phases 5–6)
│   ├── rag/                     # RAG pipeline (Phase 7)
│   └── agent/                   # Agent loop (Phase 7)
├── projects/
│   ├── 01-numpy-linalg-lib/
│   ├── 02-gradient-descent-viz/
│   ├── 03-entropy-kl-temperature/
│   ├── 04-titanic-kaggle/
│   ├── 05-spam-classifier/
│   ├── 06-mnist-mlp-vs-cnn/
│   ├── 07-loss-pathology-zoo/
│   ├── 08-bpe-vs-tiktoken/
│   ├── 09-shakespeare-ladder/
│   ├── 10-annotated-attention-paper/
│   ├── 11-attention-dashboard/
│   ├── 12-minigpt-tinystories/
│   ├── 13-scaling-laws-calc/
│   ├── 14-phi3-lora-finetune/
│   ├── 15-sf-docs-rag/
│   ├── 16-soql-agent/
│   ├── 17-eval-harness/
│   └── 18-codesage-capstone/
├── papers/
│   ├── TEMPLATE.md
│   ├── week-01-software-2.0/
│   │   ├── notes.md
│   │   └── summary.md
│   └── ...
├── assessments/                 # Copies of pre/post notebooks with your results
├── anki/                        # Exported .apkg files per phase
├── blog/                        # Source markdown of your Friday posts
└── .gitignore
```

### Commit message convention

`<phase>/<week>.<day>: <what you did>` — examples:

```
phase-1/wk3.2: implement chain rule derivation on paper + numerical verify
phase-4/wk16.5: BPE tokenizer passes 12 unit tests
phase-7/wk31.3: add reranker to RAG pipeline, retrieval@5 0.72 -> 0.86
```

### Tags for milestones

After each phase exam passes:

```bash
git tag -a phase-1-complete -m "Phase 1 exam: 18/20"
git push --tags
```

### `README.md` progress tracker (front-page of your repo)

```markdown
# Zero to AI Engineer

My 9-month journey from Senior Salesforce Engineer to AI Engineer.

![Phase](https://img.shields.io/badge/current-phase_1_week_3-blue)
![Progress](https://img.shields.io/badge/overall-8%25-orange)

## Progress

| Phase | Weeks | Status |
|-------|-------|--------|
| 0. Setup & Learning System | Week 0 | ✅ Done |
| 1. Math Engine | Weeks 1–6 | 🔨 In progress (W3) |
| 2. Classical ML | Weeks 7–9 | ⏳ Upcoming |
| ...

## Latest blog post
- [Week 3: the chain rule clicked, finally](blog/2026-W03.md)

## Capstones
- [ ] mini-GPT (Phase 6)
- [ ] CodeSage (Phase 8)
```

Update this every Sunday evening.

---

## 6. How You Know You Actually Learned Something (Four-Gate Test)

A concept is "learned" when **all four** gates pass. Not three. Not "most of them." All four.

```
Gate 1 — DRAW IT
  Can you sketch the shapes/tensors/arrows on paper without reference?

Gate 2 — DERIVE IT
  Can you write the key equation from scratch, including at least one gradient?

Gate 3 — CODE IT FROM SCRATCH
  Can you implement it in NumPy (no torch) and get the right answer
  on a toy input?

Gate 4 — PRODUCTION IT
  Can you re-do it in PyTorch with proper shapes/batching, assert numerical
  equivalence with your NumPy version, and run it on MPS?
```

Most tutorials skip Gates 1 and 2. That's why people finish courses and can't debug anything real.

**Rule:** Do NOT move to the next topic until all four gates pass for the previous one. If a topic takes two weeks instead of one, take two. This is a marathon.

---

## 7. Paper-of-the-Week List

One paper per week, picked to align with that week's content. Sunday AM read-through, 1-page summary in `papers/week-XX/summary.md`.

| Week | Paper / Post | Why |
|------|-------------|-----|
| 0 | Karpathy, *Software 2.0* (blog) | Mindset reset |
| 1 | Olah, *Visualizing Representations* | Geometric intuition for vectors |
| 2 | *A Tutorial on Principal Component Analysis* (Shlens) | PCA from first principles |
| 3 | 3Blue1Brown: *Essence of Calculus* (series) | Calc intuition |
| 4 | Karpathy, *Yes you should understand backprop* | Autograd mindset |
| 5 | *An Intuitive Guide to Maximum Likelihood Estimation* | Probability framing |
| 6 | Cover & Thomas, *Elements of Information Theory* Ch. 2 (entropy) | The bedrock |
| 7 | Breiman, *Statistical Modeling: The Two Cultures* | Why ML ≠ stats |
| 8 | Friedman, *Greedy Function Approximation: A Gradient Boosting Machine* | XGBoost lineage |
| 9 | Geirhos et al., *Shortcut Learning in Deep Neural Networks* | Debugging intuition |
| 10 | Rumelhart, Hinton, Williams, *Learning representations by back-propagating errors* (1986) | Historical context |
| 11 | Karpathy, *A Recipe for Training Neural Networks* | Phase-3 bible |
| 12 | Ioffe & Szegedy, *Batch Normalization* | Why normalization matters |
| 13 | Krizhevsky et al., *ImageNet Classification with Deep CNNs (AlexNet)* | The ImageNet moment |
| 14 | He et al., *Deep Residual Learning (ResNet)* | Residual connections |
| 15 | Smith, *Cyclical Learning Rates for Training NNs* | LR finder origin |
| 16 | Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units (BPE)* | Tokenization |
| 17 | Mikolov et al., *Efficient Estimation of Word Representations (word2vec)* | Embeddings |
| 18 | Hochreiter & Schmidhuber, *Long Short-Term Memory* (1997) | LSTM |
| 19 | Bahdanau et al., *NMT by Jointly Learning to Align and Translate* | Attention origin |
| 20 | Karpathy, *The Unreasonable Effectiveness of RNNs* | Era-ending blog |
| 21 | **Vaswani et al., *Attention Is All You Need*** | THE paper |
| 22 | Alammar, *The Illustrated Transformer* | Visual guide |
| 23 | Ba, Kiros, Hinton, *Layer Normalization* | Why LN in Transformers |
| 24 | Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2) | The pretraining thesis |
| 25 | Elhage et al., *A Mathematical Framework for Transformer Circuits* | Mechanistic interpretability |
| 26 | Kaplan et al., *Scaling Laws for Neural Language Models* | Compute-optimal thinking |
| 27 | Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla) | Revised scaling laws |
| 28 | Holtzman et al., *The Curious Case of Neural Text Degeneration* | Nucleus sampling |
| 29 | Su et al., *RoFormer: Rotary Position Embedding* (RoPE) | Modern positions |
| 30 | Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* | PEFT |
| 31 | Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive Tasks* | RAG origin |
| 32 | Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* | Agents |
| 33 | Zheng et al., *Judging LLM-as-a-Judge with MT-Bench* | Evals |
| 34 | Rafailov et al., *Direct Preference Optimization (DPO)* | Alignment |
| 35 | Shumailov et al., *The Curse of Recursion (Model Collapse)* | Risks |
| 36 | Your own blog post series as the capstone recap | Reflection |

---

## 8. When to Skip, Refresh, or Deep-Dive (Decision Tree)

Before each phase, run the **pre-phase diagnostic notebook**. Based on your score:

```
                   PRE-PHASE DIAGNOSTIC
                          │
          ┌───────────────┼────────────────┐
          │               │                │
        90%+            60–89%           <60%
          │               │                │
       SKIP AHEAD      ON PACE         DEEP-DIVE
       (halve this    (run normal     (add a 1-week
        phase's        schedule)       refresher before
        schedule)                      starting phase)
          │               │                │
          └── Still do POST-PHASE exam ──┘
              (this is the real gate)
```

**Critical rule:** the post-phase exam is non-negotiable regardless of how you did on the pre-phase. Skipping forward without passing post-phase creates hidden debt that compounds.

---

## 9. The "Teach It Back" Loom Ritual

End of every phase, record a **5-minute Loom** (or any screen recorder) explaining *one* concept from the phase with no notes. Post link in your journal retrospective.

Why this works:
- Forces testing effect at the phase boundary
- Creates an asset you can put on your portfolio / share on LinkedIn
- Reveals gaps that silent study never does

Topics:
- Phase 1: "What is a gradient, and why is it a *vector*?"
- Phase 2: "Bias-variance tradeoff in 5 minutes"
- Phase 3: "What BatchNorm actually does, with numbers"
- Phase 4: "Why RNNs couldn't scale"
- Phase 5: "Self-attention in 5 minutes, no equations"
- Phase 6: "Sampling temperature explained"
- Phase 7: "When to RAG vs fine-tune vs prompt"
- Phase 8: (your `CodeSage` demo — this one is 20 minutes, not 5)

---

## 10. Portfolio & Career Transition (Runs in Parallel from Phase 3 Onward)

You want a job, not a certificate. Do these *in parallel* with the technical work:

| Starting Phase | Action | Cadence |
|----------------|--------|---------|
| Phase 0 | Public GitHub repo live | once |
| Phase 2 | First 2 blog posts published | weekly from here on |
| Phase 3 | LinkedIn headline updated: "Software Engineer (Salesforce) learning to build AI" | once |
| Phase 4 | Follow/engage with 20 AI practitioners on Twitter/LinkedIn | ongoing |
| Phase 5 | Start attending (virtually) one AI meetup/month | monthly |
| Phase 6 | First conference talk pitch (local meetup, 10 min): "Building GPT on my laptop" | once |
| Phase 7 | Start applying to AI Engineer roles selectively (5/week, high-quality only) | weekly |
| Phase 7 | Rewrite resume around projects, not past job titles | once |
| Phase 8 | Launch `CodeSage` publicly — LinkedIn post, demo video, Show HN | once |
| Phase 8 | Mock interviews (10 total: 5 technical, 5 system-design) | final 2 weeks |

---

## 11. Red Flags & How to Recover

| Red flag | What it means | Fix |
|----------|---------------|-----|
| 🚨 Watching videos at 2× without coding | You're consuming, not learning | Rule: no video without code editor open |
| 🚨 "I'll commit when it's clean" | You'll never commit | Commit WIP with `wip:` prefix |
| 🚨 Skipping Friday blog post > 2 weeks in a row | Learning has stalled | Take a day off. Then revisit Phase 1's post to see how far you've come |
| 🚨 Not using Anki for > 2 weeks | You're forgetting Phase 1 while doing Phase 3 | Batch 1 hour of Anki catch-up on Saturday |
| 🚨 Jumping ahead in the phases | Hidden debt → will bite you in Phase 5 | Go back. Always. |
| 🚨 Feeling like an impostor around Week 10–18 | This is the "Valley of Despair" — it's expected | Re-read your Week 1 journal entry. Evidence > feelings. |
| 🚨 Hardware excuse ("I need an H100") | You don't. GPT-2-small fits on your M3 Pro | Drop it |
| 🚨 Framework FOMO ("should I be learning Mojo/JAX?") | Premature optimization of your career | Ignore until Phase 7 |

---

## 12. Bigger Picture / Minute Details Balance

The user asked for **both** dimensions. How this plan delivers both:

**Bigger picture is delivered by:**
- Phase overviews ("Why this phase")
- Paper-of-the-week (historical context + landmark ideas)
- The `CodeSage` capstone design (integrates everything)
- The cross-phase mini-projects (e.g., Shakespeare ladder in Phase 4 shows architecture evolution)
- Friday blog posts (forces top-down synthesis)

**Minute details are delivered by:**
- Four-Gate Test (draw → derive → NumPy → PyTorch)
- Manual backprop (Phase 3 Week 11)
- Implement-then-compare-to-PyTorch pattern throughout
- Anki for precise recall
- Post-phase exam notebooks with hidden `pytest` assertions

**The integration point:** every Sunday retrospective asks "what's the bigger-picture question this week's details answered?" and every Monday's paper-of-the-week supplies the next bigger-picture frame.

Neither dimension alone is enough. Together they are.

---

## 13. One-Page Printable Cheat Sheet

Print this. Pin it to your monitor.

```
┌────────────────────────────────────────────────────────────┐
│   DAILY (weekday)                                         │
│   □ Pass 1: WHY  (30 min, morning, no code)               │
│   □ Pass 2: HOW  (90–120 min, evening, code along)        │
│   □ Pass 3: WHY-IT-WORKED (20 min, bedtime journal)       │
│   □ Git commit                                            │
│                                                           │
│   FRIDAY                                                  │
│   □ 300-word blog post                                    │
│                                                           │
│   SATURDAY                                                │
│   □ Mini-project AM                                       │
│   □ Anki review (20–30 min)                               │
│                                                           │
│   SUNDAY                                                  │
│   □ Paper-of-the-week (1 hr + 1-page summary)             │
│   □ Weekly retrospective                                  │
│   □ Update README progress bars                           │
│   □ REST from 4pm onward                                  │
│                                                           │
│   END OF PHASE                                            │
│   □ Pre-phase diagnostic (before starting the next)       │
│   □ Post-phase exam (before tagging phase-X-complete)     │
│   □ 5-minute Loom "teach-it-back"                         │
│   □ Tag release in git                                    │
└────────────────────────────────────────────────────────────┘
```

---

> You don't rise to the level of your goals. You fall to the level of your systems. — James Clear
>
> This is the system. Trust the system, and in 36 weeks you'll be where you want to be.
