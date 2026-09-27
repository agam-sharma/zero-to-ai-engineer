# ═══════════════════════════════════════════════════════════════════════
# PHASE 8: CAPSTONE & PUBLIC LAUNCH (Weeks 35–36)
# ═══════════════════════════════════════════════════════════════════════
# Zero to AI Engineer — 9-Month Masterclass
#
# Two weeks. One goal: converge everything you've built into ONE polished,
# public portfolio project, launch it, and prepare for interviews.
# ═══════════════════════════════════════════════════════════════════════

---

## 🎯 Why This Phase Matters More Than Any Other

9 months of hard work is worth nothing if nobody sees it. This phase is where "I learned AI" becomes "I ship AI." It's also where you create the single asset that will get you interviews: a polished, public, technically-deep project with a live demo, a blog post, and a video walkthrough.

Recruiters don't read 500-page GitHub profiles. They watch a 3-minute demo video and click your live URL. That's the whole bar. Your job this phase is to build something worth watching and clicking.

## 🔗 The Salesforce Analogy

> This is your **Dreamforce keynote demo**. Everything under the hood matters — but at the end, someone clicks a button and something impressive happens live on stage.

---

# THE CAPSTONE: `CodeSage`

> An AI pair-programmer specialized in Salesforce development — combining everything from Phases 1–7 into a single user-facing product.

## What `CodeSage` Does (User-Facing)

A web UI where a developer can:
1. Ask a natural-language question about Salesforce development ("how do I write an Apex trigger that fires on Contact update without running into governor limits?")
2. Get a **grounded answer with citations** from official Salesforce docs (via RAG — Phase 7 Week 31)
3. See **code examples** that are stylistically consistent with their codebase (via a LoRA fine-tune on open-source Apex — Phase 7 Week 30)
4. Ask follow-up questions that trigger **tool calls** — query a connected org for metadata or run a test SOQL (Phase 7 Week 32)
5. Every response is **streamed**, **cached**, and **traced** (Phase 7 Weeks 33–34)

## Why This Capstone (and Not Something Else)

- **It's uniquely yours.** Anyone can train a sentiment classifier. Very few people have 7 years of Salesforce engineering context + AI skills. Lean into it.
- **It integrates every phase:** embedding-space retrieval (P4), Transformer (P5), fine-tuning (P7W30), RAG (P7W31), agents (P7W32), evals (P7W33), deployment (P7W34).
- **It has a clear story** for interviews: "I built an AI Engineer for the Salesforce developer ecosystem because I know both sides of the problem."
- **It's interview-bait.** An AI Engineer applying to a company with a Salesforce integration surface area (Snowflake, Databricks, HubSpot, Gong, countless mid-market B2B SaaS) is *immediately* differentiated.

> If Salesforce isn't your thing any more, swap the domain. The architecture is identical whether the corpus is Salesforce docs, Rails source, OpenAPI specs, your company wiki, or a book series. What matters is you pick a domain you **already know deeply** so your error-analysis and evaluation are high-signal.

---

## ARCHITECTURE (draw this from memory by end of Week 35 Day 1)

```
             ┌──────────────────────────┐
             │    Next.js / Gradio UI    │
             └──────────┬───────────────┘
                        │ SSE
             ┌──────────▼───────────────┐
             │    FastAPI (streaming)   │
             │   + Redis (cache)        │
             └──────────┬───────────────┘
                        │
       ┌────────────────┼────────────────────┐
       ▼                ▼                    ▼
┌──────────────┐ ┌─────────────┐   ┌────────────────────┐
│   Chroma    │ │  BM25 index │   │ Fine-tuned Phi-3    │
│  (dense     │ │  (sparse)   │   │  LoRA adapter       │
│   vectors)  │ │             │   │  (served via        │
└──────┬──────┘ └─────┬───────┘   │   llama.cpp Q4)     │
       │              │           └─────────┬──────────┘
       └──── HYBRID ──┘                     │
              │                             │
              ▼                             │
       ┌──────────────┐                    │
       │  Reranker    │                    │
       │  (BGE cross- │                    │
       │   encoder)   │                    │
       └──────┬───────┘                    │
              │ top-5 chunks               │
              ▼                             │
       ┌──────────────────────────────────┐│
       │       Grounded Prompt Builder    ││
       │  (+ tool schemas for agent mode) ││
       └──────────────────┬───────────────┘│
                          │                │
                          ▼                ▼
                    ┌──────────────────────────┐
                    │    LLM Generation        │
                    │    (with tool-call loop) │
                    └──────────┬───────────────┘
                               │
        ┌──────────────────────┼─────────────────────┐
        ▼                      ▼                     ▼
  ┌──────────┐         ┌──────────────┐      ┌──────────────┐
  │ Langfuse │         │ Salesforce   │      │ Eval harness │
  │ traces   │         │ tools (SOQL) │      │ (golden set, │
  └──────────┘         └──────────────┘      │  CI-gated)   │
                                             └──────────────┘
```

---

# WEEK 35 — Build `CodeSage` (Integration Week)

> You already have all the parts. This week is about integration, polish, and UX — things most engineers are bad at and most users notice first.

## 📅 Day-by-Day Plan

| Day | Focus | Deliverable |
|-----|-------|-------------|
| **Mon** | Architecture & project scaffold; wire LoRA adapter + RAG into a single service | Repo `projects/18-codesage-capstone/` builds and serves a stub answer |
| **Tue** | RAG quality pass — expand index to 5000+ chunks of Salesforce docs; tune chunker + reranker | Retrieval@5 ≥ 0.85 on the golden set |
| **Wed** | Fine-tuned model integrated; compare answer quality with base model on 20 prompts | Side-by-side comparison report |
| **Thu** | Agent mode: add SOQL tools, wire into a subset of queries that need live org data | Agent passes 10/12 live-org queries |
| **Fri** | UI (Gradio or Next.js) with streaming, source citations as clickable chips, error states | Working local demo; screenshot in blog draft |
| **Sat** | Docker + deploy (HF Space, Fly.io, or Railway); configure Langfuse in production | Public URL live |
| **Sun** | End-to-end evaluation: run full harness on deployed endpoint | Eval report committed; any regressions fixed |

## 🧱 Integration Checklist

- [ ] RAG retriever is a clean `Retriever` class with a `.retrieve(query, k)` API
- [ ] Model inference is abstracted behind `LLMClient.generate_stream(prompt, tools=None)` — so you can swap llama.cpp ↔ OpenAI ↔ your mini-GPT
- [ ] Tool registry has type-checked Pydantic schemas (no raw dicts)
- [ ] All long-running calls wrapped in `@observe` (Langfuse) — every production request traceable
- [ ] Redis cache keyed by `(system_prompt_hash, user_query, model_version)` — critical for cost
- [ ] CI runs eval harness on every PR; regressions block merge
- [ ] Dockerfile + docker-compose.yml work from `git clone` → `docker compose up` with no manual steps
- [ ] Graceful degradation: if LoRA adapter fails to load, fall back to base model with a log warning (do NOT crash)

## 🎨 UX Polish Checklist (the thing engineers skip)

- [ ] First token in < 1.5s (measure, then optimize — usually the retriever is the bottleneck)
- [ ] Streaming visible (don't wait for the full answer before rendering)
- [ ] Citations are **clickable** and open to the actual doc
- [ ] "I don't know" refusal is styled differently from answers (users need to see it clearly)
- [ ] Mobile-responsive (half your recruiters will open it on their phone)
- [ ] Error states: network failure, tool-call failure, rate limit — each with a distinct message
- [ ] Empty state when user first lands has 3 suggested queries (reduces thinking cost)

## 💻 Example: Integrating Everything

```python
# projects/18-codesage-capstone/main.py — structural sketch
from retrieval.hybrid import HybridRetriever
from llm.client import LLMClient
from agent.tools import salesforce_tools
from observability import langfuse_client as lf
from cache import redis_cache

retriever = HybridRetriever(chroma_path=".chroma", bm25_path=".bm25")
llm = LLMClient(
    model_path="models/phi3-Q4_K_M.gguf",
    lora_adapter="outputs/phi3-apex-lora",
)

@lf.observe(name="codesage-ask")
@redis_cache(ttl=3600)
async def ask(query: str, mode: str = "rag"):
    chunks = retriever.retrieve(query, k=5)
    prompt = build_prompt(query, chunks, tools=salesforce_tools if mode == "agent" else None)
    async for token in llm.generate_stream(prompt, max_tokens=800, temperature=0.2):
        yield token
```

## ✅ Week 35 Success Criteria

- [ ] Live URL responds with streaming, cited answers
- [ ] Eval harness green on ≥ 85% of the 30-case golden set
- [ ] Langfuse shows traces with per-stage latency
- [ ] Docker-composed deployable by someone else with no custom setup
- [ ] Architecture diagram in repo README

---

# WEEK 36 — Public Launch & Interview Prep

> Everything you've built is technical. This week is about *narrative*, *positioning*, and *readiness*. Half of this week is writing and recording, not coding.

## 📅 Day-by-Day Plan

| Day | Focus | Deliverable |
|-----|-------|-------------|
| **Mon** | Write the launch blog post: "I built an AI Engineer for Salesforce in 9 months" | Draft in `blog/launch.md` |
| **Tue** | Record the 3-minute demo video (Loom / OBS) | Video uploaded, link ready |
| **Wed** | Architecture deep-dive post: "How `CodeSage` works end-to-end" | Second blog post with diagrams |
| **Thu** | Rewrite GitHub README for `zero-to-ai-engineer` — recruiter-optimized, not student-diary-style | New README live |
| **Fri** | **Launch day:** post on dev.to, LinkedIn, Twitter/X, HackerNews (Show HN), r/learnmachinelearning | Posts shared; responses engaged with |
| **Sat** | Resume rewrite around *projects*, not past job titles; prepare 10 interview flashcards | Resume v2 committed |
| **Sun** | **Phase 8 post-exam + mock interview** (record yourself doing a 45-min system design) | Video reviewed; Phase 8 tagged `graduation` |

---

## 🎤 THE LAUNCH BLOG POST — Use This Structure

```markdown
# How I Went from Senior Salesforce Engineer to AI Engineer in 9 Months

## The hook (1 paragraph)
What you built. Show the demo GIF. URL to live demo.

## Why I did it (1 paragraph)
Credibility through vulnerability. Be honest about why you switched.

## The plan (2 paragraphs)
The 9-phase, 36-week structure. Link to your curriculum repo.

## The hardest part (1–2 paragraphs)
Be specific. "Week 12 backprop ninja" > "Neural networks were hard."

## What I built (most of the post)
The architecture. Screenshots. A short technical deep-dive on ONE interesting
sub-problem you solved (e.g., the hybrid retrieval improvements, or the
streaming endpoint, or the eval harness).

## Results (1 paragraph + table)
Your eval numbers. Your retrieval metrics. Tokens/sec. Be concrete.

## What I'd do differently
Another credibility move. Shows reflection.

## What's next
Projects you'll build. Specific job targets (without naming companies).

## Links
- Live demo
- Code
- Curriculum repo
- Every weekly blog post (they form a trail of credibility)
```

**Word-count target:** 2000–3000. Long enough to show depth; short enough to finish. Ship the post Friday morning.

---

## 🎥 THE DEMO VIDEO — Structure (Total: 3 minutes)

```
0:00 – 0:20   | Who you are + what you built (one sentence each)
0:20 – 0:40   | The live URL, click through, show the UI
0:40 – 2:00   | Three demo queries, escalating in complexity:
                1. Simple factual ("What are SOQL governor limits?") — shows RAG
                2. Code-generation ("Write a bulkified trigger for Account.Industry change") — shows fine-tune
                3. Agentic ("How many open high-priority cases do I have in my org?") — shows tool use
2:00 – 2:30   | Brief architecture diagram voice-over
2:30 – 3:00   | "Here's the code, here's the curriculum, here's my LinkedIn. Thanks."
```

**Production tips:**
- Scripted, not improvised. Write the script, then rehearse twice.
- Record the UI interaction first; do voice-over separately in a quiet room.
- Subtitles are non-negotiable for social (30% of viewers have audio off).
- Export at 1080p, 60fps if possible.

---

## 💼 RESUME REWRITE (The Rules)

Your old resume talked about Apex classes and LWC components. Your new resume has to convince a skeptical AI Engineering Manager that you're the real deal.

### Top-half priorities
1. One-line positioning headline: "Full-stack AI Engineer — LLM pretraining, fine-tuning, RAG, and production deployment — built during a 9-month self-directed curriculum"
2. Link to `CodeSage` live demo
3. Link to `zero-to-ai-engineer` curriculum repo
4. 3-line "Selected AI Projects" section with `CodeSage`, mini-GPT pretraining, and RAG eval harness

### Past experience reframing

**Before:**
> Senior Software Engineer — Salesforce (2018–2025). Built Apex triggers and Lightning Web Components for enterprise CRM workflows.

**After:**
> Senior Software Engineer — Enterprise SaaS (2018–2025). 7 years shipping mission-critical backend systems for Fortune 500 customers: high-throughput integrations (Bulk API, governor-limit-aware batch jobs), domain-specific query engines (SOQL optimization), and production observability. This engineering discipline is the foundation I now apply to AI systems: reliability, SLA-aware cost/latency trade-offs, and evaluation-driven development.

**Rule:** every bullet point must be reframed in a language that would be familiar to an AI Engineering hiring manager. Bulk API → "high-throughput batch pipeline." Apex governor limits → "strict resource-bounded execution environment." Translate, don't hide.

---

## 🧠 MOCK INTERVIEW PREP

The goal is to be fluent on **both** ends of the stack in interview settings.

### 20-Question Technical Drill (do this Saturday afternoon)

Set a timer. No notes. Record yourself. Re-watch.

**Fundamentals (10 questions):**
1. Walk me through backprop through a single Transformer block.
2. Why is cross-entropy the right loss for language modeling?
3. Explain LayerNorm vs BatchNorm and why Transformers prefer one.
4. What does the temperature parameter do, mathematically?
5. Why `Q @ K^T / sqrt(d)` and not just `Q @ K^T`?
6. Explain KV-cache: why it helps, and what memory cost it incurs.
7. Derive the LoRA update formula and explain the intuition.
8. What's the difference between perplexity and cross-entropy loss numerically?
9. Why do we use bytepair encoding instead of word- or character-level tokenization?
10. Explain positional encoding (sinusoidal vs RoPE) — why is position hard?

**Systems / Applied (10 questions):**
11. Design a RAG system that serves 10K queries/day with P95 latency < 2s.
12. When would you choose fine-tuning over RAG?
13. Your RAG system hallucinates 20% of the time. Debug it.
14. How do you evaluate a prompt change in production?
15. Your agent goes into infinite loops sometimes. Fix it.
16. Design an eval harness for a customer-support chatbot.
17. How do you decide on `chunk_size` and `chunk_overlap`?
18. Explain when you'd use DPO vs PPO vs SFT.
19. Walk me through quantization — what do you lose at Q4?
20. Your production LLM latency spiked. What's your diagnosis flowchart?

### System Design Prompt (45-minute exercise, recorded)

> *"Design an AI assistant for a large enterprise support team. It should answer internal ticket questions using company documentation, escalate when uncertain, keep audit logs, and cost less than $0.05 per query at 100K queries/day."*

Watch your recording afterwards. Note where you stall, what you forget, what you handwave. Fix it.

---

## 🏆 GRADUATION CRITERIA (End of Week 36)

You graduate the program — and are legitimately hireable as an AI Engineer — when all of the following are true:

- [ ] `CodeSage` is deployed publicly and reliably answers 8/10 curated queries correctly
- [ ] Launch blog post is live with ≥ 200 words of technical depth
- [ ] Demo video is live (YouTube or LinkedIn)
- [ ] Curriculum repo has ≥ 100 commits, weekly journal entries for 36 weeks, all 18 projects, all 8 phase exams tagged
- [ ] You passed a recorded mock system-design interview (self-reviewed and iterated on)
- [ ] Resume v2 written
- [ ] You've applied to ≥ 5 AI Engineer roles

## ➡️ What Comes After

Week 37 onwards (optional, but recommended):
- **Keep publishing.** One blog post a week. Pick a niche you care about (evals, RAG, agents) and become the person people DM about it.
- **Contribute to an OSS AI library.** Add a feature to LangGraph / LlamaIndex / TRL / Chroma. Real maintainer code review is the fastest skill accelerator there is.
- **Follow the research frontier.** Skim one new paper a week. Implement one per month.
- **Mentor someone** going through this curriculum. Teaching cements everything.

---

## 🎁 A Note From Your Instructor

You started this 9 months ago with 7 years of engineering experience and an obsession with doing things properly. Most people who attempt this transition give up around Week 10–14 because the Valley of Despair is real.

If you're reading this having *finished* the Phase 8 checklist — you are in a very small group. You didn't just learn AI. You *built* AI, from the primitives up to a deployed product, with receipts (commits, papers, journal entries, eval numbers, a blog post series).

That's the hiring signal. That's the differentiator. Use it.

> Now go build the next thing.
