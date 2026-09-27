# ═══════════════════════════════════════════════════════════════════════
# PHASE 7: AI ENGINEER TOOLKIT (Weeks 30–34)
# ═══════════════════════════════════════════════════════════════════════
# Zero to AI Engineer — 9-Month Masterclass
# ═══════════════════════════════════════════════════════════════════════
#
# This is the phase that converts your "understands AI internals" skillset
# into an "ships production AI systems" skillset. Five weeks, five modules:
#
#   Week 30 — Fine-Tuning (SFT, LoRA, QLoRA)
#   Week 31 — RAG & Vector Databases
#   Week 32 — Agents, Tools, Structured Output
#   Week 33 — Evals & Observability
#   Week 34 — Deployment & Productionization
#
# ═══════════════════════════════════════════════════════════════════════

---

## 🎯 Why a 5-Week "Toolkit" Phase?

Because the job description for "AI Engineer" in 2026 is almost never "train a model from scratch." It's:

> Take a strong off-the-shelf LLM, fine-tune it lightly on our data, wrap it in a retrieval system over our internal knowledge base, give it tools to take actions, evaluate it with regression tests, deploy it behind a streaming API, monitor it in production, and iterate.

That is a **real engineering discipline** with its own tools, patterns, and pitfalls. You already understand the models. Now you learn the system around them.

## 🔗 The Salesforce Analogy

> You have a managed package (your mini-GPT). Phase 7 is the **implementation project**: integrating it into a customer's org.
>
> - Fine-tuning = custom metadata and permission sets tailored to the org
> - RAG = **External Objects** and **Apex Callouts** to the org's data
> - Agents = **Flows** and **Apex Triggers** that take actions across the system
> - Evals = your **deployment playbook** and **smoke-test** checklist
> - Observability = **Debug Logs** and **Event Monitoring** for your AI

## 📖 The Three Decisions Every AI Engineer Must Learn to Make

Throughout this phase, pattern-match everything you see onto this decision flowchart. Interviewers love it.

```
                         NEW USE-CASE
                               │
                   ┌───────────┼───────────┐
                   │           │           │
          "Model lacks      "Model lacks   "Model lacks
           capability"      knowledge"    ability to act"
                   │           │           │
                   ▼           ▼           ▼
              FINE-TUNE       RAG         AGENTS
              (behavior)   (facts)     (tools)
                   │           │           │
                   └────────┬──┘           │
                            ▼              │
                       EVAL & MONITOR ◀────┘
                            │
                            ▼
                        DEPLOY
```

Most real systems are **all three**, layered. But understanding which lever to pull first — fine-tune vs RAG vs prompting vs tooling — is the single most underrated AI Engineer skill.

---

# WEEK 30 — Fine-Tuning: SFT, LoRA, QLoRA

## 🎯 Goal

You will:
1. Take a pre-trained model (GPT-2 small, or Phi-3-mini, or Qwen2-0.5B-Instruct).
2. Fine-tune it two ways: **full SFT** (on GPT-2) and **LoRA** (on the larger model).
3. Evaluate it *against the base model* on a golden set — prove the lift.

**Key insight to take away:** LoRA trains ~0.1–1% of the parameters and achieves 90%+ of full-fine-tune quality. On your M3 Pro, this is the difference between "possible" and "impossible."

## 📚 Resources

| # | Resource | Duration | Why |
|---|----------|----------|-----|
| 1 | [HuggingFace: Fine-tuning Tutorial](https://huggingface.co/docs/transformers/training) | ~90 min | The canonical workflow |
| 2 | [Hu et al., *LoRA*](https://arxiv.org/abs/2106.09685) | 45 min read | Paper-of-the-week |
| 3 | [HuggingFace PEFT docs](https://huggingface.co/docs/peft) | reference | LoRA/QLoRA/AdaLoRA |
| 4 | [QLoRA Paper (Dettmers 2023)](https://arxiv.org/abs/2305.14314) | 30 min skim | 4-bit quantized fine-tuning |
| 5 | [Sebastian Raschka: *Finetuning LLMs with LoRA and QLoRA*](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) | 30 min | Practical tips |
| 6 | [HuggingFace TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) | reference | The modern fine-tune API |

## 📖 Theory: What LoRA Actually Does

Given a frozen weight matrix `W ∈ R^{d×k}`, LoRA represents the update `ΔW` as a **low-rank decomposition**:

```
W_effective = W + (A @ B)       where A ∈ R^{d×r}, B ∈ R^{r×k}, r ≪ min(d, k)
```

- Only `A` and `B` are trained. Parameter count: `r·(d+k)` vs `d·k` for full FT → typical savings 99%+.
- At inference, you can either keep them separate (fast swap between task adapters) or merge them back into `W`.
- Typical rank `r = 8` or `16`. "LoRA alpha" is a scaling hyperparameter — treat `alpha/r = 1` as default.

**The key sentence on interviews:** *"LoRA assumes that the update needed to specialize a model for a task lies in a low-dimensional subspace of the weight space. Empirically, this assumption holds astonishingly well."*

## 💻 Exercise 30.1 — Data Formatting

Fine-tuning quality is dominated by **data quality**, not algorithm choices. Get this right first.

```python
# projects/14-phi3-lora-finetune/data/format.py
from datasets import load_dataset, Dataset

# Pick your own domain — Salesforce docs Q/A, or your personal journal, or
# a niche dataset. Start SMALL: 500–2000 high-quality examples beats 50k mediocre ones.

def to_chatml(row):
    """Format as ChatML (works for Phi-3, Qwen2, Mistral-Instruct, etc.)"""
    return {
        "text": (
            f"<|system|>\n{row['system']}<|end|>\n"
            f"<|user|>\n{row['user']}<|end|>\n"
            f"<|assistant|>\n{row['assistant']}<|end|>"
        )
    }

raw = Dataset.from_json("data/raw.jsonl")
formatted = raw.map(to_chatml, remove_columns=raw.column_names)
formatted.save_to_disk("data/formatted")
```

**Quality checklist before you fine-tune (do ALL of these):**
- [ ] At least 200 examples (500+ preferred for Phi-3-scale models)
- [ ] Manually read at least 50 random examples — any look wrong? fix them
- [ ] Check token length distribution — are you about to spend 90% of training on a few outliers?
- [ ] Deduplicate near-identical examples
- [ ] 80/10/10 train/val/test split — you WILL need a held-out test set for the eval step
- [ ] Ensure ChatML/template matches what the base model expects (this is the #1 silent killer)

## 💻 Exercise 30.2 — LoRA Fine-Tune on M3 Pro (Project #14)

```python
# projects/14-phi3-lora-finetune/train.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk

MODEL = "microsoft/Phi-3-mini-4k-instruct"   # or Qwen/Qwen2-0.5B-Instruct on tighter budgets

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map={"": "mps"},
)

lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                                   # rank
    lora_alpha=32,                          # alpha/r = 2 (mild amplification)
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # attention only; expand if underfitting
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()          # expect <2% trainable

dataset = load_from_disk("data/formatted")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=SFTConfig(
        output_dir="outputs/phi3-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,      # effective batch 16
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        eval_steps=50,
        save_steps=50,
        bf16=False, fp16=False,             # bf16 partial on MPS; check your torch version
        report_to="wandb",
        max_seq_length=1024,
    ),
)
trainer.train()
trainer.save_model("outputs/phi3-lora/final")
```

**On M3 Pro sizing:** Phi-3-mini (3.8B params) fits in LoRA fine-tune at batch 2 × grad-accum 8 within ~14 GB. Qwen2-0.5B is a safer starting point — iterate there, then graduate up. Don't fight OOM errors; drop model size first.

## 💻 Exercise 30.3 — Prove the Lift

**Before/after comparison on a held-out golden set:**

```python
# projects/14-phi3-lora-finetune/eval.py
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"":"mps"})
tuned = PeftModel.from_pretrained(base, "outputs/phi3-lora/final")

# Generate on the same 30 golden prompts with both models; save outputs to disk;
# score each pair {base, tuned} with a simple rubric (or an LLM-as-judge — see Week 33).

# Publish a markdown table: prompt | base | tuned | winner | notes.
# This file IS your Friday blog post material.
```

**Do not skip this.** A fine-tune you can't prove lifted anything is a fine-tune you shouldn't deploy.

## 📅 Week 30 Schedule

| Day | Focus | Deliverable |
|-----|-------|-------------|
| Mon | LoRA paper + Raschka post | Paper summary in `papers/week-30/` |
| Tue | Data formatting: build 500+ ChatML examples in your domain | `data/formatted/` with train/val/test |
| Wed | Full SFT on GPT-2 as baseline | `outputs/gpt2-sft/` + loss curves |
| Thu | LoRA fine-tune of Phi-3-mini (or Qwen2-0.5B) | `outputs/phi3-lora/final/` |
| Fri | Before/after eval on golden set; blog post | Markdown comparison table + Friday blog |
| Sat | Iterate on data quality; re-run if needed | Second run documented |
| Sun | Retro + paper-of-the-week + plan next week | Progress bar update |

## ✅ Week 30 Success Criteria

- [ ] Can explain LoRA in 2 minutes including the `A @ B` decomposition
- [ ] Fine-tuned at least one model (GPT-2 full or Phi-3 LoRA) on your own data
- [ ] Can show a side-by-side "base vs tuned" comparison on a golden set
- [ ] Model checkpoint is saved, versioned in git-lfs or HF Hub (NOT committed directly to git)

---

# WEEK 31 — RAG & Vector Databases

## 🎯 Goal

Build a production-shaped RAG system — not a toy. Chunking, embeddings, vector search, **hybrid search with BM25**, reranking, grounded generation with citations, and **evaluation with Ragas**. This is the most job-relevant module in the entire course. If you only ship one Phase 7 project, ship this one.

## 🔗 Project: "Chat with Salesforce Docs" (Project #15)

Ingest the public Salesforce developer documentation (or any domain you know) and build an app that answers natural-language questions with **cited sources** and **refuses to answer** when retrieval comes up empty.

## 📚 Resources

| # | Resource | Duration | Why |
|---|----------|----------|-----|
| 1 | Lewis et al., [*RAG paper*](https://arxiv.org/abs/2005.11401) | 1 hr read | Paper-of-the-week; originates the term |
| 2 | [Chroma docs — "Getting Started"](https://docs.trychroma.com/getting-started) | 30 min | Local vector DB |
| 3 | [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/) | 90 min | Canonical code patterns |
| 4 | [Pinecone: Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/) | 20 min | BM25 + dense |
| 5 | [Cohere Rerank docs](https://docs.cohere.com/docs/rerank-overview) | 15 min | Reranker concept |
| 6 | [Ragas](https://docs.ragas.io/) | 30 min | The eval framework |
| 7 | [Anyscale: Building RAG-based LLM applications](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1) | 90 min | Gold standard guide |
| 8 | Jerry Liu (LlamaIndex): [advanced RAG patterns](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) | 45 min | Production patterns |

## 📖 The RAG Architecture (You Should Be Able to Draw This)

```
INGEST PIPELINE (offline, one-time)
──────────────────────────────────
raw docs → clean → chunk → embed → vector DB
                    ↓
                  + BM25 sparse index
                    ↓
                  + metadata (source, page, timestamp)

QUERY PIPELINE (online, per request)
──────────────────────────────────
  user query
      ↓
  (optional) query rewrite / expansion
      ↓
  ┌────────────────┬──────────────────┐
  │  dense search  │   BM25 search    │
  │  (cosine top-k)│   (top-k)        │
  └────────┬───────┴──────┬───────────┘
           └──── FUSE (RRF) ────┐
                                ▼
                         RERANKER top-n
                                ▼
             grounded prompt: user Q + retrieved chunks
                                ▼
                     LLM generates ANSWER + citations
                                ▼
                       (optional) citation check
                                ▼
                            response
```

## 📖 The 7 Things That Make RAG Fail (And How to Fix Them)

| Failure | Cause | Fix |
|---------|-------|-----|
| 1. Answer looks right, source is wrong | No citation check | Parse citations, verify substring overlap with retrieved chunks |
| 2. Doesn't retrieve the right chunks | Bad chunking (too coarse / too fine) | Tune `chunk_size=500`, `chunk_overlap=50`; use semantic chunking |
| 3. Gets acronyms wrong | Dense embeddings miss exact matches | Add **BM25** to hybrid search |
| 4. Top-5 retrieved but right chunk at rank 7 | No reranker | Add a cross-encoder reranker (`cohere-rerank` or `bge-reranker-v2`) |
| 5. Hallucinates when context is empty | No refusal | Add explicit "if context is empty, say you don't know" in system prompt |
| 6. Slow first-token latency | Serial retrieval+gen | Parallelize retrieval + model warmup; stream generation |
| 7. Same question gives different answers | Temperature too high | Drop to 0.1–0.3 for factual QA |

## 💻 Exercise 31.1 — Minimal RAG With Chroma + Sentence-Transformers

```python
# projects/15-sf-docs-rag/ingest.py
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-en-v1.5"
)
client = chromadb.PersistentClient(path=".chroma")
col = client.get_or_create_collection("sf_docs", embedding_function=ef)

docs = []                         # populate from scraped Salesforce docs
for path in Path("data/raw").glob("**/*.md"):
    text = path.read_text()
    docs.append({"id": str(path), "text": text, "source": str(path)})

# Chunk each doc
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks, metadatas, ids = [], [], []
for d in docs:
    for i, ch in enumerate(splitter.split_text(d["text"])):
        chunks.append(ch)
        metadatas.append({"source": d["source"], "chunk_idx": i})
        ids.append(f"{d['id']}::{i}")

col.add(documents=chunks, metadatas=metadatas, ids=ids)
print(f"Indexed {len(chunks)} chunks.")
```

```python
# projects/15-sf-docs-rag/query.py
def retrieve(query: str, k: int = 5):
    return col.query(query_texts=[query], n_results=k)

SYSTEM = """You are a Salesforce documentation assistant.
Answer ONLY using the context below. If the context does not contain the answer,
respond with "I don't know based on the provided documentation." Cite sources
inline with [source] markers."""

def rag_answer(query: str, llm_generate):
    hits = retrieve(query, k=5)
    context = "\n\n".join(
        f"[{m['source']}]\n{d}"
        for d, m in zip(hits["documents"][0], hits["metadatas"][0])
    )
    prompt = f"{SYSTEM}\n\n<context>\n{context}\n</context>\n\nQUESTION: {query}\nANSWER:"
    return llm_generate(prompt, temperature=0.2)
```

## 💻 Exercise 31.2 — Add BM25 Hybrid Search

Use `rank_bm25`:

```python
from rank_bm25 import BM25Okapi
tokenized_corpus = [doc.lower().split() for doc in chunks]
bm25 = BM25Okapi(tokenized_corpus)

def hybrid_retrieve(query, k=5):
    dense_hits = col.query(query_texts=[query], n_results=20)
    dense_ranked = [(doc, i + 1) for i, doc in enumerate(dense_hits["documents"][0])]

    scores = bm25.get_scores(query.lower().split())
    bm25_top = sorted(range(len(scores)), key=lambda i: -scores[i])[:20]
    bm25_ranked = [(chunks[i], j + 1) for j, i in enumerate(bm25_top)]

    # Reciprocal Rank Fusion (RRF) — simple, effective
    rrf = {}
    for doc, rank in dense_ranked: rrf[doc] = rrf.get(doc, 0) + 1 / (60 + rank)
    for doc, rank in bm25_ranked:  rrf[doc] = rrf.get(doc, 0) + 1 / (60 + rank)
    return sorted(rrf.items(), key=lambda x: -x[1])[:k]
```

## 💻 Exercise 31.3 — Add a Reranker

```python
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query, candidates, top_n=5):
    pairs = [(query, doc) for doc, _ in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c[0] for c in reranked[:top_n]]
```

**Typical lift:** hybrid retrieval + reranker usually adds 10–20 points of Retrieval@5 over naive dense-only on real-world corpora. Measure it with Ragas.

## 💻 Exercise 31.4 — Evaluate with Ragas

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Build a golden set of 30 Q/A pairs BEFORE you trust your system.
eval_set = Dataset.from_dict({
    "question": [...],
    "answer": [...],       # your system's answers
    "contexts": [...],     # list of retrieved chunks
    "ground_truth": [...], # human-authored correct answer
})

result = evaluate(eval_set, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(result)
```

**Targets to aim for:**
- `context_recall` ≥ 0.85 (you're retrieving the right stuff)
- `faithfulness` ≥ 0.90 (answers don't hallucinate beyond retrieved context)
- `answer_relevancy` ≥ 0.80

## 📅 Week 31 Schedule

| Day | Focus | Deliverable |
|-----|-------|-------------|
| Mon | Read RAG paper; draw the architecture from memory | Paper summary + architecture sketch |
| Tue | Ingest pipeline: scrape/clean/chunk/embed Salesforce docs | Chroma index populated |
| Wed | Basic dense RAG with grounded prompt + citations | Working `rag_answer()` with 5 manual test queries |
| Thu | Hybrid (BM25) search + reranker | Retrieval@5 improved over dense-only (measure!) |
| Fri | Ragas evaluation + refusal behavior + Friday blog post | Ragas scores committed; blog post live |
| Sat | Build 30-Q golden set + iterate to targets | Golden set in `eval/golden.jsonl` |
| Sun | Retro + paper-of-the-week | Progress update |

## ✅ Week 31 Success Criteria

- [ ] "Chat with Salesforce Docs" answers 8/10 manual test queries correctly with citations
- [ ] Ragas context_recall ≥ 0.85, faithfulness ≥ 0.90
- [ ] Refuses to answer on out-of-scope queries
- [ ] System works end-to-end in a single `.py` or Gradio UI

---

# WEEK 32 — Agents, Tools, Structured Output

## 🎯 Goal

Go from "chatbot that answers" to "agent that takes actions." The agent will call your tools (functions you define), get structured outputs back, handle errors, and reason across multiple steps. Specifically: **a SOQL agent** that queries a Salesforce scratch org and returns grounded answers about your records.

## 🔗 Project: SOQL-writing Agent (Project #16)

Given natural language like "how many open high-priority cases assigned to me?", the agent must:
1. Decide it needs the Salesforce schema
2. Call the `describe_object` tool to get fields for `Case`
3. Construct valid SOQL
4. Call `run_soql` tool
5. Return a formatted answer

## 📚 Resources

| # | Resource | Duration | Why |
|---|----------|----------|-----|
| 1 | Yao et al., [*ReAct*](https://arxiv.org/abs/2210.03629) | 1 hr | Paper-of-the-week; the reasoning+acting pattern |
| 2 | [OpenAI: Function Calling guide](https://platform.openai.com/docs/guides/function-calling) | 30 min | The standard interface |
| 3 | [Instructor library](https://python.useinstructor.com/) | 30 min | Pydantic-validated LLM output |
| 4 | [LangGraph tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/) | 90 min | Agent orchestration |
| 5 | [Anthropic: *Building effective agents*](https://www.anthropic.com/research/building-effective-agents) | 30 min | Best practical essay on agents |
| 6 | [DSPy](https://dspy.ai/) | 60 min | Declarative prompt programming |

## 📖 Theory: The Agent Loop (ReAct)

```
while not done:
    thought = LLM(system_prompt + history + "Thought:")
    action  = LLM(system_prompt + history + thought + "Action:")
    if action is TOOL_CALL:
        observation = tools[action.name](**action.args)
    elif action is FINAL_ANSWER:
        done = True
        break
    history += (thought, action, observation)
```

The whole trick of modern agents is: **let the LLM decide what to do next, based on what it just observed**. This sounds trivial. Making it reliable is 80% of the engineering.

## 📖 The Four Agent Failure Modes (Interview Gold)

1. **Tool-call hallucination** — LLM invents a tool that doesn't exist, or calls one with wrong args.
   → Fix: strict JSON schema validation (Instructor/Pydantic) + retry on parse failure.
2. **Infinite loops** — agent calls the same tool over and over.
   → Fix: hard step cap (`max_iterations=10`) + step-budget prompting.
3. **Lost context** — agent forgets an earlier observation.
   → Fix: summarize-as-you-go, or use a scratchpad with explicit pointers.
4. **Premature finalization** — agent says "done" without solving the task.
   → Fix: self-consistency check (another LLM call: "does this answer the original question?") or eval on final answer.

## 💻 Exercise 32.1 — Structured Output with Instructor

```python
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI   # or any client; Instructor wraps many

client = instructor.from_openai(OpenAI())

class SOQLQuery(BaseModel):
    query: str = Field(description="A syntactically valid SOQL string.")
    reasoning: str = Field(description="Why you chose these fields and filters.")

def nl_to_soql(natural_language: str) -> SOQLQuery:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=SOQLQuery,
        max_retries=3,
        messages=[
            {"role": "system", "content": "Convert the user's question to SOQL."},
            {"role": "user", "content": natural_language},
        ],
    )
```

Instructor's value: **you get a validated Python object, not a string you have to parse**. It auto-retries on `ValidationError` and asks the model to fix the issue.

## 💻 Exercise 32.2 — The SOQL Agent Loop (Project #16)

```python
# projects/16-soql-agent/agent.py — simplified sketch
from typing import Callable, Any

TOOLS: dict[str, Callable] = {}

def tool(name: str):
    def wrap(fn): TOOLS[name] = fn; return fn
    return wrap

@tool("describe_object")
def describe_object(sobject: str) -> dict:
    """Return field metadata for a Salesforce SObject."""
    return sf_client.describe(sobject)      # e.g. via simple_salesforce

@tool("run_soql")
def run_soql(query: str) -> list[dict]:
    return sf_client.query_all(query)["records"]

@tool("final_answer")
def final_answer(answer: str) -> str:
    return answer

SYSTEM = """You are a SOQL agent. You have tools:
- describe_object(sobject: str): get field info
- run_soql(query: str): execute SOQL
- final_answer(answer: str): return final answer to the user

Think step-by-step. Always describe_object BEFORE running SOQL on an unfamiliar object.
If SOQL errors, revise and retry. Call final_answer when done.
"""

def run_agent(user_query: str, max_steps: int = 8):
    history = [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": user_query}]
    for step in range(max_steps):
        action = llm_call_with_tools(history, tools=list(TOOLS.values()))
        if action.name == "final_answer":
            return action.args["answer"]
        try:
            observation = TOOLS[action.name](**action.args)
        except Exception as e:
            observation = f"ERROR: {e}"
        history.append({"role": "assistant", "content": str(action)})
        history.append({"role": "tool", "name": action.name, "content": str(observation)[:2000]})
    return "Step budget exceeded."
```

## 💻 Exercise 32.3 — Evaluate the Agent

Build a test set of 15 natural-language questions. Label each with:
- Correct final answer (ground truth)
- Correct tool sequence (e.g., `describe→run_soql→final`)

Evaluate:
- **Task success** = fraction with correct final answer
- **Tool-call accuracy** = fraction with correct tool sequence
- **Step efficiency** = avg steps used vs budget

Target: task success ≥ 80%, step efficiency ≤ 50% of budget.

## 📅 Week 32 Schedule

| Day | Focus | Deliverable |
|-----|-------|-------------|
| Mon | ReAct paper + Anthropic *Building effective agents* | Paper summary + agent failure-mode list |
| Tue | Instructor + Pydantic structured output (Ex 32.1) | `nl_to_soql()` with 10 test queries |
| Wed | Build tools + agent loop (Ex 32.2) | `run_agent()` works on 3 manual test queries |
| Thu | Error handling + step cap + `final_answer` pattern | Robust agent survives malformed SOQL |
| Fri | 15-question test set + evaluation + blog post | Evaluation results + Friday blog |
| Sat | Optional: rebuild in LangGraph for experience | LangGraph version committed |
| Sun | Retro + paper-of-the-week | Progress update |

## ✅ Week 32 Success Criteria

- [ ] Agent passes ≥ 12/15 test queries with correct final answer
- [ ] Malformed-output rate < 2% (Instructor retry works)
- [ ] Step cap prevents runaway loops (verified with adversarial prompt)
- [ ] Can articulate the 4 agent failure modes and their fixes

---

# WEEK 33 — Evals & Observability

## 🎯 Goal

Stop guessing whether your LLM changes help. Build a **prompt regression-test harness** (golden set + LLM-as-judge) and wire up **observability** (Langfuse or Phoenix) so every production request is traceable. Without these, you cannot safely ship LLM systems — period.

## 📚 Resources

| # | Resource | Duration | Why |
|---|----------|----------|-----|
| 1 | Zheng et al., [*MT-Bench / LLM-as-Judge*](https://arxiv.org/abs/2306.05685) | 45 min | Paper-of-the-week |
| 2 | [PromptFoo docs](https://promptfoo.dev/) | 45 min | The canonical prompt-regression tool |
| 3 | [Langfuse quickstart](https://langfuse.com/docs/get-started) | 30 min | Self-hostable LLM observability |
| 4 | [Arize Phoenix](https://docs.arize.com/phoenix) | 30 min | OSS alternative |
| 5 | Hamel Husain, [*Your AI product needs evals*](https://hamel.dev/blog/posts/evals/) | 45 min | Essential blog |
| 6 | Eugene Yan, [*Task-specific LLM evals*](https://eugeneyan.com/writing/llm-evaluators/) | 30 min | Practical patterns |

## 📖 The Three Layers of Eval

1. **Unit tests (deterministic)** — "for input X, output must contain substring Y" or "must be valid JSON." Fast, cheap, first line of defense.
2. **Reference-based metrics** — ROUGE, BLEU, exact match vs golden. Good for summarization, translation, QA.
3. **LLM-as-judge (rubric-based)** — a strong model scores your model's output against a rubric. Good for subjective tasks: helpfulness, tone, style. Expensive and biased; use alongside reference-based metrics when possible.

The real-world mix for a production app is about 60% unit tests, 25% reference metrics, 15% LLM-as-judge.

## 📖 The Eval Harness Design Pattern

```
┌──────────────────┐      ┌──────────────────┐      ┌────────────────┐
│  Golden Set      │ ───▶ │  Under-Test      │ ───▶ │  Judges        │
│  (prompts + gt)  │      │  System (your    │      │  (unit tests + │
│                  │      │   LLM pipeline)  │      │   LLM-as-judge)│
└──────────────────┘      └──────────────────┘      └────────┬───────┘
                                                              │
                                                              ▼
                                                     ┌────────────────┐
                                                     │  Report        │
                                                     │  (pass rate,   │
                                                     │   deltas vs    │
                                                     │   last commit) │
                                                     └────────────────┘
```

**Rule:** the golden set is version-controlled. Every prompt change runs the harness. Regressions block merges. **This is CI/CD for prompts.**

## 💻 Exercise 33.1 — Prompt Regression Harness (Project #17)

```python
# projects/17-eval-harness/run_evals.py
import json, yaml
from pathlib import Path
from openai import OpenAI

judge = OpenAI()

RUBRIC = """Score the ASSISTANT response from 1-5 on each axis:
- correctness: matches the golden_answer in substance
- faithfulness: makes no claims beyond what's supported
- tone: professional and helpful
Return JSON: {"correctness": int, "faithfulness": int, "tone": int, "notes": str}
"""

def llm_judge(query, answer, golden):
    prompt = f"QUERY: {query}\nGOLDEN: {golden}\nASSISTANT: {answer}\n{RUBRIC}"
    resp = judge.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        response_format={"type":"json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def run_suite(suite_path: Path, system_under_test):
    suite = yaml.safe_load(suite_path.read_text())
    results = []
    for case in suite["cases"]:
        answer = system_under_test(case["query"])
        scores = llm_judge(case["query"], answer, case["golden"])
        # Unit tests
        must_contain = all(s.lower() in answer.lower() for s in case.get("must_contain", []))
        must_not_contain = all(s.lower() not in answer.lower() for s in case.get("must_not_contain", []))
        results.append({
            "id": case["id"], "answer": answer, **scores,
            "must_contain_pass": must_contain,
            "must_not_contain_pass": must_not_contain,
        })
    return results
```

Golden-set format (yaml), committed to `eval/golden.yaml`:

```yaml
cases:
  - id: GQ-001
    query: "What are the governor limits for SOQL queries per transaction?"
    golden: "Salesforce allows up to 100 SOQL queries per transaction..."
    must_contain: ["100", "SOQL"]
    must_not_contain: ["I'm not sure"]
```

Wire it into CI (`.github/workflows/evals.yml`) so every PR runs evals.

## 💻 Exercise 33.2 — Observability with Langfuse

```python
from langfuse import Langfuse
lf = Langfuse()

@lf.observe(name="rag-pipeline")
def rag_answer(query):
    with lf.trace(name="retrieve") as span:
        chunks = retrieve(query)
        span.update(output={"num_chunks": len(chunks)})
    with lf.trace(name="generate") as span:
        answer = llm(build_prompt(query, chunks))
        span.update(output={"tokens": count_tokens(answer)})
    return answer
```

Every request now shows up in the Langfuse dashboard: latency per stage, token costs, full trace. This is what you send to your manager when they ask "what's happening in production?"

## 📅 Week 33 Schedule

| Day | Focus | Deliverable |
|-----|-------|-------------|
| Mon | Hamel Husain + MT-Bench paper | Paper summary + eval-layer sketch |
| Tue | Build golden-set YAML (30 cases); build judge prompt | `eval/golden.yaml` committed |
| Wed | Build the harness (Ex 33.1); run against Week 31 RAG | Baseline eval report committed |
| Thu | Langfuse integration into the RAG + agent systems | Traces visible in dashboard |
| Fri | Wire harness into GitHub Actions; Friday blog | CI runs on every PR; blog post live |
| Sat | Add PromptFoo for comparative prompt A/B testing | PromptFoo config + first A/B run |
| Sun | Retro + paper-of-the-week | Progress update |

## ✅ Week 33 Success Criteria

- [ ] 30+ case golden set committed, running as CI on every PR
- [ ] Harness reports at least 3 metrics per case (unit test, reference, judge)
- [ ] Langfuse dashboard shows traces from both RAG (Week 31) and Agent (Week 32) systems
- [ ] Can explain the three layers of eval with a concrete example of when each fails

---

# WEEK 34 — Deployment & Productionization

## 🎯 Goal

Go from "works in a notebook" to "answers HTTP requests reliably under load." Streaming responses, caching, token accounting, quantization for cheaper inference, a conceptual tour of DPO/RLHF, and a proper Docker-ready API.

## 📚 Resources

| # | Resource | Duration | Why |
|---|----------|----------|-----|
| 1 | [FastAPI: streaming responses](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse) | 30 min | The SSE pattern |
| 2 | [llama.cpp GGUF guide](https://github.com/ggerganov/llama.cpp) | 30 min | Apple Silicon–optimized inference |
| 3 | [vLLM intro](https://docs.vllm.ai/en/latest/) | 30 min | Production serving |
| 4 | Rafailov et al., [*Direct Preference Optimization*](https://arxiv.org/abs/2305.18290) | 1 hr | Paper-of-the-week |
| 5 | [HuggingFace TRL DPO example](https://huggingface.co/docs/trl/dpo_trainer) | 45 min | DPO in code |
| 6 | [Gradio docs](https://www.gradio.app/docs) | 30 min | Quick UIs |
| 7 | Chip Huyen, *Designing ML Systems* Ch. 10–11 | 3 hrs | Deployment + monitoring |

## 📖 Theory: The Production Stack You Need

```
┌─────────────┐
│   Gradio /  │  ← user interface (Gradio for demo; Next.js for real prod)
│   Next.js   │
└──────┬──────┘
       │ HTTPS/SSE
┌──────▼──────┐
│   FastAPI   │  ← request handler
│   (stream)  │
└──────┬──────┘
       │
┌──────▼──────┐     ┌────────────┐
│   Redis     │◀────│  Cache     │  ← (query, response) cache for common prompts
│   (cache)   │     │   key      │
└──────┬──────┘     └────────────┘
       │
┌──────▼──────┐
│  Inference  │  ← llama.cpp server / vLLM / OpenAI API
│   server    │
└──────┬──────┘
       │
┌──────▼──────┐
│  Langfuse   │  ← observability (Week 33)
│  + Evals    │
└─────────────┘
```

## 📖 Quantization Quick Reference

| Format | Bits | Size (7B) | Quality Loss | Speed on M3 Pro |
|--------|------|-----------|--------------|-----------------|
| FP32 | 32 | 28 GB | 0% | too large |
| FP16 | 16 | 14 GB | ~0% | slow |
| Q8_0 | 8 | 7 GB | negligible | fast |
| Q5_K_M | ~5 | 4.8 GB | ~1% | very fast |
| **Q4_K_M** | ~4 | **4.0 GB** | ~2-3% | **recommended default** |
| Q3_K_M | ~3 | 3.3 GB | ~5% | fastest (quality drops) |

**On M3 Pro:** a 7B-parameter model in Q4_K_M fits in <4 GB RAM and runs >20 tokens/sec in `llama.cpp`. Think about that — a 7B LLM on your laptop at real-time speeds.

## 💻 Exercise 34.1 — FastAPI Streaming Endpoint

```python
# src/agent/api.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio, redis, hashlib, json

app = FastAPI()
r = redis.Redis()

class Query(BaseModel):
    query: str

def cache_key(q: str) -> str:
    return "rag:" + hashlib.sha256(q.encode()).hexdigest()

async def stream_answer(q: str):
    # Check cache
    if cached := r.get(cache_key(q)):
        yield f"data: {json.dumps({'cached': True})}\n\n"
        yield f"data: {cached.decode()}\n\n"
        return
    buf = ""
    async for token in llm_stream(q):      # your generator
        buf += token
        yield f"data: {json.dumps({'token': token})}\n\n"
    r.setex(cache_key(q), 3600, buf)       # cache for 1 hr
    yield f"data: {json.dumps({'done': True, 'tokens': len(buf)})}\n\n"

@app.post("/ask")
async def ask(payload: Query):
    return StreamingResponse(stream_answer(payload.query), media_type="text/event-stream")
```

**Test with `curl`:**
```bash
curl -N -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "what are SOQL governor limits?"}'
```

## 💻 Exercise 34.2 — Quantize & Serve with llama.cpp

```bash
# Download a gguf-quantized model (or convert your own)
huggingface-cli download bartowski/Phi-3-mini-4k-instruct-GGUF \
    --include "Phi-3-mini-4k-instruct-Q4_K_M.gguf"

# Run the server
./llama.cpp/server \
    -m Phi-3-mini-4k-instruct-Q4_K_M.gguf \
    --port 8080 \
    -c 4096 \
    --n-gpu-layers 99   # offload everything to MPS

# Benchmark
curl http://localhost:8080/completion -d '{"prompt":"Hello","n_predict":128}' -w "\n%{time_total}s\n"
```

Measure tokens/sec across Q3/Q4/Q5/Q8 quantizations. Publish the chart.

## 💻 Exercise 34.3 — DPO (Concept + Tiny Demo)

DPO is RLHF without the PPO complexity: given preference pairs (chosen, rejected), directly optimize the log-probability ratio. Mathematically:

```
L_DPO = -E[log σ(β · (log π_θ(chosen|x) - log π_ref(chosen|x)
                        - log π_θ(rejected|x) + log π_ref(rejected|x)))]
```

Translation: "make my model more likely than the reference on chosen, and less likely on rejected." No reward model, no PPO, no GAE — just BCE on preference pairs.

```python
from trl import DPOTrainer, DPOConfig
# training on 1k–10k preference pairs; start from your Week 30 SFT checkpoint
trainer = DPOTrainer(
    model=sft_model, ref_model=ref_model, tokenizer=tokenizer,
    train_dataset=preferences, args=DPOConfig(beta=0.1, ...)
)
```

On M3 Pro, you can run DPO on a tiny model (Qwen2-0.5B) with a small preference set — enough to *feel* the mechanics. Don't try to replicate ChatGPT's RLHF here; the goal is conceptual understanding.

## 📅 Week 34 Schedule

| Day | Focus | Deliverable |
|-----|-------|-------------|
| Mon | DPO paper + read Chip Huyen Ch. 10 | Paper summary + production stack diagram |
| Tue | FastAPI streaming + Redis cache (Ex 34.1) | Working `/ask` streaming endpoint |
| Wed | llama.cpp quantized serving + benchmark (Ex 34.2) | Benchmark chart across quantization levels |
| Thu | DPO mini-demo on Qwen2-0.5B (Ex 34.3) | Before/after preference evaluation |
| Fri | Dockerize the whole stack; Friday blog post | `Dockerfile` + `docker-compose.yml` + blog |
| Sat | Deploy to HuggingFace Space or Fly.io | Public URL shared |
| Sun | **Phase 7 post-exam** + retrospective | Phase 7 complete, tagged `phase-7-complete` |

## ✅ Week 34 / Phase 7 Final Success Criteria

- [ ] `/ask` endpoint streams responses with caching; verified with `curl`
- [ ] 7B model quantized to Q4 runs > 20 tokens/sec on M3 Pro
- [ ] DPO mini-demo shows measurable shift on preference set
- [ ] Full stack (API + vector DB + frontend) Dockerized and deployable
- [ ] **Phase 7 post-exam** ≥ 80%
- [ ] 5-minute Loom: "When to RAG vs fine-tune vs prompt"

---

## 🏆 PHASE 7 MILESTONE CHECK

By the end of Week 34 you should be able to:

1. Decide fine-tune vs RAG vs prompt vs agent for any new use-case and defend the choice.
2. Fine-tune a small LLM with LoRA on domain data *and prove the lift on a golden set.*
3. Build a production-grade RAG system with hybrid search, reranking, and grounded citations.
4. Build an agent that uses tools, handles malformed output, and has a step budget.
5. Write regression tests for prompts; wire them into CI.
6. Trace every production request in an observability tool.
7. Serve quantized models at real-time speed on consumer hardware.
8. Explain DPO vs PPO in 3 minutes.

If 6+ feel uncertain, extend Phase 7 by 1–2 weeks before starting the capstone.

## 📚 Phase 7 Project Outputs

| # | Project | Repo Folder | Mini-project Ladder # |
|---|---------|-------------|----------------------|
| 1 | LoRA fine-tune of Phi-3-mini on your domain | `projects/14-phi3-lora-finetune/` | 14 |
| 2 | Chat with Salesforce Docs RAG | `projects/15-sf-docs-rag/` | 15 |
| 3 | SOQL Agent | `projects/16-soql-agent/` | 16 |
| 4 | Prompt Regression Eval Harness | `projects/17-eval-harness/` | 17 |
| 5 | Streaming FastAPI + quantized serving | `src/agent/api.py` + `docker/` | — |

## ➡️ Next

[Phase 8: Capstone & Public Launch (Weeks 35–36)](./Phase_8_Capstone_and_Launch.md) — Everything you've built, integrated into one polished public portfolio piece: `CodeSage`.
