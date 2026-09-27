# HOW TO RUN THIS
### The operating manual — read once, then refer back

---

## 1. Where everything lives

Everything goes in **one git repo**: `~/Projects/zero-to-ai-engineer/`

```
~/Projects/zero-to-ai-engineer/
├── CLAUDE.md                  ← makes Claude Code load the mentor role in Cursor
├── README.md                  ← public progress tracker (GitHub front page)
├── PROGRESS.md                ← LIVING STATE. Claude maintains this.
│
├── curriculum/
│   ├── HOW_TO_RUN_THIS.md     ← this file
│   ├── MENTOR_PROTOCOL.md     ← how Claude teaches you
│   ├── CURRICULUM.md          ← the full 11-stage program
│   └── Stage_0_and_1_Setup_and_Math.md
│
├── src/                       ← importable modules you build
│   ├── mymath/                  (Stage 1: your NumPy library)
│   └── micrograd/               (Stage 1: your autograd engine)
├── notebooks/                 ← exploration, one folder per stage
├── projects/                  ← the 33 numbered projects
├── drills/                    ← Re-Implementation Drill attempts
├── journal/                   ← daily entries + weekly retros
├── papers/                    ← paper-of-the-week summaries
├── blog/                      ← Friday posts
├── anki/                      ← exported decks
│
├── reference_code/            ← from your v1 curriculum — still useful
└── assessments/               ← pre/post notebooks from v1
```

**One repo, everything versioned.** Your progress file lives beside your code, so `git log` is a literal record of the fifteen months.

### Setting it up (run once)

```bash
mkdir -p ~/Projects
mv ~/Downloads/AI_Learning ~/Projects/zero-to-ai-engineer
cd ~/Projects/zero-to-ai-engineer

mkdir -p curriculum src/mymath src/micrograd notebooks projects drills \
         journal/retros papers blog anki

mv CURRICULUM.md MENTOR_PROTOCOL.md HOW_TO_RUN_THIS.md Stage_*.md curriculum/
# PROGRESS.md, CLAUDE.md and README.md stay at the root

git init
git add -A
git commit -m "stage-0/wk0.1: curriculum v2 + learning system"
```

Then create the repo on GitHub (public) and push.

**Last step:** in the Claude desktop app, click **Add folder** and add `~/Projects/zero-to-ai-engineer`. That's how I reach your files from here. You can remove the old `AI_Learning` folder from the connected list afterwards.

---

## 2. The two-tool setup

You'll use Claude in two different places, for two different jobs. This split matters — using the wrong one for a task wastes your time.

### Claude Code in Cursor — your daily driver

Install once:

```bash
npm install -g @anthropic-ai/claude-code
```

Then, in Cursor's integrated terminal, from the repo root:

```bash
cd ~/Projects/zero-to-ai-engineer
claude
```

The `CLAUDE.md` at the repo root loads automatically. It tells Claude Code to read `MENTOR_PROTOCOL.md` and `PROGRESS.md` before doing anything — so it arrives already knowing where you are and how to treat you.

**Use it for:**
- Pass 2 — the main coding session
- Debugging ("my gradients don't match, here's the file")
- Code review ("review `src/mymath/vectors.py`")
- Gate 3 and Gate 4 verification — it can run your tests
- Updating `PROGRESS.md` at the end of a session
- Committing

It reads and writes your actual files. You don't paste code at it.

### The desktop app (where you are now) — teaching and gates

**Use it for:**
- Pass 1 — conceptual teaching before you touch code
- **Gate 5 defences** — these are conversations, not file operations
- Sunday retrospective and planning
- Re-Implementation Drill supervision
- When you're stuck *conceptually* rather than mechanically ("I don't understand why backprop goes backwards")
- Writing the next Stage detail file when you approach it

**The rule of thumb:** if the answer involves your files, use Cursor. If the answer involves your head, use the app.

Both read the same `PROGRESS.md`, so they stay in sync — as long as whichever one you used updates it before you stop.

---

## 3. The daily loop (~2 hrs on a weekday)

### Pass 1 — WHY (20 min, morning)

Watch the video or read the section. **No code. No notes yet. Don't pause every ten seconds.**

Finish by writing in `journal/YYYY-Wnn.md`:
- 3–5 bullets: "what is this for?"
- One question you couldn't answer

### Pass 2 — HOW (75 min, evening) ← the real work

In Cursor, with Claude Code running.

- Write the code yourself. Claude reviews, hints, debugs — it does not author.
- Derive one equation by hand, on paper.
- **Break something on purpose once.** Wrong shape, wrong sign, wrong learning rate. Watch the failure. This is where debugging intuition comes from and it cannot be read into existence.
- Commit before you stop, even if it's `wip:`.

### Pass 3 — WHY-IT-WORKED (15 min, before bed)

Close every tab. In your journal, explain today's concept in five sentences to your pre-AI self, using no jargon you don't define.

**If you can't do it in five sentences, you didn't learn it.** Mark it `revisit` and do a second Pass 2 tomorrow. This is the single most reliable honesty check in the program.

---

## 4. The weekly loop (15–20 hrs)

| Day | What | Hours |
|---|---|---|
| **Mon** | Daily loop + open the paper-of-the-week during Pass 1 | 2 |
| **Tue–Thu** | Daily loop | 2 each |
| **Fri** | Daily loop + **blog post** (300 words minimum) | 2.5 |
| **Sat** | Project work + Anki review | 4 |
| **Sun AM** | Paper summary + retro + **session with me** | 2 |
| **Sun PM** | Rest. Non-negotiable. | — |

### The Minimum Viable Week

A release lands, someone gets sick, you travel. On those weeks the target is **3 hours, not zero**:

- One Pass 2 on one concept (90 min)
- Anki (30 min)
- One commit
- Tell me it's an MVW so I log it as a deliberate reduction, not a stall

A three-hour week is a week. A zero week starts a habit of zero weeks. Protect the streak over the volume — this is the single most important sentence in this document.

---

## 5. How progress tracking works

**I maintain `PROGRESS.md`, not you.** But I can only do it if you end sessions properly.

At the end of any real session, say: **"Update progress."**

I then write: hours logged, any gate that passed with its date, anything you struggled with, what you owe me, and where the next session starts.

### The struggle log is the important part

Every concept you got wrong, found hard, or needed twice goes in there — *even if you eventually got it*. That's what I use to decide what comes back for review and what to target with a Re-Implementation Drill. Don't let me write "went well" when it didn't; the log is only useful if it's honest.

### The gate log

No gate counts until it's logged with a date. Five gates per concept:

```
1. DRAW IT        — sketch it from memory
2. DERIVE IT      — write the equation, including a gradient
3. CODE IT        — NumPy only, no autograd
4. PRODUCTION IT  — PyTorch, batched, asserts equivalence with (3)
5. DEFEND IT      — explain it to me, survive three follow-ups
```

Gates 3 and 4 get verified in Cursor, where I can run your code. Gate 5 happens here, in conversation.

### Sunday, in the app

Say: **"Sunday retro."**

I'll read `PROGRESS.md`, ask you the retrospective questions, update the file, and set the next week's top three.

---

## 6. What to say to me

Literal phrases. Use them; they load the right mode.

| Situation | Say |
|---|---|
| Starting a coding session (Cursor) | `Mentor session. Stage 1, module 1.2.` |
| Starting a teaching session (app) | `Pass 1 on [topic]. Teach me.` |
| You're stuck | `Stuck on X. I tried Y. Expected Z, got W.` |
| Ready to be tested | `I want to attempt Gate 5 on [concept].` |
| Code review | `Review src/mymath/vectors.py` |
| End of session | `Update progress.` |
| Sunday | `Sunday retro.` |
| Monthly | `Give me a Re-Implementation Drill.` |
| Approaching a new stage | `I'm a week out from Stage N. Write the detail file.` |

### When you're stuck — the format matters

Don't say "it's not working." Say:

```
Stuck on: the gradient check in micrograd
I tried: printing the graph, checking topological order
I expected: grad of 1.0 at the root
I got: 0.0 everywhere
```

That format gets you a useful answer in one round instead of four. It's also just good debugging practice — half the time writing it out solves it before you hit send.

---

## 7. What I will and won't do

**I will:** teach, hint, review, debug alongside you, test you, refuse to advance you past an unpassed gate, and maintain your progress file.

**I won't:** write your project code. That's the deal, and it's in `MENTOR_PROTOCOL.md` so future sessions of me honour it too.

If you ask me to just write `cosine_similarity`, I'll decline and ask you a question instead. That's not me being difficult — handing you working code is how you end up fifteen months in with a repo full of things you can't explain. Debugging code you wrote is worth ten times reading code I wrote.

**When I say you're not ready to move on, argue with me if you disagree** — but bring evidence, not frustration. If you can pass the gate, pass it.

---

## 8. Your first week, day by day

### Day 1 (2 hrs) — Environment
Follow `Phase_0_Environment_and_Mindset_Setup.md`. Finish when this prints `True`:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Also: `pip install pytest matplotlib jupyterlab`

### Day 2 (2 hrs) — The repo
Run the setup commands in §1. Create the GitHub repo, public, push. Add the folder in the desktop app. Install Claude Code, run `claude` in the repo, confirm it greets you knowing your stage.

### Day 3 (2 hrs) — Anki + journal
Install Anki, create deck `S1-math`, write your first five cards. Start `journal/2026-W40.md`.

### Day 4 (2 hrs) — Baseline
Run `assessments/00_Foundation_Assessment_Exercises.ipynb`. Score yourself honestly — nobody sees it but us, and an inflated number costs you in month four. Report the score to me.

### Day 5 (2 hrs) — Paper + contract
Read Karpathy's *Software 2.0*, write a one-page summary in `papers/week-00/`. Then, in the app: **"Let's do the Stage 0 contract session."**

### Saturday (4 hrs) — Module 1.1 begins
Functions, graphs, and change. Plot things. Build `numerical_derivative`. Watch a tangent line tilt.

### Sunday (2 hrs) — First retro
Say **"Sunday retro."** Set week two.

---

## 9. When it gets hard

Around month three — somewhere in module 1.5 or 1.7 — this stops feeling like progress. The novelty is gone, the math is genuinely hard, and you haven't built anything that looks like AI yet.

That feeling is not information about your ability. It's the standard shape of the curve, and it's exactly where people quit and restart with a different course, resetting to zero.

Three things, all concrete:

1. **Re-read your week-one journal entry.** You won't believe you didn't know what a dot product was.
2. **Shrink, don't stop.** Minimum Viable Week. Three hours.
3. **Tell me.** Don't go quiet for a fortnight and come back apologising. Say "I'm in the valley." I'll cut scope, not standards.

Then module 1.7 arrives, you build a working autograd engine in 150 lines, and it clicks.

---

> One repo. Two hours a day. Five gates. Update the file.
>
> That's the whole system.
