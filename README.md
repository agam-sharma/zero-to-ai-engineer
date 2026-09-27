# Zero to AI Engineer

My study log for going from senior Salesforce/Agentforce engineer to building AI from first
principles — the math, the architectures, and the systems — over about fifteen months.

Everything here is written by me. Claude acts as mentor: it teaches, reviews and tests,
but it does not write the code in this repo.

![stage](https://img.shields.io/badge/stage-0_setup-blue)
![progress](https://img.shields.io/badge/progress-0%25-lightgrey)
![pace](https://img.shields.io/badge/pace-15--20_hrs%2Fweek-green)

## Progress

| Stage | Hours | Status |
|---|---|---|
| 0 · Setup & Baseline | 20 | 🔨 In progress |
| 1 · Math You Can Run | 190 | ⏳ |
| 2 · Classical ML | 70 | ⏳ |
| 3 · Neural Networks | 140 | ⏳ |
| 4 · Language & Sequences | 110 | ⏳ |
| 5 · The Transformer | 120 | ⏳ |
| 6 · Build Your GPT | 110 | ⏳ |
| 7 · Post-Training & Alignment | 90 | ⏳ |
| 8 · AI Engineering in Production | 110 | ⏳ |
| 9 · Breadth | 60 | ⏳ |
| 10 · Capstone & Launch | 80 | ⏳ |

**Capstones:** ⬜ mini-GPT pretrained from scratch · ⬜ CodeSage (deployed)

Live detail: [`PROGRESS.md`](PROGRESS.md) · Full program: [`curriculum/CURRICULUM.md`](curriculum/CURRICULUM.md)

## The method

Every concept has to clear five gates before I move on:

1. **Draw it** — sketch the shapes from memory
2. **Derive it** — write the equation, including a gradient
3. **Code it** — NumPy only, no autograd
4. **Production it** — PyTorch, batched, asserting equivalence with (3)
5. **Defend it** — explain it and survive the follow-up questions

Four out of five isn't four-fifths learned. It isn't learned.

## Layout

```
curriculum/   the program, the mentor protocol, the operating manual
src/          libraries I build (mymath, micrograd, …)
projects/     the 33 numbered projects
notebooks/    exploration
drills/       from-blank re-implementations of older work
journal/      daily entries and weekly retros
papers/       paper-of-the-week summaries
blog/         Friday write-ups
```

## Selected work

_Links appear here as projects land._
