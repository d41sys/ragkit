# Chunk pool and per-paper cap for paper corpora (P-4), frozen before any code or run

Two limits of the shipped paper packing (36 000 estimated tokens, whole papers, the papers of the
top 10 chunks) are known and unmeasured as fixes:

1. **Pool.** On MDA-QA only 0.773 of the needed papers have a chunk in the top 10 (0.875 in the
   top 20, 0.950 in the top 50; measured after P-2's scoring). P-3 found 0.37 of the multi-paper
   answers fully correct.
2. **Long PDFs.** A paper with appendices (about 23 000 estimated tokens) fills the budget alone:
   1.10 papers packed on P-1's 20 forecasting PDFs, evidence recall 0.358 on its questions that
   need several papers (measured after P-1, on dev data).

P-4 chooses one packing configuration on data already studied and tests it once on fresh data.
Retrieval and packing only: no answer is generated (P-3 used most of the week's Codex quota).

## Configurations

Budget 36 000, overflow `stop` and the ranking (`hybrid+ce@100`) are fixed. A configuration is
(pool, cap, slices):

- **pool**: the papers of the top 10, 20, 30 or 50 chunks are candidates.
- **cap**: a paper longer than the cap is packed as a window of that size around its best chunk:
  none, 18 000 or 12 000 estimated tokens (so at least two or three papers fit).
- **slices** (with a cap only): one window, or up to three windows around the paper's first three
  chunks in the pool, sharing the cap.

20 configurations: 4 pools x (no cap; 18 000 x 1; 18 000 x 3; 12 000 x 1; 12 000 x 3).
**B36** = (10, none, 1) is the shipped packing.

## Data

| | QASPER | MDA-QA | Long PDFs |
|---|---|---|---|
| Dev (already studied) | P-2's test split, 1 310 questions, 416 papers | P-2's test pool, shuffled ids 100 to 200 | P-1's 50 questions on the 20 forecasting PDFs |
| Test (fresh) | train papers at shuffled positions 416 to 832 (P-3 used 0 to 416), all their questions with text evidence | two pools, shuffled ids 300 to 400 and 400 to 500, each with its own corpus (the dev pool size); 200 questions scored together | 40 new questions on the same 20 PDFs |
| Primary metric | evidence recall | share of needed papers packed | evidence recall |

**The 40 new long-PDF questions** (`src/ragkit/data/qa_papers_forecasting_fresh.jsonl`) are written
after this commit by Claude subagents that see only the extracted paper text and P-1's 50
questions (to avoid repeats), never a retrieval result and never the configurations; a second
subagent pass checks them; no human review. Rules as P-1 (a verbatim evidence quote of 10 to 60
words per gold paper, from the body). 20 **single** questions, one per paper (10 `named`, 10
`described`), and 20 **cross** questions over 2 to 4 papers. `ragkit papers validate --fresh` must
pass before the test run. They are written before the test run and independently of the dev grid.

## Choice on dev

The dev score of a configuration is the mean of its three dev primary metrics. Chosen: **C** = the
simplest configuration within 0.005 of the best dev score, where simpler means fewer settings that
differ from B36 (pool, cap), then the smaller pool, then the order of the cap list above.

## Test (scored once)

Systems: B36, C and, not gating, C's two halves: (C's pool, none, 1) and (10, C's cap, C's
slices). Paired bootstrap over questions (10 000 resamples, seed 0, 95 % CI) of the primary metric,
C - B36, per benchmark. If C = B36 nothing is tested and nothing changes.

**C becomes the default for paged corpora** (`search()` with nothing set, as the 36 000 budget) iff

1. the CI lower bound is above 0 on MDA-QA **or** on the long PDFs (the two targets; two chances,
   no correction, stated here), and
2. no benchmark loses: the CI lower bound is above -0.02 on QASPER and on MDA-QA and above -0.05
   on the long PDFs.

Otherwise the default stays and C's settings remain options (`pack_chunks`, `window_tokens`). The
EU corpus has no page breaks, so its packing cannot change; a test pins that an unpaged corpus
keeps the top 10 chunks and whole articles.

Reported: papers packed, estimated tokens per search, all needed papers packed (MDA-QA), and the
long-PDF result by question type.

## Predictions (written before any run)

- P1: C has a pool above 10 and a cap.
- P2: MDA-QA test rises by 0.03 to 0.10 (the budget, not the pool, limits how many papers fit).
- P3: QASPER test changes by less than 0.01.
- P4: long PDFs: cross questions rise by at least 0.15; single questions lose at most 0.05.

## Not claimed

Answer quality (P-3 found that fully correct answers follow packing gains only in part); the
ezcross agent, which may search several times; papers of other fields or the group's own papers;
overflow `skip` and budgets other than 36 000 are not in the grid.
