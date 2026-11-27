# An 18 000-token cap per paper for comparison questions on long PDFs (P-5), frozen before any new data

P-4 ([`paper_pool_protocol.md`](paper_pool_protocol.md)) adopted a pool of 30 chunks and no cap:
on its dev data a cap gained nothing overall. Looked at by question type after P-4's scoring
(post hoc, on P-1's 50 and P-4's 40 long-PDF questions, both now dev data), a cap of 18 000
estimated tokens per paper trades single-paper questions for comparison questions:

| Evidence recall, pool 30 | named | described | cross |
|---|---|---|---|
| P-1's 50: no cap / cap 18 000 | 0.950 / 0.850 | 0.900 / 0.950 | 0.358 / 0.500 |
| P-4's 40: no cap / cap 18 000 | 1.000 / 0.900 | 1.000 / 1.000 | 0.350 / 0.625 |

Without a cap 1.10 papers are packed per search, with it 1.95 to 1.98. P-5 tests this one
configuration once on new questions. Retrieval and packing only.

## Systems (fixed)

- **S30** = pool 30, no cap, 36 000 budget (the default after P-4).
- **W18** = S30 with `window_tokens=18_000`: a paper longer than 18 000 estimated tokens is packed
  as the window of that size around its best chunk.

## Data

40 new questions on the 20 forecasting PDFs (`src/ragkit/data/qa_papers_forecasting_fresh2.jsonl`),
written after this commit under P-4's rules by Claude subagents that see only the extracted text
and the 90 earlier questions (to avoid repeats), checked by a second pass, no human review: 20
**cross** (2 to 4 papers) and 20 **single** (one per paper, 10 `named`, 10 `described`).
`ragkit papers validate --fresh 2` must pass before the run. Guard data: P-4's QASPER and MDA-QA
test sets (scored once in P-4 for other configurations; no choice was made on them).

## Rules (paired bootstrap over questions, 10 000 resamples, seed 0, 95 % CI of W18 - S30)

1. **Recommended for comparison questions** iff the CI lower bound of evidence recall on the 20
   cross questions is above 0. Then the README names `window_tokens=18_000` as the setting for
   questions that compare papers, and callers that know the question type (the ezcross agent's
   paper tool) may offer it as an option.
2. **Default for paged corpora** iff rule 1 holds, the lower bound on the 20 single questions is
   above -0.05, and the lower bound is above -0.02 on QASPER and on MDA-QA.

Otherwise `window_tokens` stays an undocumented-as-recommended option and nothing changes.

## Predictions (written before any new question exists)

- P1: cross rises by 0.15 to 0.30 (rule 1 holds).
- P2: single questions lose 0.05 to 0.10, so rule 2 fails.
- P3: QASPER and MDA-QA change by less than 0.005.

## Not claimed

Answer quality; other cap sizes or several windows per paper (18 000 x 1 was picked post hoc from
four variants, which is why it is tested here on new questions); papers of other fields.

## Outcome (scored once, `docs/sota-runs/paper_cap_score.txt`)

| Evidence recall (MDA-QA: needed papers packed) | S30 | W18 | W18 - S30 |
|---|---|---|---|
| Long PDFs, 20 cross questions | 0.317 | 0.533 | +0.217 [+0.117, +0.317] (10 better, 0 worse) |
| Long PDFs, 20 single questions | 1.000 | 0.950 | -0.050 [-0.150, +0.000] (0 / 1) |
| QASPER, 1 001 questions | 0.985 | 0.983 | -0.002 [-0.005, +0.000] (0 / 2) |
| MDA-QA, 200 questions | 0.832 | 0.834 | +0.003 [+0.000, +0.007] (1 / 0) |

**Verdict: recommended for comparison questions, not the default.** Rule 1 holds (cross lower
bound +0.117). Rule 2 fails on the single questions: the lower bound is -0.150, the rule needs
above -0.05 (one `described` question of 20 lost its quote, and 20 questions cannot show a loss
smaller than that). The README names `window_tokens=18_000` as the setting for questions that
compare papers; `search()` defaults are unchanged.

- With the cap 1.95 long papers are packed per search instead of 1.15, and 34 612 estimated
  tokens instead of 24 699.
- P1 held (+0.217). P2 held at its edge (-0.050; rule 2 fails as predicted). P3 held.
- Limits: the second pass changed 13 of the 20 cross questions (dropped gold papers, better
  quotes, three replaced properties); cross gold sets can be incomplete; even with the cap half of
  the cross evidence is still not packed (0.533).

## After the outcome: what limits comparison questions (dev data, not a claim)

Hypothesis after a smoke run on three PDFs: one long paper takes all 30 chunks of the pool. On
the 50 cross questions of the three long-PDF sets (all dev data now) this is **not** the limit
(`docs/sota-runs/paper_slots_dev.txt`):

- the top 30 chunks hold 8.8 distinct papers on average and never fewer than 3; 0.930 of the gold
  papers are among them (1.000 in the top 100);
- limiting the pool to 3 chunks per paper changes nothing at equal window size (0.645 against
  0.645 at 9 000, 0.678 against 0.673 at 6 000);
- the limit is budget = papers x window: smaller windows fit more papers and lift cross evidence
  (18 000: 0.563 with 2 papers; 9 000: 0.645 with 4; 4 000: 0.688 with 7.7) while single-paper
  questions fall (0.925, 0.812, 0.625); a 54 000 budget at 18 000 windows gives 0.647 with 3
  papers and leaves single questions at 0.925;
- even with 9 papers packed, cross evidence stays below 0.70: the window around a paper's best
  chunk often misses the passage that answers a comparison question.

No study follows: a pool limit per paper has no effect to test, and the remaining levers are a
larger budget (the caller's choice, `budget_tokens`) or several searches by the caller.
