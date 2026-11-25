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
