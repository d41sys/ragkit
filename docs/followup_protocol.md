# Papers that matched but did not fit, and a search inside chosen papers (P-6), frozen before any code or run

## Problem

The ezcross-ems agent, run with an omp model on 20 forecasting PDFs uploaded through the API, said
PatchTST was not in the library after two searches, although it was. Reproduced offline on the same
20 PDFs with the shipped `search()` (pool 30, 36 000 tokens, overflow `stop`):

| Query | Packed | Rank of PatchTST's paper in the pool |
|---|---|---|
| How does PatchTST tokenize a time series? | Jin et al. (Time-LLM), 27 154 tokens | 4 |
| PatchTST results on ETTh1 | Lin et al. (SparseTSF), 17 106 tokens | 2 |

One long paper fills the budget, so the paper the question names never reaches the caller, and
nothing in the result says it matched. P-5 measured the same limit on comparison questions: the top
30 chunks hold 8.8 papers on average and 0.930 of the gold papers, while about 1.15 are packed.

## Change (additive; the packed articles do not change)

1. `SearchResult.also_matched`: the documents that have a chunk in the pool but were not packed,
   in the rank order of their best chunk, each with `id`, `act` (the paper title), `citation` and
   the `pages` of its best chunk. No text.
2. `search(..., documents=[ids])`: packs only those documents, from the top 100 chunks of the same
   ranking restricted to them; a listed document without a chunk there is packed from its start.
   An unknown id raises `ValueError`. `also_matched` is empty then.

A caller (the agent) that sees a paper in `also_matched` can search again inside it.

## Data and measures (retrieval only, no answer is generated)

The 130 long-PDF questions on the 20 forecasting PDFs: P-1's 50 (`qa_papers_forecasting.jsonl`),
P-4's 40 (`_fresh`) and P-5's 40 (`_fresh2`), all dev data now. Default `search()`, `hybrid+ce`.

- **M1, seen papers**: share of gold papers packed or listed in `also_matched`, against packed only.
- **M2, two-step evidence recall**: a mechanical caller that runs the question once, then once more
  with `documents=[p]` for each of the first two papers of `also_matched`; evidence recall over the
  union of the packed texts, against one search. Reported per type (named, described, cross).
- **Check**: for every question the default call returns the same articles as before the change.

## Predictions (written before any run)

- P1: M1 is at least 0.90 on cross questions (P-5: 0.930 of gold papers in the top 30).
- P2: M2 raises cross evidence recall by at least 0.15 over one search.
- P3: named and described questions lose nothing (the first search is unchanged; follow-ups only add).

## Not claimed

That the agent follows up (it must be told to, in its tool description, and that is measured only
by a trial); answer quality; other corpora. Two follow-ups cost two more searches.
