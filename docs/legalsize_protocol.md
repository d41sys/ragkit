# LegalBench-RAG-mini: size-matched chunk control (equal k vs equal character budget), frozen before scoring

Date: 2026-11-06. **No API call, no key, no download.** Only the already cached
`intfloat/e5-small-v2` (revision `ffb93f3b`) is loaded, offline (`HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1`, `uv run --offline`), sequentially, one model at a time, to embed the
documents of two new chunkings (a timing probe on 640 chunks, run before this file, measured about
940 chunks/s for 250-character chunks and 240 chunks/s for 1000-character chunks on Apple MPS: about
75 s in all for 43.5k chunks, so the dense retriever is included) and the queries. This file is
committed before any code for this study exists and before any number of it is computed. The numbers
come from `ragkit legalsize dev|test` and are reported whatever they are.

## Problem

The LegalBench-RAG metric (`docs/legalbench_protocol.md`) takes k as a number of **chunks** and
sums, over the retrieved chunks, the ground-truth characters they overlap, without merging or
de-duplicating (`docs/legalbench-rag-published-numbers.md`; a recall above 1.0 is possible, as
`tests/test_ragkit_legalbench.py::test_overlapping_retrieved_chunks_are_double_counted_like_the_official_code`
shows). A larger chunk size therefore retrieves more characters at the same k, and with overlapping
chunkers (published rows with 15 % overlap or sliding windows) the same characters can count more
than once. Comparisons across chunkers at equal k (`naive500` vs a structure-aware chunker, ragkit
vs StrucChunk R@8 48.18) may thus partly reflect the character budget, not retrieval quality. The
control below measures that part on ragkit's own retrievers.

## Data, sample, split

- Same sample, corpus, metric and k grid as `docs/legalbench_protocol.md`: 776 queries, 72 documents
  pooled, character-overlap Recall@k and Precision@k, k in {1, 2, 4, 8, 16, 32, 64}; ALL =
  equal-weight mean of the four source means; the query is the full query string, unrouted.
- **Dev / test split**: the existing one of `docs/legalsplit_protocol.md` (`ragkit.legalsplit.
  dev_test_split`: per source, `random.Random(20261106)` shuffle, first `round(0.2 * 194) = 39`
  queries dev): dev = 156, test = 620. The test queries are scored once.
- **What dev is for.** Nothing is tuned: chunk sizes, budgets, the primary contrast, retriever
  settings and the rule are fixed in this file, a priori. The dev run is a pipeline pilot (does the
  harness reproduce the committed `naive500` rows, are the budget k values valid, what are the
  magnitudes); its report is committed before the test run and cannot change the rule below. Dev and
  test share the 72 documents (the split is by query).

## Chunkings and retrievers

Chunkings (the paper's "Naive" chunker with a different size, no overlap):
`naive250`, `naive500`, `naive1000` = `ragkit.chunking.fixed(size=s, overlap=0)`, i.e.
`content[i:i+s]` for i in steps of `s` (34,766 / 17,403 / 8,723 chunks; all of size `s` except the
last chunk of each document). Non-overlapping, so the metric cannot double count here: any
size effect below is a budget effect of retrieving more characters per hit, not a double-counting
artefact. (Overlapping chunkers are not part of this control; see Limitations.)

| id | definition |
|---|---|
| `bm25` | `BM25Index(k1=0.9, b=0.4)` with `lucene_english` (the Lucene-matched row: 13.37 on `naive500`), positive scores only, at most 64 hits. Same settings for every size (not re-tuned per size). |
| `dense-e5` | cosine of unit-norm `e5-small-v2` vectors (`query: ` / `passage: ` prefixes, max length 512 tokens; 1000 characters is about 250 tokens, nothing is truncated in practice); document matrices from `data/raw/emb/` or, when absent (`naive250`, `naive1000`), computed once offline and cached there. |

Six cells = {bm25, dense-e5} x {naive250, naive500, naive1000}. Harness check on all 776 queries:
`bm25-naive500` and `dense-e5-naive500` must reproduce the committed 13.37 and 19.69 (Recall@8 ALL)
to within 0.05 points, else this is reported (the comparison is within-harness either way).

## Two ways to line cells up

- **Equal k**: Recall@k ALL at the same k (the published convention).
- **Equal character budget B**: Recall@(B / size) ALL, i.e. the same nominal number of retrieved
  characters. B in {1000, 2000, 4000, 8000, 16000}; every B / size is in the k grid {1, ..., 64}
  for the three sizes (e.g. B = 4000: k = 16 / 8 / 4 for 250 / 500 / 1000). Precision at the same
  points is reported. The nominal budget equals the retrieved characters except for the last chunk
  of a document; the number of chunks shorter than `size` is printed.

## The contrast, the rule and the size-alone share (pre-registered)

Primary cell and contrast: **`bm25`, test split, `naive1000` vs `naive500`, ALL Recall at B = 4000
characters** (`naive1000` at k = 4 minus `naive500` at k = 8; 4000 = the budget of the headline R@8
of `naive500`). `D_B` = that difference (points). `D_8` = Recall@8 `naive1000` minus Recall@8
`naive500` (equal k, same cells). Each difference gets a 95 % percentile-bootstrap CI: 2000
resamples, seed 0, queries of each source resampled with replacement (the same resample for the two
cells and for `D_8` and `D_B`), four source means averaged with equal weight (as
`ragkit.legalsplit.paired_all_ci`).

> **Rule.** If the CI of `D_8` is above 0 (a gain at equal k exists to be explained) and the CI of
> `D_B` includes 0, report: **"chunk-size gains on this metric are mostly a budget effect"**. If the
> CI of `D_B` excludes 0, do not say that: report the size effect at equal budget with its sign and
> CI (above 0: larger chunks help beyond their budget; below 0: smaller chunks win per retrieved
> character). If the CI of `D_8` includes 0 there is no equal-k gain to explain, and that is
> reported. The rule is applied to the primary cell only. The same rule is applied, and reported
> separately, to the other cells (`dense-e5` for 1000 vs 500, at B = 4000 like the primary; 500 vs
> 250 at B = 4000 and 1000 vs 250 at B = 8000, for both retrievers), and none of these changes the
> verdict above. (Wording fixed before any code or number existed: the first draft said "the
> larger size's k = 8" for these budgets, which holds for 500 vs 250 and 1000 vs 250 but not for
> the primary 1000 vs 500; the budgets written here are the ones used. In every contrast `D_8` is
> the equal-k difference at k = 8.)

Share of an equal-k difference explained by size alone (descriptive, reported for every contrast
above): `S = 1 - D_B / D_8`, with a paired bootstrap CI (same resamples); `S` is undefined (and
printed as such) when the bootstrap `D_8` is not above 0 in every resample. `S = 1`: the whole
equal-k difference disappears at equal budget; `S = 0`: it is unchanged; `S > 1`: the order reverses
at equal budget. "Explained by chunk size alone" in the reports means exactly this `S` (a statement
about the metric on this benchmark, not a causal model of retrieval quality).

Also printed, descriptive: Recall@k and Precision@k ALL at every k and every B for all six cells, the
per-source Recall at B = 4000, number of chunks and of short chunks. Anything computed on all 776
queries (dev included) is labelled "post-hoc, includes dev" and exists only to check reproduction of
the committed `naive500` rows.

## Predictions (stated before any number)

- P1: Recall@k at equal k rises with chunk size for both retrievers (twice the characters per hit).
- P2: at equal B, the gap between sizes shrinks a lot for `bm25` (the larger chunk dilutes the
  matching terms, but gold spans are long, so one hit covers many characters either way); a CI
  that includes 0 for `D_B` is plausible but not predicted with confidence.
- P3: `dense-e5` keeps more of the size effect than `bm25` (a larger chunk embeds more content
  into the same vector), so `S` is lower for `dense-e5` than for `bm25`.
- Not predicted: the sign of `D_B` if its CI excludes 0.

## Comparison rules (fixed now, whatever the outcome)

The size effect measured here is for ragkit's own retrievers on `naive` chunkings. It is a
calibration of how much R@k moves with the character budget on this metric, not an adjustment of
the published rows: the published systems' chunk sizes (StrucChunk: stated inconsistently) and
retrievers differ, so a published R@8 cannot be converted into a size-matched number from this
data. The docs state the caveat next to every comparison with published rows, give the measured
size sensitivity as the yardstick, and keep the existing headline rows and the default of
`ragkit legalbench` unchanged.

## Limitations stated in advance

- One benchmark, one metric, two retrievers; `bm25` is not re-tuned per size (k1 and b may favour
  one size).
- Non-overlapping chunks only: the double-counting part of the metric (overlapping chunkers) is
  argued from the metric's definition and the unit test, not measured here.
- Equal budget in nominal characters; a retriever that returns shorter lists (BM25 returns only
  positive-score chunks) is counted as it is.
- Split by query: dev and test share documents.
