# LegalBench-RAG-mini: query-prefix removal (arm A) and document routing (arm B), frozen before scoring

Date: 2026-11-06. **No API call, no key, no download.** Only the already cached
`intfloat/e5-small-v2` (revision `ffb93f3b`) is loaded, offline (`HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1`, `uv run --offline`), and only to embed queries; the document matrices come
from the existing cache `data/raw/emb/`. This file is committed before any code for this study
exists and before any number is computed. The numbers come from `ragkit legalsplit tune|score` and
are reported whatever they are.

## Problem

On LegalBench-RAG-mini the model-free and small-encoder retrievers are far below the published
rows (Recall@8 ALL: BM25 13.37, `dense-e5` + `recursive500` 20.40, best comparable published 48.18;
`docs/legalbench_protocol.md`, `docs/legalbench-rag-published-numbers.md`). Every query is
`Consider the <contract type> between <A> and <B>; <question>` (privacy_qa and some contractnli
queries use `Consider "<Name>"'s privacy policy; <question>` or `Consider <Name>'s <type>;
<question>`). Checked while writing this file, on the 776 sampled queries: every query contains
exactly one `;`, every one starts with `Consider`, and the part before the `;` takes exactly one
value per gold document (distinct prefixes: privacy_qa 7, contractnli 18, maud 18, cuad 29 = the
number of gold documents) and every query has gold snippets in one file only. Hypothesis: the
shared prefix pulls retrieval towards chunks of the wrong documents.

## Data, sample, split

- Same sample, corpus, metric and chunkers as `docs/legalbench_protocol.md`: 776 queries, 72
  documents pooled, character-overlap Recall@k and Precision@k, k in {1, 2, 4, 8, 16, 32, 64},
  chunkers `naive500` and `recursive500`. ALL = equal-weight mean of the four source means.
- **Dev / test split**: per source, the queries in benchmark order are shuffled with
  `random.Random(20261106)` (one generator, sources in the order privacy_qa, contractnli, maud,
  cuad) and the first `round(0.2 * 194) = 39` become dev; the other 155 are test. Dev = 156
  queries, test = 620. Every design choice below is made on dev only; the test queries are scored
  once, with the frozen choices. The split is by query, so dev and test share documents (each
  document has several queries): the dev-chosen choices are not document-specific, but the
  shared documents make the test split not independent of dev in that sense.

## Retrievers and baselines (same chunker and same engine as the existing rows)

| id | definition |
|---|---|
| `bm25` | `BM25Index(k1=0.9, b=0.4)` with `lucene_english` (the Lucene-matched row, 13.37 on `naive500`); positive scores only, at most 64 hits. |
| `dense-e5` | cosine of unit-norm `e5-small-v2` vectors (`query: ` / `passage: ` prefixes, max length 512), document matrices read from the existing cache; a cache miss is an error (no corpus is ever re-embedded). |

Each retriever is run on both chunkers, giving four cells: `bm25-naive500`, `bm25-recursive500`,
`dense-e5-naive500`, `dense-e5-recursive500`. **Baseline** of a cell: the full query, all 72
documents, no change. As a harness check the four baselines are also computed on all 776 queries;
they must reproduce the committed rows (13.37, 11.84, 19.69, 20.40 at two decimals) to within 0.05
points; if one does not, that is reported and the recomputed value is the baseline (the arms are
paired against it either way).

## Arms

**Split rule.** `prefix, question = query.split(";", 1)`; `question` is stripped; `prefix` is
stripped and loses a leading `Consider`. A query with no `;` is kept whole (prefix empty). The
check above says this never happens on the 776 queries; the number is printed.

**Arm A (question only, all documents).** The retriever runs on `question` over all 72 documents.
Nothing is tuned for A. A changes the input of the official run (which is the full query string),
so it is not a like-for-like row against the published numbers (see Comparison rules).

**Arm B (uses document metadata; not comparable to the published rows).**
1. *Route*: rank the 72 documents by matching `prefix` against a document profile. The profile is
   one of: `name` (file name without directory and `.txt`), `name+open500` (name plus the first
   500 characters of the text), `name+open2000`, `open2000` (the first 2000 characters only).
   Matching: `BM25Index(k1=0.9, b=0.4)` with `lucene_english` over the 72 profiles (one pseudo
   chunk each), ranking by score, positive scores only, ties to the lower index of the sorted
   path list. The router has no access to chunk text beyond the opening, to the gold files or to
   the source label of a query.
2. *Search*: keep the top `n` routed documents and run the retriever over the chunks of those
   documents only (same chunks, same scores; chunks of other documents are removed). The search
   text is the full query or `question` (the "in-document query"). BM25 still returns at most 64
   positive-score chunks, dense the 64 best.
3. *Fallback*: if no document has a positive routing score, B searches all 72 documents with the
   full query (counted and printed).

B is labelled everywhere "uses document metadata, not comparable to the published rows": the
published systems get the whole query and a corpus of unlabelled chunks.

## Design choices, on dev only

1. **Profile** (retriever-independent): maximise the dev document top-1 accuracy (mean of the four
   source accuracies; gold document = the single file of the query's truth); ties go to the
   profile earlier in the order `name`, `name+open500`, `name+open2000`, `open2000`. Document
   top-2 accuracy is printed, not used.
2. **`n` and in-document query** (per retriever, not per chunker): with the chosen profile, `n` in
   {1, 2} and in-document query in {full, question}: maximise the mean over the two chunkers of
   dev Recall@8 ALL; ties go to `n = 1`, then to `question`.

Arm A has no free choice. The chosen settings are written to `docs/sota-runs/legalsplit_tuning.json`
(committed before any test query is scored) and `ragkit legalsplit score` reads them.

## Scoring and the rule (pre-registered)

Test split only (620 queries). Primary measure: Recall@8 ALL. For each cell and arm the paired
difference `arm - baseline` in Recall@8 ALL gets a 95 % percentile-bootstrap CI: 2000 resamples,
seed 0, the queries of each source resampled with replacement (the same resample for the arm and
the baseline), the four source means averaged with equal weight, as in
`LegalResult.all_ci`.

> **Rule.** Arm A is reported as an *improvement* in a cell if the lower bound of the paired
> bootstrap CI of the Recall@8 ALL difference (arm A minus the same-retriever, same-chunker
> baseline, test split) is above 0; as *worse* if the upper bound is below 0; otherwise as
> *no detectable effect*. Arm B is judged by the same rule, separately from A. The two primary
> cells are `bm25-naive500` (the 13.37 headline row) and `dense-e5-recursive500` (the 20.40
> headline row). An arm is an improvement *overall* only if it is an improvement in both
> primary cells; if it is an improvement in one only, it is reported as an improvement in that cell
> only. The other two cells are reported with the same rule and do not change the verdict.

Consequences, fixed now: whatever the outcome, the official full-query rows and the default of
`ragkit legalbench` stay unchanged and stay the headline rows. A and B are added as separately
labelled rows (opt-in command `ragkit legalsplit`); a negative or null result is reported as such.

Also reported, descriptive (not part of the rule): Recall@k at every k, Precision@8, per-source
Recall@8, the document routing accuracy (top-1, top-2) per source on test, the number of fallbacks.
Anything computed on all 776 queries (dev included) is labelled "post-hoc, includes dev"; it is
there only to line the rows up with the published full-sample numbers.

## Predictions (stated before any number)

- P1: A improves `bm25` in both chunkers; the shared words of the prefix (`agreement`, `between`,
  company names) score against chunks of every document, and without them the question terms
  dominate. A smaller gain for `dense-e5`.
- P2: routing by profile finds the gold document at top-1 for most queries on privacy_qa,
  contractnli and maud (file names carry the party names) and less often on cuad (upper-case,
  run-together file names; the opening text helps).
- P3: B beats A and the baselines clearly, because it removes the wrong-document errors outright;
  its gain is bounded by the in-document ranking, so MAUD stays low.
- Not predicted: whether B can reach the published 48.18 (it is not comparable anyway).

## Comparison rules

The published rows (paper Tables 4, 5, 7; StrucChunk) take the full query as the input and an
unlabelled pool of chunks. Arm A removes the document-naming prefix from the query, which the
official run does not do: A rows are listed as "same sample, metric and k, different query input:
not a like-for-like row". Arm B additionally uses the document identity named in the query, which
makes it a document-routed (oracle-like) setting: it is listed with the oracle-document systems in
`docs/legalbench-rag-published-numbers.md` and is not compared with the published rows above. The
fair-comparison table in the README keeps the full-query rows; the best comparable published
number stays 48.18.

## Limitations stated in advance

- One benchmark, one fixed template: the effect of removing the prefix says nothing about queries
  that do not carry a document-naming prefix.
- A split by query leaves dev and test with the same 72 documents.
- Routing depends on the prefix naming the document: that is a property of this benchmark's
  templates, not of legal retrieval in general.

## Outcome (scored once, no deviation from the protocol above)

Commits: protocol `d38195d`, code and tests `c8a6b56`, dev tuning `b98fdb7`, test scoring
`d04f8ab`. Commands (offline, no key, no download; the e5 document matrices came from
`data/raw/emb/`, only queries were embedded):

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline ragkit legalsplit tune  --out docs/sota-runs/legalsplit_tune.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline ragkit legalsplit score --out docs/sota-runs/legalsplit_score.txt
```

Raw output: `docs/sota-runs/legalsplit_tune.txt`, `docs/sota-runs/legalsplit_tuning.json`,
`docs/sota-runs/legalsplit_score.txt`. All four full-query baselines on the 776 queries reproduce
the committed rows (gap 0.00 in every cell).

**Dev choices** (156 dev queries): profile `name+open2000` (document top-1 on dev: `name` 56.4 %,
`name+open500` 90.4 %, `name+open2000` 94.2 %, `open2000` 89.1 %); for both retrievers `n = 1` and
the question-only in-document query (dev Recall@8 ALL, mean of the chunkers: `bm25` 45.65 against
27.58 with the full query at `n = 1` and 37.32 / 21.52 at `n = 2`; `dense-e5` 51.33 against 34.45,
42.09 and 31.45).

**Test, Recall@8 ALL in % (620 queries), paired difference to the same-cell baseline, 95 % CI:**

| cell | baseline | A: question only | A minus baseline | B: routed, `n = 1` | B minus baseline |
|---|---|---|---|---|---|
| `bm25-naive500` (primary) | 12.95 | 6.92 | -6.03 [-8.11, -3.90] | 47.24 | +34.29 [+31.10, +37.48] |
| `dense-e5-recursive500` (primary) | 20.70 | 12.65 | -8.06 [-10.94, -5.00] | 54.39 | +33.68 [+30.17, +37.24] |
| `bm25-recursive500` | 11.08 | 7.18 | -3.90 [-5.97, -1.75] | 47.10 | +36.02 [+32.59, +39.32] |
| `dense-e5-naive500` | 20.09 | 10.30 | -9.80 [-12.65, -6.79] | 52.96 | +32.86 [+29.50, +36.37] |

**Verdict under the pre-registered rule.** Arm A: *worse* in every cell (upper CI bounds below 0),
so it is not an improvement and the hypothesis that removing the prefix helps is rejected on this
benchmark. Arm B: *improvement* in both primary cells and in the other two (lower bounds above
+29). Routing finds the gold document at top-1 for 95.6 % of the test queries (privacy_qa 94.2,
contractnli 88.4, maud 100.0, cuad 100.0), with no fallback. B stays lowest on MAUD (21 to 30 %
Recall@8) against 60 to 71 % on CUAD and 66 to 67 % on ContractNLI.

**Predictions.** P1 (A improves BM25) is false: A lowers Recall@8 in all four cells, BM25 included.
P2 holds for privacy_qa, contractnli and maud, but not for CUAD: the dev-chosen profile with the
opening text routes CUAD perfectly (the run-together upper-case file names alone were not enough:
`name` has 56.4 % top-1 on dev). P3 holds (B beats A and the baselines, MAUD stays the lowest).

**Reading** (what the numbers support, and what they do not):
- The wrong-document error is large: once the right document is known (B), Recall@8 rises by 33 to
  36 points in every cell.
- Dropping the prefix does not remove that error. In B, on dev, the question-only query beats the
  full query by 17 to 18 points inside a single document, so the prefix words hurt the choice of
  chunk; in A, over all documents, the same removal costs 4 to 10 points on test. [INFERENCE, not
  tested here] The prefix is the only place that names the document, and the party names and
  contract type recur in the contract text, so without it nothing steers the search towards the
  right contract.
- B is a different task from the published one (the system is told which document to search, by
  the query itself). Its numbers are listed apart from the published rows and not compared with
  them; the best comparable published number stays 48.18.
- Anything beyond the above (why MAUD stays low, whether an oracle router would add more than the
  4.4 % of routing misses) was not run.

The post-hoc 776-query table in `legalsplit_score.txt` includes the 156 dev queries; it is only a
line-up with the full-sample rows (A 7.02, 10.35, 7.25, 12.46 and B 46.98, 52.34, 46.74, 54.06 in
the order `bm25-naive500`, `dense-e5-naive500`, `bm25-recursive500`, `dense-e5-recursive500`).
