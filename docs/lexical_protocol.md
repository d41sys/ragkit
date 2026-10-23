# Lexical retrieval protocol (frozen before scoring)

Goal: improve retrieval on the EU corpus (`--corpus eu`, 94 articles, 42 answerable questions of
type `direct` / `paraphrase` / `multi-hop`) **without any model**. Everything below is numpy and
standard library only. This file and `src/ragkit/expand.py` are committed before any new
configuration is scored on any question; the numbers are produced by `ragkit lexical` and
reported whatever they turn out to be.

## What was already known before this protocol

- The existing `ragkit eval --corpus eu --strategies bm25` table: BM25 recall@5 0.857, MRR 0.676,
  nDCG@10 0.702; by type R@5 / MRR: direct 1.000 / 0.828, multi-hop 0.750 / 0.938,
  paraphrase 0.765 / 0.400.
- The text of two questions (the first two lines of `qa_eu.jsonl`, printed by a `head` while
  inspecting the file format): one on the minimum share of capacity TSOs must make available for
  cross-zonal trade (Regulation 2019/943 Art. 16), one on non-market-based redispatching (Art. 13).
  No other question text was read.
- The corpus articles (heading structure, defined terms, acronym census).
- The label reviewers (README, "Review status") checked relevant-article labels against BM25 and
  dense top-10 lists. This can favour BM25-style retrieval and is a caveat for every result.

## Configurations (all frozen)

Ranking unit is the **document** (article): chunks are scored, the chunk list is deduplicated to
documents by first occurrence, and metrics use the top 10 documents. Relevance is by
`relevant_doc_ids`, exactly as in `ragkit eval`.

| id | name | definition |
|---|---|---|
| A0 | `bm25-pinned` | The existing `BM25Index()` (k1 = 1.5, b = 0.75) on `recursive` chunks (500 chars, 50 overlap). Top 10 **chunks**, then deduplicated. Identical to the BM25 / None row of `ragkit eval`. Baseline of every paired difference. |
| A1 | `+doc-depth` | A0 with all positive-score chunks deduplicated to documents before the top-10 cut. Control: later rows rank documents from the full chunk list, so this isolates that change. |
| A2 | `+structure` | `structured` chunker (below) with the article heading prepended to every chunk, plain `BM25Index` (k1 = 1.5, b = 0.75). |
| A3 | `+bm25f` | `structured` chunks without prepended heading, `BM25FIndex` with the heading as a `title` field, title weight 2.0, body weight 1.0, k1 = 1.5, b = 0.75 (one b for both fields). |
| A4 | `+expansion` | A3 plus query expansion with `SYNONYM_GROUPS`, expansion weight 0.4. **Headline configuration.** |
| A5 | `+cv-params` | A4 with (k1, b, title weight) chosen by cross-validation (below). Secondary: does tuning add anything over the frozen values? |

**Structured chunker** (`ragkit.chunking.structured`): `min_chars = 250`, `max_chars = 1000`.
The article heading is the leading `# ` line. The body is cut into paragraph units: a unit starts
at a block beginning `N.` or `(N)`, and `(a)`-style points stay with their paragraph. Units shorter
than `min_chars` are merged forward; a chunk is closed as soon as it reaches `min_chars` or the next
unit would exceed `max_chars`. A unit longer than `max_chars` is split at point boundaries, and a
single block longer than `max_chars` falls back to `recursive`. `start`/`end` are exact spans of
the body in the source file; the heading is not part of the span.

**BM25F** (`ragkit.bm25f.BM25FIndex`): per-field length-normalised term frequency
`tf~ = sum_f w_f * tf_f / (1 - b + b * len_f / avglen_f)`, score `idf * (k1 + 1) * tf~ / (k1 + tf~)`
with the same Robertson/Sparck Jones idf as `BM25Index` (df counts chunks where the term is in a
field with nonzero weight). With title weight 0 it reproduces `BM25Index` scores exactly (tested).

**Expansion** (`ragkit.expand`): query-side only. When a member of a synonym group occurs in the
query, the tokens of the other members are added once each at weight 0.4; original tokens keep
weight 1.0 and their multiplicity. The map is in `src/ragkit/expand.py`.

### How the synonym map was kept out of the questions

1. It is written from the corpus and from domain knowledge. The sources were the acronym census of
   the 94 articles (`TSO` 256 times, `FRR` 141, `LFC` 129, `FCR` 73, `BSP` 16, ...), the defined
   terms of the acts, and plain-English renderings of legal terms. The corpus was read, the
   question file was not (apart from the two questions listed above, which exercise `TSO`,
   cross-zonal capacity and redispatching: terms that are in the task description itself).
2. Every group must contain at least one phrase that occurs in the corpus
   (`tests/test_ragkit_expand.py`), so an expansion always has a target.
3. The file is committed, with this protocol, before the first configuration is scored; `git log`
   shows the order. It is not edited after scoring.
4. Nothing was counted on the questions: no check of how many map entries fire on them.

## Parameters and leakage control

All parameters of A0 to A4 are frozen here (values in the table above); none was tuned on the 42
scored questions. The only data-driven parameter choice is A5, evaluated by stratified 2-fold
cross-validation:

- Folds: the 42 questions are stratified by `type`, shuffled within type with
  `numpy.random.default_rng(seed)` and alternated into two folds. Repeats: seeds 0 to 4.
- Grid: k1 in {0.9, 1.2, 1.5, 2.0}, b in {0.5, 0.75, 0.9}, title weight in {1, 2, 4};
  expansion weight 0.4 and chunker settings stay frozen. The default (1.5, 0.75, 2) is evaluated first and a
  later grid point replaces it only with a strictly higher training-fold MRR.
- Per fold: choose the grid point with the highest MRR on the training fold, apply it to the
  held-out fold. A5's per-question score is the mean of the held-out scores over the five repeats.

## Metrics and statistics

Recall@5, MRR and nDCG@10 per question (functions in `ragkit.eval`), overall and by type, mean with
a 95% percentile bootstrap CI over questions (2000 resamples, seed 0). Paired bootstrap difference
against A0 for every other configuration, overall and by type (same resampling). The synthetic
corpus (`--corpus synthetic`, 40 questions) is run through the identical configurations as a
regression check.

## Decision rule

A4 is recommended as the model-free lexical retrieval path iff **all** of:

1. EU: paired dMRR (A4 minus A0) has a 95% CI whose lower bound is above 0;
2. EU: paired dRecall@5 point estimate is at least 0;
3. synthetic: dMRR point estimate is at least -0.02 and dRecall@5 at least -0.025 (one question
   in 40).

If any condition fails, A4 is reported as not adopted. Components (A1 to A3, A5) are reported with
the same intervals but cannot rescue or sink A4; no configuration is dropped or re-parameterised
after scoring. With 42 questions the intervals are wide, and about 20 paired intervals are shown,
so an isolated interval that excludes 0 is not strong evidence; the decision rule uses one
pre-specified comparison.

## Scope

Dense, hybrid, rerank and HyDE rows are **not re-run**. `ragkit lexical` builds no dense index and
loads no model (tested: `sentence_transformers`, `torch` and `transformers` are not imported).
