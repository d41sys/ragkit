# BEIR protocol: model-free ragkit vs published numbers (frozen before scoring)

Goal: measure ragkit's model-free retrievers on public BEIR test sets and put them next to
published numbers produced with the same dataset, split and metric. No model is downloaded or
loaded; everything is numpy and the standard library. This file, `src/ragkit/analysis.py` and
the `BM25Index` / `BM25FIndex` / `LexicalReranker` analyzer parameters are committed before any
BEIR query is scored. Numbers come from `ragkit beir` and are reported whatever they are.

## Data

- Datasets: `scifact`, `nfcorpus`, `fiqa`, `arguana`, `trec-covid`, `quora` (BEIR v1.0.0 zips from
  `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/<name>.zip`, read in place
  from `data/raw/beir/`, gitignored). SCIDOCS (142 MB zip) is left out for the disk budget.
- Split: `qrels/test.tsv`, the official test queries. A document is relevant when its qrels score
  is greater than 0 (trec_eval `-l 1`); graded scores (NFCorpus 1-2, TREC-COVID 0-2) are kept
  for nDCG.
- Corpus unit: one BEIR document = one retrieval unit (no chunking).
- Queries scored: every test query with at least one relevant document. A query with an empty
  result list scores 0 (trec_eval `-c`, as in the Pyserini commands).
- Self hits: a document whose id equals the query id is removed from the ranking before scoring
  (Pyserini `--remove-query`; matters for ArguAna and Quora).

## Metrics (trec_eval conventions, as used by the Pyserini 2CR BEIR table)

- **nDCG@10**: `ndcg_cut.10`, linear gain `rel`, discount `1 / log2(rank + 1)`, ideal list = all
  judged documents with `rel > 0` sorted by `rel`. `ragkit.eval.ndcg_at_k` uses `2^rel - 1`; it is
  not used here because it differs from trec_eval on graded sets.
- **Recall@100**: relevant documents in the top 100 divided by all relevant documents.
- Per-query scores are averaged; 95 % percentile-bootstrap CI over queries (2000 resamples,
  seed 0). Ties are broken by corpus order (Lucene breaks them by internal docid).

## Systems (frozen)

Document text: `title + " " + text` ("flat", as in the Anserini flat index) unless a field
split is named.

| id | definition |
|---|---|
| `bm25-ragkit` | `BM25Index(k1=1.5, b=0.75)`, default `tokenize` (`[a-z0-9]+`, no stemming, no stop words), flat text. The shipped default. |
| `bm25-lucene` | `BM25Index(k1=0.9, b=0.4)` with `lucene_english` (UAX#29-like tokens, possessive filter, Lucene's 33 English stop words, Porter stemmer), flat text. Replica of Anserini/Pyserini `bm25-flat`. |
| `bm25-mf-lucene` | Replica of Pyserini `bm25-multifield` (`--fields contents=1.0 title=1.0`): two `BM25Index` (k1 = 0.9, b = 0.4, `lucene_english`), one on the text, one on the title, scores added. Per-field idf and average length use only documents whose field is non-empty (Lucene's field `docCount`). Only for datasets with titles. |
| `bm25f-ragkit` | `BM25FIndex()` defaults (k1 = 1.5, b = 0.75, title weight 2, body weight 1, `tokenize`); title and text as fields. Only for datasets with titles. |
| `bm25f-lucene` | `BM25FIndex(k1=0.9, b=0.4, title_weight=2, body_weight=1, analyzer=lucene_english)`. Only for datasets with titles. |
| `rerank-ragkit` | `LexicalReranker()` defaults (weights 0.4 / 0.3 / 0.3, k1 = 1.5, b = 0.75, `tokenize`) over the top 100 of `bm25-ragkit`. |
| `rerank-lucene` | `LexicalReranker(k1=0.9, b=0.4, analyzer=lucene_english)` (same weights) over the top 100 of `bm25-lucene`. |

Reranking reorders the same 100 candidates, so Recall@100 equals the first stage's by
construction; only nDCG@10 can change. Datasets without titles (FiQA, Quora; check
`corpus.jsonl`) have no multifield / BM25F rows: the title field is empty, so Pyserini's
multifield equals its flat run there.

No parameter above is chosen by looking at test qrels: `k1 = 0.9, b = 0.4` are Anserini's
published defaults, title weight 2 is ragkit's pre-existing default, the reranker weights are the
shipped defaults.

## BM25 sanity check against Pyserini

Reference: the BM25 Flat and BM25 MF columns of the Pyserini BEIR 2CR table
(`https://castorini.github.io/pyserini/2cr/beir.html`). Criterion: `bm25-lucene` and
`bm25-mf-lucene` agree with the Pyserini value within 0.01 nDCG@10 on each dataset.

If any dataset exceeds 0.01, the diagnostic ladder is run (it is also run when everything
agrees, as documentation). Each row changes one thing relative to the previous one, flat text:

| id | analyzer | k1, b |
|---|---|---|
| L0 | `tokenize` | 1.5, 0.75 (= `bm25-ragkit`) |
| L1 | Lucene tokens + possessive filter (no stop words, no stemming) | 1.5, 0.75 |
| L2 | + Lucene stop words | 1.5, 0.75 |
| L3 | + Porter stemmer (= `lucene_english`) | 1.5, 0.75 |
| L4 | same | 0.9, 0.4 (= `bm25-lucene`) |
| L5 | same | 0.9, 0.4, Lucene one-byte document-length norm (`lucene_length`) |

Sources of remaining difference that the ladder cannot remove and that are reported as
unexplained, not tuned away: our token regex is an approximation of Lucene's UAX#29 tokenizer;
Lucene breaks score ties by internal docid; Anserini's published index may differ from the
BEIR zip text in small ways (it is built from the same corpus files).

## Published numbers admitted to the comparison

A published number is placed in the head-to-head table only if dataset (BEIR v1.0.0, same
dataset name), split (test), metric (nDCG@10 or Recall@100) and cutoff are the same as ours and
a link, table and date were opened. Anything else (different split, different metric such as
MTEB-average, different corpus version, candidate depth unstated) goes to a separate
"not comparable" table with the reason. Sources and their dates are in
`docs/beir-published-numbers.md`.

## What was known before scoring

- The Pyserini numbers above (read from the page while writing this protocol), for the datasets
  the page lists.
- Nothing about ragkit's own BEIR scores: no BEIR query had been run through any ragkit
  retriever when this file was written. The code was only exercised by unit tests on
  synthetic fixtures.
- Expectation, stated as such and not as a result: `bm25-lucene` and `bm25-mf-lucene` land
  within 0.01 of Pyserini; `bm25-ragkit` loses on datasets where stemming and the stop list help
  (long natural-language queries) and may win or tie where they do not; the lexical reranker does
  not beat a BM25 first stage, as on the EU corpus.
