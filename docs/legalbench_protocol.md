# LegalBench-RAG-mini protocol: model-free ragkit vs published numbers (frozen before scoring)

Goal: score ragkit's model-free retrievers on LegalBench-RAG-mini with the benchmark authors'
own metric, and compare with the published numbers that use the same sample, metric and k grid.
No model is downloaded or loaded. This file is committed before any LegalBench-RAG query is
run through a ragkit retriever; the numbers come from `ragkit legalbench` and are reported
whatever they are. Published numbers and their comparability verdicts are in
`docs/legalbench-rag-published-numbers.md`.

## Data and sample

- Data: the authors' Dropbox archive (`data/raw/legalbenchrag.zip`, gitignored; 90.6 MB;
  `benchmarks/{privacy_qa,contractnli,maud,cuad}.json`, `corpus/<source>/<file>.txt`), linked
  from https://github.com/zeroentropy-ai/legalbenchrag.
- "Mini" is not a file: it is the runtime sample of `legalbenchrag/benchmark.py`, reproduced
  here exactly: per source, if there are more than 194 tests, sort them by
  `random.Random(test.snippets[0].file_path).random()` (stable sort) and keep the first 194.
  The corpus is the sorted set of files referenced by the kept tests, pooled over all four
  sources (a query is searched against all of them, as in the official run).
- Text is read as Python text mode does (`\r\n` and `\r` become `\n`), which is the only reading
  for which every ground-truth span reproduces its `answer` string (checked while writing this
  protocol: 0 mismatches over 1,307 snippets).
- Acceptance check (paper Table 3): 72 documents (ContractNLI 18, MAUD 18, CUAD 29,
  PrivacyQA 7), 8,682,104 characters, 776 queries (194 per source). The command prints these
  counts. Verified while writing this file: the sample has exactly these counts.

## Metric (official `run_benchmark.py`, character overlap, as read from the repo)

For a query with ground-truth snippets G and retrieved chunks R (each a `(file, start, end)`
half-open span):

- precision = sum over retrieved chunks of the characters overlapping some ground-truth span of
  the same file, divided by the total retrieved characters (0 if nothing was retrieved);
- recall = sum over ground-truth snippets of the characters overlapped by some retrieved chunk,
  divided by the total ground-truth characters.

Retrieved chunks are not merged. k is a number of chunks, k in {1, 2, 4, 8, 16, 32, 64}.
Per source: mean over its 194 queries. ALL: equal-weight mean of the four source means (0.25
each), which on mini equals the mean over all 776 queries. A 95 % percentile-bootstrap CI is
given for ALL (queries resampled within each source, 2000 resamples, seed 0).

## Chunkers

| id | definition |
|---|---|
| `naive500` | `content[i:i+500]` for i in steps of 500 (`ragkit.chunking.fixed(size=500, overlap=0)`): the paper's "Naive" chunker, exactly. |
| `recursive500` | `ragkit.chunking.recursive(chunk_size=500, overlap=0)`: separators `\n\n`, `\n`, `. `, space, character. Not the paper's RCTS (LangChain, other separators), so rows built on it are "partly comparable". |

## Systems (frozen)

| id | definition |
|---|---|
| `bm25-naive500` | `BM25Index(k1=1.5, b=0.75)`, default `tokenize`, on `naive500` chunks of all 72 documents. |
| `bm25-recursive500` | the same on `recursive500` chunks. |
| `bm25-lucene-naive500` | `BM25Index(k1=0.9, b=0.4)` with `lucene_english` on `naive500` chunks. |
| `bm25-lucene-recursive500` | the same on `recursive500` chunks. |
| `rerank-naive500` | `LexicalReranker()` defaults over the top 300 of `bm25-naive500` (the paper's Cohere setting also reranks 300 candidates). |
| `rerank-recursive500` | `LexicalReranker()` defaults over the top 300 of `bm25-recursive500`. |

The query is the full query string, as in the official run (it contains a "Consider ..."
prefix naming the document; the document names are not in the chunk text). A query with fewer
than k matching chunks returns fewer than k chunks (BM25 returns only positive scores; no
padding). The number of such queries at k = 64 is reported.

Nothing is tuned: k1/b and the analyzer are Anserini's published values, chunk size 500 is the
paper's, reranker weights are the shipped defaults.

## Comparison rules

A published number enters the head-to-head only if it is on the same mini sample (the paper's
own Tables 4, 5, 7), with the same metric and k. BM25-only numbers from later papers are listed
as not comparable or unverified, with the reason, in
`docs/legalbench-rag-published-numbers.md`. The paper's Table 4 "ALL" row is not the mean of its
dataset rows (documented there); the head-to-head uses the recomputed mean of the four dataset
rows, labelled as such.

## What was known before scoring

- Published tables of the paper (read while writing the published-numbers file): dense
  text-embedding-3-large with Naive 500 reaches R@8 about 32-35 % and P@8 about 4-5 %.
- The sample, the span checks and the document/character counts above (no retrieval).
- Expectation, stated as such and not as a result: BM25 is competitive with the paper's dense
  baseline on CUAD / PrivacyQA / ContractNLI, where queries share legal wording with the
  snippets, and loses on MAUD, where the paper's own numbers are near zero.

## Follow-up study

The effect of the "Consider ..." prefix (question-only queries, and routing to a document by the
prefix) is a separate, later pre-registered study: `docs/legalsplit_protocol.md`. It does not change
any system, number or rule in this file.

The chunk-size confound of the metric (k counts chunks, so Recall@k grows with chunk size) is
measured in `docs/legalsize_protocol.md`; it is the caveat for every comparison of these rows with
published rows of another chunker or chunk size. It changes nothing in this file.
