# Neural retrieval protocol (K-R1): dense, hybrid, cross-encoder (frozen before scoring)

Goal: measure what small public neural models add to ragkit's BM25 on the same benchmarks,
splits and metrics as the published numbers in `docs/beir-published-numbers.md` and
`docs/legalbench-rag-published-numbers.md`, and on the four EU question sets. This file is
committed before any neural score is computed. Smoke runs (`--max-queries 5`) only check that the
code runs; their output is discarded and no choice below depends on it. Results are reported
whatever they are; "no detectable effect" means the paired 95 % CI includes 0.

The core install stays numpy-only. Every model import is lazy and lives in `ragkit.neural`; the
extras are `uv sync --extra st`. The default `bm25` path imports no model library (the existing
"no model import" tests still run).

## Models (pinned; public, ungated, MIT / Apache-2.0)

| role | model | revision (Hugging Face `sha`, read 2026-10-30) | size on disk |
|---|---|---|---|
| dense (primary) | `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | 133 MB |
| dense (secondary, dense-only rows) | `intfloat/e5-small-v2` | `ffb93f3bd4047442299a41ebb6fa998a38507c52` | 133 MB |
| cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | 91 MB |

Disk cap for new models: 600 MB (planned 357 MB). Device: Apple MPS when available, else CPU;
fp32; sentence-transformers `encode(..., normalize_embeddings=True)`, `max_seq_length` 512.

- bge-small: queries get the instruction `Represent this sentence for searching relevant
  passages: `; documents get nothing. Cosine = dot product of unit vectors.
- e5-small-v2: `query: ` / `passage: ` prefixes.
- Cross-encoder: raw logit of `(query, document text)`, `max_length` 512, no activation.

## Systems (frozen)

| id | definition |
|---|---|
| `bm25` | The benchmark's existing BM25: BEIR and LegalBench `bm25-lucene` (k1 0.9, b 0.4, Lucene English analyzer); EU the shipped `BM25Index` (k1 1.5, b 0.75) |
| `dense` | bge-small cosine over the same retrieval units |
| `dense-e5` | e5-small-v2 cosine (secondary; no hybrid or reranker built on it) |
| `hybrid` | RRF (k = 60) of `bm25` top 100 and `dense` top 100 |
| `hybrid-convex` | `alpha * minmax(bm25) + (1 - alpha) * minmax(dense)` over the same two top-100 lists, a document missing from a list scores 0 in it. `alpha` in `{0, 0.1, ..., 1}`, chosen on the dev split only (below) |
| `bm25+ce` | cross-encoder rerank of the `bm25` top 100 (the published "BM25+CE" setting) |
| `hybrid+ce` | cross-encoder rerank of the `hybrid` top 100 |

Cross-encoder depth: top 100 is the primary row. Because a pair's cross-encoder score does not
depend on what else is in the list, the top-20 and top-50 rows rerank the first 20 / 50 of the
same first-stage list from the same scores (sensitivity rows; no selection among depths).
A rerank only permutes its candidates, so Recall@100 of `bm25+ce` / `hybrid+ce` equals the first
stage's.

## Benchmarks and splits

### BEIR (units: one document = `title + " " + text`; no chunking)

SciFact, NFCorpus, FiQA-2018, ArguAna, test split, trec_eval conventions of `ragkit.beir`
(nDCG@10 linear gain, Recall@100, self hit removed, bootstrap CI over queries). The large
corpora (TREC-COVID 171k, Quora 523k documents) are skipped: only `ragkit beir`'s fetch path is
used and bge-small encoding on this laptop is measured on SciFact first; a set is run only if
the measured rate predicts under 20 minutes.

Dev for `hybrid-convex`: SciFact `qrels/train.tsv`, NFCorpus `qrels/dev.tsv`, FiQA
`qrels/dev.tsv`. ArguAna ships only a test split: RRF only, no `hybrid-convex` row. Dev metric:
nDCG@10; ties between `alpha` values go to the one closest to 0.5, then the smaller. Test qrels
are never used to choose anything.

Published numbers beside ours, same dataset, split and metric:

| system | SciFact | NFCorpus | FiQA | ArguAna | source |
|---|---|---|---|---|---|
| bge-small-en-v1.5 (MTEB), nDCG@10 | 0.7127 | 0.3426 | 0.4034 | 0.6035 | results repo commit `7818c6fe`, `results/BAAI__bge-small-en-v1.5/` (`5c38ec7c` dir for SciFact, NFCorpus, ArguAna; `external/` model-card entry for FiQA) |
| bge-small-en-v1.5 (MTEB), Recall@100 | 0.945 | 0.3116 | 0.6955 | 0.9879 | same files |
| e5-small-v2 (MTEB), nDCG@10 | 0.6885 | 0.3245 | 0.3743 | 0.4179 | `external/` entries; ArguAna from the `dca8b1a9` dir |
| e5-small-v2 (MTEB), Recall@100 | 0.9277 | 0.298 | 0.6819 | 0.9239 | same files |
| BM25+CE (BEIR paper), nDCG@10 | 0.688 | 0.350 | 0.347 | 0.311 | `docs/beir-published-numbers.md` section 3 |
| BGE-base-en-v1.5 (Pyserini), nDCG@10 / R@100 | 0.741 / 0.967 | 0.373 / 0.337 | 0.406 / 0.742 | 0.636 / 0.992 | same, sections 3 and 4 |

The bge-small rows are the reproduction check. Pre-registered tolerance: our `dense` nDCG@10
within 0.015 (absolute) of the MTEB value on every set. A miss is reported as a reproduction
failure, not tuned away. Published numbers were produced with other library versions and batch
settings; the MTEB bge-small entries were run by mteb 1.18 to 1.20.

### LegalBench-RAG-mini (776 queries, 72 contracts)

Official character-overlap Precision/Recall@k of `ragkit.legalbench`; both existing chunkers
(`naive500`, `recursive500`, no overlap) so that rows line up with the shipped BM25 rows. `bm25`
is the Lucene-matched BM25 of the same chunker (Recall@8 13.37 for `naive500`, 11.84 for
`recursive500`, `docs/sota-runs/legalbench_mini.txt`). No dev split exists: RRF only, no `hybrid-convex`.
Cross-encoder rerank of the top 100.

### EU question sets (934 chunks of 94 articles, chunks 500 / 50 as in `ragkit.ctxprefix`)

- Retrieval unit: the chunk. The ranking is the first 10 chunks, deduplicated to articles
  (`rank_documents`), as for every earlier EU table. Metrics: Recall@5, MRR, nDCG@10
  (`ragkit.lexical.score_question`).
- Dev (the only place `alpha` is chosen, by nDCG@10): the original 42 questions
  (`qa_eu.jsonl`; already seen during earlier development, so it is not a test set).
- Test: fresh 29 (`qa_eu_fresh.jsonl`), hard 39, keyword-free 49 (`qa_eu_nokw.jsonl`), each
  alone and pooled (117 questions).
- Paired 95 % bootstrap CIs (2000 resamples, seed 0) of MRR and Recall@5 against `bm25` on the
  same questions; wrong-act count = questions whose top-1 article is from an act none of the
  relevant articles belong to (`ragkit.ctxprefix.is_wrong_act`).

## Predictions (written before any run, from published numbers only)

- P1 reproduction: `dense` nDCG@10 within 0.015 of MTEB bge-small on each of the four BEIR sets.
- P2 dense vs BM25 on BEIR: `dense` beats `bm25` on every set; the gap is largest on ArguAna and
  FiQA (published: +0.2, +0.17).
- P3 hybrid: `hybrid` mean nDCG@10 over the four sets exceeds `bm25` and is within 0.01 of
  `dense` or higher.
- P4 cross-encoder: `bm25+ce` reproduces the BEIR-paper BM25+CE within 0.02 on SciFact, NFCorpus
  and FiQA, and is below `bm25` on ArguAna (published 0.311 vs 0.397 to 0.414). `hybrid+ce`
  improves on `hybrid` on SciFact and FiQA and not on ArguAna.
- P5 LegalBench-RAG-mini: `dense` and `hybrid` Recall@8 exceed BM25's 13.37; the cross-encoder
  does not help (the benchmark paper reports a general reranker hurting); none reaches the
  paper's `text-embedding-3-large` naive-500 baseline (Recall@8 31.86 recomputed from its four
  printed rows; 34.51 as printed), which uses a larger embedding model.
- P6 EU: `hybrid` and `dense` raise Recall@5 on the keyword-free set above BM25's 0.347.
- P7 EU wrong-act count does not rise for `hybrid` relative to `bm25` on the pooled test sets.

## Decision rule (EU; no default changes either way)

A neural retriever X (`dense`, `hybrid`, `hybrid-convex`, `hybrid+ce`) is recommended for
EU-style text when, on the pooled 117 held-out questions, its paired MRR difference to `bm25` has
a CI lower bound above 0 and its wrong-act count is not above `bm25`'s. The numpy-only default is
not changed in any case; the neural retrievers are opt-in (`--retriever dense|hybrid|hybrid+ce`).
BEIR and LegalBench carry no decision rule: they are reproduction checks and comparisons to
published numbers.

## Not claimed

Published bge-small / BGE-base rows come from other harnesses; differences below 0.015 are not
interpreted. SciFact's train split, used here as dev for `alpha`, may overlap bge's training data
(`[INFERENCE]`; the model card does not list it). Hybrid weights chosen on one dataset's dev
split are not transferred to another. The EU labels are agent-reviewed (README caveat).
