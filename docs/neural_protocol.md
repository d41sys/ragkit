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

## Outcome (scored runs; raw output in `docs/sota-runs/neural_*.txt`)

Runs: `ragkit beir --dataset <d> --retriever hybrid+ce` (four sets), `ragkit legalbench --retriever
hybrid+ce`, `ragkit neural --retriever hybrid+ce`; Apple MPS, models pinned as above. TREC-COVID
and Quora were not run (encoding predicted above the 20-minute rule). The header line of
`neural_legalbench_mini.txt` was relabelled from "model-free" to "neural" by hand after the run;
the code now prints that title.

BEIR nDCG@10 (test; mean of 300, 323, 648 and 1406 queries; CIs are in the raw files):

| System | SciFact | NFCorpus | FiQA | ArguAna | mean of 4 |
|---|---|---|---|---|---|
| `bm25` (Lucene-matched) | 0.678 | 0.320 | 0.235 | 0.404 | 0.409 |
| `dense` bge-small | 0.713 | 0.344 | 0.403 | 0.606 | 0.517 |
| `dense-e5` e5-small-v2 | 0.687 | 0.326 | 0.375 | 0.420 | 0.452 |
| `hybrid` (RRF 60) | 0.727 | 0.362 | 0.361 | 0.535 | 0.496 |
| `hybrid-convex` (alpha on dev) | 0.734 (0.4) | 0.363 (0.4) | 0.419 (0.3) | none | n/a |
| `bm25+ce@100` | 0.687 | 0.352 | 0.346 | 0.417 | 0.451 |
| `hybrid+ce@20` | 0.703 | 0.362 | 0.373 | 0.453 | 0.473 |
| `hybrid+ce@100` | 0.688 | 0.353 | 0.371 | 0.415 | 0.457 |
| published bge-small (MTEB) | 0.7127 | 0.3426 | 0.4034 | 0.6035 | 0.516 |
| published BM25+CE (BEIR paper) | 0.688 | 0.350 | 0.347 | 0.311 | 0.424 |
| published BGE-base (Pyserini) | 0.741 | 0.373 | 0.406 | 0.636 | 0.539 |

Mean-of-4 interval (added after the scored run, Phase L; the "mean of 4" column above had no CI
when it was first written): from the same per-query scores, recomputed from the cached BEIR
matrices (`ragkit beir --retriever dense --dataset all`, `docs/sota-runs/neural_beir_mean.txt`;
it reproduces the per-dataset rows above), resampling the queries of each dataset independently
(2000 resamples, seed 0) and averaging the four dataset means: `bm25` 0.409 [0.393, 0.425],
`dense` 0.517 [0.500, 0.533], paired difference `dense` minus `bm25` +0.107 [+0.095, +0.120]. The
interval covers query sampling within four datasets, not the choice of datasets.

Predictions:

- P1 holds: `dense` is within 0.003 of MTEB on all four sets (SciFact +0.0000, NFCorpus
  +0.0012, FiQA +0.0001, ArguAna +0.0026).
- P2 holds: `dense` beats `bm25` on every set (paired differences +0.035, +0.023, +0.168,
  +0.202, all CIs above 0); the largest gaps are ArguAna and FiQA.
- P3 does not hold: `hybrid` mean 0.496 exceeds `bm25` (0.409) but is 0.020 below `dense`
  (0.517): RRF with a much weaker BM25 pulls FiQA (0.361 vs 0.403) and ArguAna (0.535 vs 0.606)
  down. The dev-chosen convex weight fixes FiQA (0.419) but ArguAna has no dev split.
- P4 partly holds: `bm25+ce@100` reproduces the BEIR-paper BM25+CE within 0.02 on SciFact
  (0.687 vs 0.688), NFCorpus (0.352 vs 0.350) and FiQA (0.346 vs 0.347). It is not below `bm25`
  on ArguAna (0.417 vs 0.404; the paper's 0.311 is not reproduced), and `hybrid+ce` does not
  improve on `hybrid` on SciFact (0.688 vs 0.727); on FiQA it does (0.371 vs 0.361, unpaired).
- P5 partly holds: `dense` and `hybrid` Recall@8 exceed BM25 (18.51 and 17.48 against 13.37 on
  `naive500`; 17.14 and 16.16 against 11.84 on `recursive500`) and none reaches 31.86 (best:
  `dense-e5` + `recursive500`, 20.40). The cross-encoder does not help `hybrid` (16.4 to 17.7
  against 17.48), but it raises BM25 (15.2 to 16.5 against 13.37; CIs overlap).
- P6 holds: keyword-free Recall@5 is 0.490 (`dense`) and 0.531 (`hybrid`) against 0.347; the
  paired CI excludes 0 for `hybrid` ([+0.061, +0.306]) and not for `dense` ([-0.041, +0.306]).
- P7 holds: pooled wrong-act count 31 for `hybrid` against 34 for `bm25`.

EU decision rule (pooled 117 held-out questions, paired MRR difference to `bm25` with CI lower
bound above 0, and wrong-act count not above `bm25`'s 34): `hybrid` (+0.059 [+0.010, +0.110],
31), `hybrid-convex` (alpha 0.6 chosen on the original 42; +0.056 [+0.022, +0.093], 32), every
`+ce` row (+0.081 to +0.114; wrong-act 26 to 29) are recommended; `dense` (+0.057 [-0.009,
+0.123]) and `dense-e5` (+0.017 [-0.052, +0.088], 36) are not. The best row, `hybrid+ce@20`,
has pooled Recall@5 0.816 and MRR 0.714 against 0.679 and 0.600 for `bm25`. The keyword-free
set stays hard: `hybrid+ce@100` Recall@5 0.612 [0.469, 0.735]. Defaults are unchanged: the
numpy-only pipeline stays the default; `--retriever dense|hybrid|hybrid+ce` is opt-in.

The cross-encoder depth rows (20, 50, 100) of `hybrid+ce` differ by at most 0.015 nDCG@10 on
SciFact, NFCorpus and FiQA, by about 0.013 pooled MRR on EU, and by 0.038 on ArguAna (0.453 at
`@20`, 0.415 at `@100`); no depth is better everywhere.

## Phase K-R3: retrieval misses and candidate fixes (protocol: `docs/retrieval_miss_protocol.md`)

Six of the original 42 answerable EU questions have no gold article in the top-5 chunks of
`hybrid+ce@100` (which caps article-RAG at 32/42). A follow-up study diagnosed the misses and
tested four candidate fixes (article-level indexing, depth 50/100/200, convex fusion, query
instruction on/off), tuned only on fresh 29 + synthetic 40, and scored once on the pooled
held-out sets (130 questions: original 42, hard 39, keyword-free 49).

Outcome (`docs/sota-runs/retrieval_miss_heldout.txt`, `retrieval_miss_diagnosis.txt`):
- **Diagnosis**: 4 of the 6 misses were primary vocabulary mismatch (<50% term overlap), 1 chunk
  boundary split (term overlap 0.50 article vs 0.25 chunk), 1 query too short (13 words).
  For 3 of the 6 questions the cross-encoder demoted an article that the first stage had placed in
  the top 2; on 2 questions the article had a high article rank (2, 5) but its chunks were
  displaced past rank 5 by earlier articles.
- **Tuning**: F2 (depth 100), F3 (RRF), F4 (instruction off) and S (joint stack) all tied with B0
  at objective 0.9765; tie-break chose B0 for F2, F3 and S. F1 (article units) scored 0.9478.
- **Held-out scoring (130 questions)**:
  - B0 (`chunk/inst=on/rrf/D100/hybrid+ce`): MRR 0.664 [0.596, 0.732], G@5 (prod) 0.762, wrong-act 32.
  - F1 (`article` units): MRR 0.630, dMRR -0.034 [-0.089, +0.016], G@5 (prod) 0.792 (+0.031),
    wrong-act 38 (+6). Not adopted (no evidence; MRR CI spans 0, wrong-act increased).
  - F4 (`inst=off`): MRR 0.664, dMRR +0.000 [-0.004, +0.004], wrong-act 32. Not adopted (no evidence).
- **Verdict**: default unchanged (B0: chunk/inst=on/rrf/D100/hybrid+ce).
