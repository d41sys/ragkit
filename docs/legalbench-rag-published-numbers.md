# LegalBench-RAG(-mini): metric, published numbers, comparability

Compiled 2026-10-26. Every URL below was opened with `read`/`grep` during this compilation unless marked `[INFERENCE]`.
No models were run. "Not found" means I looked and the number is not in the page I opened.

Short version:

1. The official metric is **character-overlap precision/recall at top-k chunks**, averaged per query, then equal-weighted over 4 sub-datasets (§A).
2. "Mini" is **not a file**. It is a deterministic runtime sample of 194 queries per source (776 total) taken by `benchmark.py` (§A.3, §E).
3. The paper has **no BM25 row**. The only BM25-only numbers on mini that I found are in an unreviewed 2026-10-19 arXiv draft (StrucChunk, §D).
4. The paper's own tables contain internal inconsistencies (§B.3). Do not treat its "ALL" row of Table 4 as a macro-mean.
5. **Reference ceiling (2026-10-30 audit).** The paper's own dense baseline (Table 4, text-embedding-3-large, R@8 ALL 34.51 printed, 31.86 recomputed) and RCTS (26.30) are not a ceiling. The best published number on the same mini protocol is **StrucChunk + Hybrid (BGE-M3), R@8 ALL 48.18** (arXiv 2610.26313v1, 2026-10-19, §D; single unreviewed preprint, chunker description internally inconsistent). **PAKTON R@8 56.66 (EMNLP 2025 main 403, Table 2) is oracle-document retrieval** (the source document is known and search is in-document, Appendix F.2): not comparable, not a ceiling.
6. **Chunk size is a confound for every Recall@k comparison on this metric (2026-11-06, §A.5).** k counts chunks and the numerator sums overlapped gold characters without merging, so Recall@8 rises with chunk size (and overlap). In ragkit's own control, doubling the chunk size from 500 to 1000 characters adds 11 to 14 points of R@8, and about three quarters of that disappears at equal character budget. A published row with larger, overlapping or unstated chunk size is not a like-for-like gap to ours.

---

## A. Metric and protocol (from the official code)

Sources opened:

- Paper: `https://arxiv.org/pdf/2408.10343` and `https://arxiv.org/html/2408.10343v1` (arXiv v1, submitted 2024-08-19; only one version exists). Authors: Pipitone, Houir Alami (ZeroEntropy).
- Code: `https://github.com/zeroentropy-ai/legalbenchrag` (old org name in the paper: `zeroentropy-cc`). **Default branch is `master`**, not `main`. Raw files: `https://raw.githubusercontent.com/zeroentropy-ai/legalbenchrag/master/<path>`.
- Files read: `legalbenchrag/benchmark.py`, `legalbenchrag/run_benchmark.py`, `legalbenchrag/benchmark_types.py`, `legalbenchrag/methods/baseline.py`, `legalbenchrag/methods/retrieval_strategies.py`, `README.md`.
- Commit dates for `benchmark.py` / `run_benchmark.py`: 2024-08-14 "Add benchmark with baseline"; 2024-08-14 "Formalize LegalBenchRAG-Mini, with embedding batching and sqlite3 vector store"; 2024-08-15 "Add analytics with CSV generation over multiple retrieval methods". Latest repo commit 2025-05-30 (README only).

### A.1 Per-query precision and recall (character overlap)

Condensed from `run_benchmark.py` as I read it (not byte-for-byte; logic is exact):

```python
# precision(retrieved_snippets, gt_snippets)
total_retrieved_len = 0
relevant_retrieved_len = 0
for snippet in retrieved_snippets:
    total_retrieved_len += snippet.span[1] - snippet.span[0]
    for gt in gt_snippets:
        if gt.file_path == snippet.file_path:
            common_min = max(snippet.span[0], gt.span[0])
            common_max = min(snippet.span[1], gt.span[1])
            if common_max > common_min:
                relevant_retrieved_len += common_max - common_min
return 0 if total_retrieved_len == 0 else relevant_retrieved_len / total_retrieved_len

# recall(retrieved_snippets, gt_snippets)
total_relevant_len = 0
relevant_retrieved_len = 0
for gt in gt_snippets:
    total_relevant_len += gt.span[1] - gt.span[0]
    for snippet in retrieved_snippets:
        if snippet.file_path == gt.file_path:
            common_min = max(snippet.span[0], gt.span[0])
            common_max = min(snippet.span[1], gt.span[1])
            if common_max > common_min:
                relevant_retrieved_len += common_max - common_min
return 0 if total_relevant_len == 0 else relevant_retrieved_len / total_relevant_len
```

Consequences you must replicate or knowingly deviate from:

- Spans are half-open `[start, end)` character offsets into the document text. Match is on `file_path` equality.
- **Retrieved spans are not merged or de-duplicated.** Overlapping retrieved chunks are double counted in both numerator and denominator of precision, and recall can exceed 1.0. (OpenContracts independently confirms "no merging", §C.6.)
- Ground-truth spans within a query are validated as disjoint (`validate_snippet_list` raises if `prev.end >= cur.start`, so even touching spans raise). Retrieved-span validation is commented out in the code I read.
- Empty retrieval returns 0 for precision (no NaN). `avg([])` returns NaN.
- The unit of "k" is **number of chunks**, not characters. Precision@k therefore depends on chunk size.

### A.2 Averaging over queries and sources

`benchmark.py` sets `benchmark_name_to_weight = {"privacy_qa": 0.25, "contractnli": 0.25, "maud": 0.25, "cuad": 0.25}` and, per source, `weights.extend([weight / len(tests)] * len(tests))`. Each test is tagged `tags=[benchmark_name]`.
`BenchmarkResult.get_avg_recall_and_precision(tag_filter)` filters by tag, computes `avg_weight = avg(filtered_weights)`, and averages `value * weight / avg_weight`.

- Per-source number = plain mean over that source's queries.
- Overall ("ALL") = **equal-weight macro-mean over the four sources (0.25 each)**, independent of the number of queries or documents. The paper says the same: "we weight the metrics equally on each dataset, independently from the number of documents or queries they contain."
- **On mini, every source has exactly 194 queries, so the macro-mean equals the plain mean over all 776 queries.** On the full benchmark (194 / 977 / 1,676 / 4,042) they differ. `[INFERENCE]` from the weight formula above; the arithmetic is exact if the code is as read.
- Printed per source: "Avg Recall", "Avg Precision". Results go to `./benchmark_results/<run>/{i}_results.json` and `results.csv`.

### A.3 How "mini" is made (`benchmark.py`, quoted logic)

```python
MAX_TESTS_PER_BENCHMARK = 194
SORT_BY_DOCUMENT = True
if len(tests) > MAX_TESTS_PER_BENCHMARK:
    if SORT_BY_DOCUMENT:
        tests = sorted(
            tests, key=lambda test: (random.seed(test.snippets[0].file_path), random.random())[1]
        )
    else:
        random.seed(benchmark_name)
        random.shuffle(tests)
    tests = tests[:MAX_TESTS_PER_BENCHMARK]
```

- With `SORT_BY_DOCUMENT = True` (the shipped value) each test's sort key is the first draw of a Mersenne Twister seeded by the **string** `snippets[0].file_path`. All tests on the same document share the same key, so tests cluster by document; `sorted` is stable, so ties keep original JSON order.
- Effect: the 194 kept tests concentrate on few documents. Result per the paper (Table 3): 72 documents, 8,682,104 characters, versus 714 documents / 79,704,214 characters for the full benchmark (paper Table 2).
- **The retrieval corpus for mini is only the files referenced by the sampled tests** (sorted set of `file_path`s). If you index the whole corpus you are not running mini.
- PrivacyQA has exactly 194 queries, so its "mini" is the full file.
- Python's `random.seed(str)` is deterministic across CPython 3.x versions in the way used here (string seeds go through SHA-512 since 3.2) `[INFERENCE]`; do not substitute `numpy` or another RNG.

**Checksums to verify after sampling (paper Table 3, `https://arxiv.org/pdf/2408.10343`):**

| Source | Mini docs | Mini chars | Queries |
|---|---|---|---|
| ContractNLI | 18 | 184,267 | 194 |
| MAUD | 18 | 7,109,200 | 194 |
| CUAD | 29 | 1,211,773 | 194 |
| PrivacyQA | 7 | 176,864 | 194 |
| Total | 72 | 8,682,104 | 776 |

Caveats on those checksums (these matter):

- The paper's Table 3 and the 2508.13107 table label CUAD and MAUD counts in opposite order from each other (§C.5). Match by the character counts: MAUD docs are huge (≈7.1M chars), CUAD ≈1.2M chars.
- A third-party HF dataset card (`https://huggingface.co/datasets/awinml/legalbench-rag/raw/main/README.md`, date not visible) says the authors' mini code "covers 72 documents", which matches Table 3. It prints a Python re-implementation of the same sort key.
- Two independent re-implementations report different document counts: OpenContracts (§C.6) 20/17/16/7 (ContractNLI/CUAD/MAUD/PrivacyQA, = 60) and arXiv 2508.13107v2 (§C.5) the same 20/17/16/7. They did not reproduce 72. I did not resolve the cause. `[INFERENCE]` possible causes: different JSON ordering/version in the data they downloaded, or my counts are mislabeled by source. **Treat Table 3 doc/char counts as the acceptance test; if you get 60, you are on a different sample than the paper.**
- The paper says it ran "exactly 194 queries from each of the four datasets" (Section 3.3.2) and "All the experiments conducted in this work are run on LegalBench-RAG-mini" (Section 4). GitHub issue #2 on the repo (opened 2025-05-06, still open, 0 comments, `https://github.com/zeroentropy-ai/legalbenchrag/issues/2`) asks for "the exact questions used for the mini split" because it "was not mentioned anywhere", which supports that no mini file ships.

### A.4 k values, chunking, retrieval (official baseline)

From `retrieval_strategies.py`, `baseline.py`, and the paper Section 4.2:

- `top_ks = [1, 2, 4, 8, 16, 32, 64]`. All paper tables are percentages.
- `chunk_strategy_names = ["naive", "rcts"]`, `chunk_sizes = [500]`, `rerank_models = [None, Cohere rerank-english-v3.0]` → 2×1×2×7 = 28 pipelines (Embedding-Free RAG, §C.4, also cites "28 RAG pipelines").
- Embeddings: OpenAI `text-embedding-3-large` (query embedded as query type, chunks as document type), batch 2048. Vector store: sqlite-vec, distance ascending.
- Without reranker: `embedding_topk = top_k`. With reranker: `embedding_topk = 300`, `rerank_topk = top_k`. `token_limit = None`.
- **Naive chunker:** `content[i:i+500]` for i in steps of 500, over characters, no overlap.
- **RCTS:** LangChain `RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "!", "?", ".", ":", ";", ",", " ", ""], chunk_size=500, chunk_overlap=0, length_function=len, is_separator_regex=False, strip_whitespace=False)`. The code asserts `"".join(splits) == content`, so spans are contiguous and non-overlapping.
- Query = the full query string (including the "Consider <document description>; ..." prefix; descriptions are GPT-4o-mini generated per the paper). The benchmark **does leak the document title into the query**; GitHub issue #3 (2026-02-21, open) notes a fuzzy file-name match on entities finds the right file "almost every time" for MAUD.

### A.5 Chunk size and the character budget (ragkit's own control, 2026-11-06; not a published number)

Because k is a number of chunks (§A.1) and the numerator sums the overlapped gold characters of every retrieved chunk without merging or de-duplicating, **Recall@k at equal k grows with chunk size** (and with overlap, where the same characters count more than once). Any comparison between a 500-character row and a row with larger, structure-aware or overlapping chunks (StrucChunk, SAC, RCTS vs Naive) mixes retrieval quality with the characters retrieved per hit.

Ragkit measured that on its own retrievers (`docs/legalsize_protocol.md`, pre-registered `23f2040`; raw output `docs/sota-runs/legalsize_test.txt`): `naive250`, `naive500`, `naive1000` chunks, BM25 (Lucene-matched) and `e5-small-v2`, 620 test queries of the 776 mini queries (the legalsplit split), Recall ALL in %, paired 95% bootstrap CIs.

| retriever | R@8, size 250 / 500 / 1000 (equal k) | R at B = 4000 characters, size 250 / 500 / 1000 (k = 16 / 8 / 4) |
|---|---|---|
| `bm25` | 6.41 / 12.95 / 23.88 | 9.67 / 12.95 / 15.88 |
| `dense-e5` | 10.73 / 20.09 / 34.36 | 15.30 / 20.09 / 22.98 |

- 1000 vs 500, equal k = 8: +10.93 [+8.76, +13.22] (BM25), +14.26 [+11.69, +16.87] (`dense-e5`). At equal budget (B = 4000): +2.93 [+0.99, +4.95] and +2.88 [+0.43, +5.51]. Share of the equal-k gain that disappears at equal budget: 0.73 [0.58, 0.90] (BM25), 0.80 [0.65, 0.96] (`dense-e5`).
- 500 vs 250 (B = 4000) and 1000 vs 250 (B = 8000): shares 0.50 and 0.42 (BM25), 0.49 and 0.41 (`dense-e5`); the equal-budget differences are above 0 in all four.
- Pre-registered rule: the "mostly a budget effect" sentence applies only if the CI of the equal-budget BM25 1000-vs-500 difference includes 0; it does not (it is above 0), so the size effect beyond the budget is reported with its CI. The residual shows up from B = 4000 up; at B = 1000 and 2000 the 1000-vs-500 difference includes 0 in both retrievers.
- The control uses non-overlapping chunks, so it measures the budget effect, not the double counting of overlapping chunkers; that part follows from the metric definition in §A.1 (and `tests/test_ragkit_legalbench.py`).
- **How to read a comparison with a published row:** quote the chunk size and overlap of both sides; treat a gap at equal k between different chunkers as an upper bound on the retrieval-quality gap, not as that gap. The control is a yardstick for the size sensitivity of this metric (about 11 to 14 points of R@8 for 500 to 1000 characters), not an adjustment: it does not convert any published row.

---

## B. Published tables from the original paper

Source: `https://arxiv.org/pdf/2408.10343` (read), dated 2024-08-19 (arXiv v1). Table numbers below are **the paper's own**. The tau-labs leaderboard numbers the same tables 3–6 (own LaTeX labels).
Setup for all four: LegalBench-RAG-mini, text-embedding-3-large, chunk 500 chars no overlap, k ∈ {1,2,4,8,16,32,64}, values in percent. Column order: P@1, P@2, P@4, P@8, P@16, P@32, P@64 then R@1, R@2, R@4, R@8, R@16, R@32, R@64.

### B.1 Table 4. Naive (500 chars), no reranker

| Dataset | P@1..64 | R@1..64 |
|---|---|---|
| PrivacyQA | 7.86, 7.31, 6.41, 5.06, 3.58, 2.41, 1.54 | 7.45, 12.53, 20.88, 32.38, 42.45, 54.27, 66.07 |
| ContractNLI | 16.45, 14.80, 12.53, 9.73, 6.70, 4.65, 3.04 | 11.32, 19.10, 29.79, 45.59, 56.75, 69.88, 86.57 |
| MAUD | 3.36, 2.65, 2.18, 1.89, 1.48, 1.06, 0.75 | 2.54, 3.12, 4.53, 8.75, 13.16, 18.36, 25.62 |
| CUAD | 9.27, 8.05, 5.98, 4.33, 2.77, 1.77, 1.09 | 12.60, 19.47, 27.92, 40.70, 51.02, 64.38, 75.71 |
| ALL (as printed) | 2.40, 3.76, 4.97, 4.33, 3.39, 2.17, 1.29 | 3.37, 8.44, 21.30, 34.51, 48.88, 64.47, 76.39 |
| ALL (my mean of the 4 rows above, computed) | 9.23, 8.20, 6.78, 5.25, 3.63, 2.47, 1.60 | 8.48, 13.55, 20.78, 31.86, 40.84, 51.72, 63.49 |

### B.2 Table 5. RCTS, no reranker

| Dataset | P@1..64 | R@1..64 |
|---|---|---|
| PrivacyQA | 14.38, 13.55, 12.34, 9.03, 6.06, 4.17, 2.81 | 8.85, 15.21, 27.92, 42.37, 55.12, 71.19, 84.19 |
| ContractNLI | 6.63, 5.29, 3.89, 2.81, 1.98, 1.29, 0.90 | 7.63, 11.33, 17.34, 24.99, 35.80, 46.57, 61.72 |
| MAUD | 2.65, 1.77, 1.96, 1.40, 1.39, 1.15, 0.82 | 1.65, 2.09, 4.59, 6.18, 12.93, 21.04, 28.28 |
| CUAD | 1.97, 4.03, 4.83, 4.20, 2.94, 1.99, 1.25 | 1.62, 8.11, 17.72, 31.68, 44.38, 60.04, 74.70 |
| ALL | 6.41, 6.16, 5.76, 4.36, 3.09, 2.15, 1.45 | 4.94, 9.19, 16.90, 26.30, 37.06, 49.71, 62.22 |

(ALL equals the mean of the four rows within 0.01. Checked by arithmetic.)

### B.3 Table 6. "Naive + Cohere" (as printed)

The page I read gives per-dataset values that equal Table 5 within ±0.01: P@8 PrivacyQA 9.02, P@1 MAUD 2.64, P@2 ContractNLI 5.28, R@2 ContractNLI 11.34, R@4 MAUD 5.59, R@16 ALL 37.05. The ALL row carries a † whose footnote text is not visible in the conversion.
`[INFERENCE]` this looks like a copy/paste error in the paper (Table 6 ≈ Table 5), so **no trustworthy "Naive + Cohere" numbers exist** in the paper. The tau-labs leaderboard (`https://lee-jiahe-steven.github.io/tau-labs-legal-evals/legalbench-rag.html`, date unknown, states "reproduced verbatim from the paper; scores never recomputed") and PAKTON (§C.3) both reproduce the duplicate.

### B.4 Table 7. RCTS + Cohere rerank-english-v3.0

| Dataset | P@1..64 | R@1..64 |
|---|---|---|
| PrivacyQA | 13.94, 15.91, 13.32, 9.57, 6.88, 4.68, 3.28 | 7.32, 16.12, 25.65, 35.60, 51.87, 64.98, 79.61 |
| ContractNLI | 5.08, 5.59, 5.04, 3.67, 2.52, 1.75, 1.17 | 4.91, 9.33, 16.09, 25.83, 35.04, 46.90, 62.97 |
| MAUD | 1.94, 2.63, 2.05, 1.77, 1.79, 1.55, 1.12 | 0.52, 2.48, 4.39, 7.24, 14.03, 22.60, 31.46 |
| CUAD | 3.53, 4.18, 6.18, 5.06, 3.93, 2.74, 1.66 | 3.17, 7.33, 18.26, 28.67, 42.50, 55.66, 70.19 |
| ALL | 6.13, 7.08, 6.65, 5.02, 3.78, 2.68, 1.81 | 3.98, 8.82, 16.10, 24.34, 35.86, 47.54, 61.06 |

(ALL equals the mean of the four rows within 0.01. Checked by arithmetic.)

### B.5 Flags: paper-internal inconsistencies

- **Corpus counts disagree inside the paper.** Abstract and Table 1: 6,858 pairs (ContractNLI 946, CUAD 4,042, MAUD 1,676, PrivacyQA 194). Table 2: 6,889 (ContractNLI 977), 714 documents (ContractNLI 95, MAUD 150, CUAD 462, PrivacyQA 7), 79,704,214 characters. The HF card (`awinml/legalbench-rag`) and OpenContracts both give 6,889. GitHub issue #5 (2026-10-22, open) asks the authors which is intended. Mini is unaffected (it caps at 194).
- **Table 4 ALL ≠ mean of its dataset rows** (computed above: e.g. P@1 2.40 printed vs 9.23; R@64 76.39 vs 63.49). Tables 5 and 7 do match. `[INFERENCE]` Table 4's ALL row is an error or computed differently; do not use it as the macro-mean reference. Use the recomputed row, labeled as mine.
- **Text vs tables.** The text says RCTS with no reranker was best ("Both precision and recall were highest with RCTS + No reranker"). Table 4 (Naive) has higher recall for k ≥ 4 on every dataset row and on ALL as printed. The paper's Table 4 per-dataset rows are corroborated at k=32 on CUAD/MAUD by OpenContracts (§C.6), so the rows look right and the prose looks wrong.
- **Table 6 duplicates Table 5** (above).

### B.6 Reference points for a model-free baseline (what to beat)

At k = 8 on mini, ALL, from the paper: Naive R@8 34.51 (printed ALL; my recomputed mean 31.86) with P@8 4.33 (printed; recomputed 5.25); RCTS R@8 26.30, P@8 4.36; RCTS + Cohere R@8 24.34, P@8 5.02. The authors' stated best is RCTS without reranker.

---

## C. Later work: numbers and comparability verdicts

Verdict key: **COMPARABLE** = same mini sample, same char-overlap metric, same k-chunk protocol. **PARTLY** = same metric/k grid, but different embedder, chunker or sample (so it informs but is not a head-to-head). **NOT** = different task unit, different corpus scope, or different metric.

### C.1 SAC (Summary-Augmented Chunking), Reuter et al.
- Sources: `https://arxiv.org/abs/2510.06999` (submitted 2025-10-08), `https://aclanthology.org/2025.nllp-1.3.pdf` (NLLP 2025, pp. 17–30). Code: `https://github.com/DevelopedByMarkus/summary-augmented-chunking` (7 stars; built on the official `legalbenchrag` code).
- Setup: `thenlper/gte-large`, FAISS cosine, recursive character chunking 500 chars (also 200, 800), no overlap; metrics: text-level precision/recall (official style) plus Document-level Retrieval Mismatch (DRM); k ∈ {1,2,4,8,16,32,64}; averaged over k and seeds.
- Main results (Fig. 3, Fig. 4) are **figures only; no numeric per-k tables**. Fig. 4 embedding ablation: `text-embedding-3-large`, `gte-large`, `bge-base-en-v1.5`, `legal-bert-base-uncased` (figure only; text says OpenAI best).
- Appendix A Table 1 (averaged over the k grid; chunk size and summary length in characters). Column-to-configuration mapping is `[INFERENCE]` from the text (500/150 chosen, bold) and from Table 2's first row matching it:

  | Chunk / summary | Precision | Recall | DRM |
  |---|---|---|---|
  | 200 / 150 | 10.64 | 23.43 | 20.70 |
  | 200 / 300 | 8.05 | 22.91 | 23.49 |
  | 500 / 150 (chosen) | 11.03 | 41.80 | 19.29 |
  | 500 / 300 | 8.45 | 37.77 | 20.61 |
  | 800 / 150 | 7.76 | 42.93 | 34.96 |
  | 800 / 300 | 6.79 | 43.90 | 29.68 |

- Appendix B Table 2 (dense + BM25 weighting at 500/150, avg over k): 100/0 → P 11.03, R 41.80, DRM 19.29; 75/25 → 10.57, 42.54, 19.11; 50/50 → 9.54, 41.47, 18.45; 25/75 → 8.23, 36.56, 18.18. **Pure BM25 (0/100) is not reported.** Text: BM25 weight improves DRM but lowers precision and recall.
- Dataset subset: not stated as mini. Captions say "averaged over seeds", implying random query sampling, and a "362 documents" ContractNLI pool is cited, which matches neither mini (18) nor full (95). `[INFERENCE]` they evaluate a different sample/pool than the paper's deterministic mini.
- **Verdict: PARTLY at best, in practice NOT head-to-head.** Averages over k cannot be compared with the per-k tables. Reproducibility of the subset is unknown.

### C.2 StrucChunk (see §D for BM25 numbers)
- `https://arxiv.org/abs/2610.26313v1` (v1 2026-10-19; 7 days old at compile time; non-peer-reviewed, independent authors). Code: `https://github.com/SinghSuryaDeep/legal-rag-strucchunk-1` (0 stars; has `run_legalbench_rag.py` and `src/benchmarks/legalbench_rag_evaluator.py`).
- Claims: Char-Recall@8 ALL 48.18% vs 23.59% best baseline (Recursive+Dense BGE-M3) "surpassing the published state of the art by +83%". The +83% compares against RCTS 26.30 and skips the Naive 34.51 row in Table 11 (against 34.51 the gain is ≈ +40%, my arithmetic). One sentence misreads R@32 as R@64 ("Recall@64 dropped from 61.06% to 47.54%"). Draft hygiene flags: an "X/100" placeholder and a "[?]" citation.
- Baseline chunking is described inconsistently: §4.1 says "500-char fixed-size chunks with no overlap"; §4.5 and Table 6 say RecursiveCharacterTextSplitter at 512 tokens with 15% overlap. "~17K chunks" over the 8.68M-char mini corpus implies ≈ 511 chars per chunk `[INFERENCE]`. Overlapping chunks are double counted by the official metric.
- **Verdict: PARTLY, unverified; best published comparable number as of 2026-10-30 (R@8 ALL 48.18, StrucChunk + Hybrid with BGE-M3).** It states LBR-mini (776 queries, 72 documents, 0.25 weights, char-level P/R, k 1..64, seed 42), so it is the closest independent mini run, but chunker, corpus parsing and sampling are not independently confirmed. Single unreviewed preprint.
- **Chunk-size caveat (2026-11-06, §A.5).** The StrucChunk Hybrid row is a Recall@8 at equal k from a structure-aware chunker whose chunk size this sheet does not record (its baseline description is inconsistent, above). Ragkit's own size-matched control shows that Recall@8 on this metric rises by 11 to 14 points when only the chunk size is doubled from 500 to 1000 characters, about three quarters of it a character-budget effect. The 48.18 versus our 13 to 20 is therefore not a like-for-like retrieval-quality gap: it cannot be converted into a size-matched gap from the data here; part of it may be the characters retrieved per hit.

### C.3 PAKTON (multi-agent legal system)
- EMNLP 2025 main, paper 403: `https://aclanthology.org/anthology-files/anthology-files/pdf/emnlp/2025.emnlp-main.403.pdf` (read; Table 2 and Appendix F; date: EMNLP 2025, exact date not visible in the text I read).
- Table 2 reproduces the paper's four baselines verbatim (including the Table 6 duplicate). PAKTON ALL row, **without** LLM filtering, config 1: P@1..64 22.34, 18.37, 11.52, 7.63, 5.26, 4.08, 3.52; R@1..64 26.77, 36.32, 45.26, 56.66, 69.17, 83.50, 89.58. Per-dataset rows (e.g. ContractNLI R@64 99.82, PrivacyQA P@1 19.94) are in their Table 2. The CUAD and ALL cells were misaligned by PDF extraction and I did not verify the column alignment; **re-verify against the PDF before quoting**.
- Protocol (Appendix F.2): "Each document from the dataset was indexed by the Archivist, and the corresponding queries were directly submitted to the Researcher", with the Researcher doing "only in-document search". GPT-4o query rewriting, BM25 top-100 (threshold 0.6) + dense top-100, RRF, then BAAI/bge-reranker-v2-m3 top-64, RCTS 1,000-char fallback chunks.
- Whether they use mini or the full set: not stated in the lines I read.
- **Verdict: NOT comparable, oracle-document retrieval.** It is per-document (the source document is known and search runs in-document, Appendix F.2), uses an LLM query optimizer, 1,000-char chunks, and a cross-encoder. It also has no BM25-only row. Its R@8 56.66 is higher than StrucChunk's 48.18 but is not a ceiling for corpus-level retrieval.

### C.4 Embedding-Free RAG (Goldman Sachs)
- `https://aclanthology.org/2025.findings-emnlp.1360.pdf` (EMNLP Findings 2025, pp. 24974–24985, Nov 4–9, 2025).
- Method: an LLM quotes anchors in the document and sentence windows (Gemini 1.5 Flash, w=5) are expanded around them. Not top-k chunks. Evaluated on the full "nearly 7000 queries", with P/R/F1 per query. Fig. 2 F1 per dataset is a figure only; the text says 2.6× the best of the paper's 28 pipelines on average (F1) and 4.6× on MAUD, where recall is 0.66 and precision 0.08 against the best paper recall 0.31 and precision 0.01.
- Table 1 (10-query PrivacyQA subset): Single Document P 0.1489 / R 0.6571 / F1 0.2428; Multi-Document 0.1395 / 0.6339 / 0.2287. Table 2 (full PrivacyQA, varying LLM): Llama 3.1 8B 0.0991 / 0.6315 / 0.1714; Llama 3.1 70B 0.1709 / 0.6611 / 0.2717; Qwen 2.5 14B 0.1787 / 0.6684 / 0.2821; R1-Distill Llama 70B 0.1525 / 0.7456 / 0.2533.
- **Verdict: NOT comparable** (single-document oracle setting, anchor-window units, not k-chunk, not mini).

### C.5 "All for law and law for all: Adaptive RAG Pipeline for Legal Research" (UCL)
- `https://arxiv.org/html/2508.13107v2` (v2 dated 2025-09-10).
- Their own mini re-creation ("exact reproduction was impossible" because the seed is unknown to them): 194 QA per dataset; ContractNLI 20 docs, CUAD 17, MAUD 16, PrivacyQA 7 (total 60 vs 72 in the paper's Table 3). Their Table 1 label order differs from the paper (their "CUAD 18 / MAUD 29" is the swapped pair).
- Retrieval: SBERT `all-mpnet-base-v2`, `all-MiniLM-L6-v2`, `gte-large`; cosine and BM25 similarity; top-50 then rerank, K 1–50; adds file-level filters via a query translator.
- Results are figures only (Figs. 2, 3, 7). Text: cosine beats BM25 similarity; RCTS + SBERT + cosine + unranked best. No numeric BM25 table found.
- **Verdict: NOT comparable.** Different sample, document filtering added, figure-only.

### C.6 OpenContracts results page
- `https://open-source-legal.github.io/OpenContracts/benchmarks/legalbench_rag_results` (page last updated 2026-04-28).
- Ports `benchmark.py` sampling and the metric verbatim; equivalence-tested over 200 random trials per the page. Records the full JSON sizes: privacy_qa 194, contractnli 977, cuad 4,042, maud 1,676 (6,889 total).
- Config B (500-char sliding window, text-embedding-3-large, no rerank), k=32: recall privacy_qa 0.7033, contractnli 0.6274, cuad 0.6478, maud 0.1848, macro 0.5408; precision macro 0.0240. The paper's Table 4 at k=32 is 0.5427 / 0.6988 / 0.6438 / 0.1836, with printed ALL 0.6447 (my mean of the four rows 0.5172). The CUAD/MAUD values agree to ≈0.004/0.001; PrivacyQA and ContractNLI differ by 0.16 and 0.07 (different sample or chunking details `[INFERENCE]`).
- Config A (MiniLM, paragraph chunks): recall macro 0.6339, precision macro 0.0074. Config C (3-large, paragraph chunks, 6,000 chars): recall macro 0.8190. The authors reject C as a "budget artifact" (30–137K characters retrieved per query).
- Its sample has 20/17/16/7 documents, not 18/29/18/7 (§A.3). It also notes that the paper does not say whether its numbers used the 194-cap sample or the full subsets.
- **Verdict: COMPARABLE on protocol/metric code, PARTLY on data** (sample differs from Table 3). A project report, not peer reviewed.

### C.7 tau-labs leaderboard
- `https://lee-jiahe-steven.github.io/tau-labs-legal-evals/legalbench-rag.html` (date unknown). Copies the paper's four tables verbatim, with its own numbering 3–6. No new results.
- **Verdict: not independent**; same numbers as §B, including the Table 6 duplicate.

### C.8 "RAG Components Are Not Independently Rankable" (PleIAs / THWS)
- `https://aclanthology.org/2026.docinsights-1.22.pdf` (Workshop on Document Intelligence and Understanding, pp. 303–323, October 24, 2026).
- Uses LegalBench-RAG **full "complete frozen archive": 714 documents, 6,889 questions**; metric is LLM-judge claim-level recall at equal token capacities (256–4,096 tokens), not char-overlap P/R@k. Top-20 candidates, retrievers BM25 / dense / late-interaction / RRF fusion.
- Only relevant statement about BM25: adding BM25 to a matched neural parent *reduces* claim recall on LegalBench-RAG by 2.85 points (averaged over six parents and four chunkers; FinanceBench −5.92; MultiHop-RAG +4.53). No BM25-only absolute number on LegalBench-RAG found in the text I read; per-cell tables are figures.
- **Verdict: NOT comparable** (different metric, full set, token budget). Useful only as direction-of-effect evidence that BM25 fusion does not reliably help on this benchmark.

### C.9 Candidates that do NOT contain usable LegalBench-RAG numbers
- `https://github.com/Akshitha024/legal-retrieval-benchmark` (personal repo, 0 stars): BM25 nDCG@10 0.245, Recall@10 0.493, MRR@10 0.171; dense bge-small 0.133 / 0.301 / 0.084; RRF 0.230 / 0.459 / 0.161, on "CUAD-QA, 1,000 queries over 36 unique contracts", doc-level qrels, "span-level eval is TODO". Its dataset table (contractnli 607 docs / 2,091 queries; maud 152 / 187; privacy_qa 8 / 1,750) does **not** match LegalBench-RAG (95/977, 150/1,676, 7/194). **NOT comparable; do not cite as a LegalBench-RAG BM25 number.**
- ZeroEntropy posts, all opened: `https://zeroentropy.dev/articles/legalbench-rag-the-first-open-source-retrieval-benchmark-for-the-legal-domain` (2024-11-29; difficulty figure only, describes mini as 776 pairs), `https://zeroentropy.dev/articles/best-reranker-legal` (2025-07-14; qualitative, recommends BM25 first stage then reranking, no numbers), `https://zeroentropy.dev/articles/announcing-zeroentropy-s-first-rerankers-zerank-1-and-zerank-1-small` (2025-07-10; internal "zbench" nDCG@10, no LegalBench-RAG), `https://www.zeroentropy.dev/articles/best-embedding-for-legal` (2026-04-10; "0.6723 NDCG@10" on an unnamed "legal benchmark" average; OpenAI 3-large 0.5099; benchmark composition not stated, **not LegalBench-RAG per this page**). ZeroEntropy joined Notion 2026-07-24 (same site).
- MLEB (`https://arxiv.org/pdf/2510.19365`, 2025-10-22) only cites LegalBench-RAG as prior work; it does not report LegalBench-RAG scores.
- `emergentmind.com/topics/legalbench-rag*` pages are secondary summaries, not opened in full, not used for numbers.

### C.10 ragkit's own prefix-free and document-routed rows (2026-11-06; not published numbers)

`docs/legalsplit_protocol.md`, raw output `docs/sota-runs/legalsplit_score.txt`; test split of 620 of the 776 mini queries, Recall@8 ALL in %.

| ragkit system (test split) | same-system full-query baseline | arm |
|---|---|---|
| `bm25-lucene` + `naive500`, question only after the first `;` | 12.95 | 6.92 |
| `dense-e5` + `recursive500`, question only | 20.70 | 12.65 |
| `bm25-lucene` + `naive500`, routed to one document by the query prefix, question only | 12.95 | 47.24 |
| `dense-e5` + `recursive500`, routed to one document by the query prefix, question only | 20.70 | 54.39 |

- **Question-only rows: NOT head-to-head.** Same sample, metric and k, but the query text differs from the official run (it removes the document-naming prefix). They are also lower than the full-query baselines, so there is no reason to quote them against a published row.
- **Document-routed rows: NOT comparable, oracle-like.** The system picks the document from the file name and opening text matched to the query's own prefix (95.6 % top-1 on test) and searches only inside it, as PAKTON's Appendix F.2 does with a known source document (C.3). They use document metadata that the published rows (paper Tables 4, 5, 7, StrucChunk) do not use. Do not put them next to the 48.18 row; they belong with the oracle-document rows of C.3 and C.1.
- The full-query rows stay the only ragkit rows in the head-to-head.

---

## D. Published BM25 numbers

| Source | BM25 on LegalBench-RAG? | Details |
|---|---|---|
| Original paper (2408.10343) | **None.** No BM25 baseline | Only dense (3-large) ± Cohere rerank |
| SAC (2510.06999) | Mixed only, no pure BM25 | Table 2: dense+BM25 weight sweep, averaged over k; 25% semantic/75% BM25 gives P 8.23, R 36.56 |
| 2508.13107v2 | Figures only | BM25 similarity worse than cosine; no numbers read |
| Docinsights 2026 | Delta only | BM25-RRF vs neural parent: −2.85 pp claim recall, different metric |
| **StrucChunk (2610.26313v1)** | **Yes, BM25-only on mini** | Tables 8–10 below |

StrucChunk BM25 numbers (`https://arxiv.org/pdf/2610.26313`, v1 2026-10-19). `bm25s` v0.2.0 over "recursively chunked text" (chunker ambiguity in §C.2). Unreviewed single source; no second confirmation.

Table 8, Char-Recall@k (%), ALL, equal-weighted:

| Method | R@1 | R@2 | R@4 | R@8 | R@16 | R@32 | R@64 |
|---|---|---|---|---|---|---|---|
| Recursive + Dense (BGE-M3) | 6.05 | 9.93 | 16.42 | 23.59 | 33.87 | 43.55 | 51.30 |
| **BM25 only** | 2.24 | 3.65 | 6.21 | 10.10 | 16.70 | 22.94 | 32.72 |
| Naive Hybrid | 3.46 | 5.80 | 11.21 | 18.46 | 27.13 | 36.26 | 47.25 |
| StrucChunk + Hybrid | 15.45 | 24.92 | 36.62 | 48.18 | 56.96 | 65.30 | 70.65 |

Table 9, Char-Precision@k (%), ALL:

| Method | P@1 | P@2 | P@4 | P@8 | P@16 | P@32 | P@64 |
|---|---|---|---|---|---|---|---|
| Recursive + Dense | 6.64 | 5.85 | 5.10 | 3.85 | 2.87 | 1.94 | 1.21 |
| **BM25 only** | 1.92 | 1.71 | 1.52 | 1.39 | 1.24 | 0.93 | 0.70 |
| Naive Hybrid | 3.01 | 3.04 | 3.00 | 2.82 | 2.22 | 1.54 | 1.05 |
| StrucChunk + Hybrid | 4.51 | 3.85 | 2.94 | 2.07 | 1.25 | 0.69 | 0.34 |

Table 10, Char-Recall@8 per source:

| Method | ContractNLI | CUAD | MAUD | PrivacyQA | ALL |
|---|---|---|---|---|---|
| Recursive + Dense | 35.30 | 22.42 | 4.76 | 31.90 | 23.59 |
| **BM25 only** | 22.82 | 7.15 | 1.04 | 9.38 | 10.10 |
| Naive Hybrid | 35.61 | 15.53 | 3.20 | 19.51 | 18.46 |
| StrucChunk + Hybrid | 58.36 | 46.22 | 16.91 | 71.24 | 48.18 |

Read-across: their BM25-only R@8 (10.10) sits well below the paper's OpenAI-dense Naive R@8 (34.51 printed) and RCTS R@8 (26.30), but their dense BGE-M3 baseline is also well below the paper's dense numbers, so their chunker/corpus pipeline is stricter or different. `[INFERENCE]` Treat the BM25-to-dense *ratio* within StrucChunk as the more reliable signal than the absolute level; the gap to the paper's tables is unexplained.

---

## E. Data layout and how mini is identified

- README (`https://raw.githubusercontent.com/zeroentropy-ai/legalbenchrag/master/README.md`) requires `./data/corpus` (raw text files, possibly nested) and `./data/benchmarks` (JSON files). Data link: a Dropbox folder in the README. The folder resolves to a zip of 90,591,976 bytes (my fetch was refused above the 52,428,800-byte tool limit). **I did not download it, so the exact in-zip file listing is unverified.**
- Layout implied by code + the HF card (`https://huggingface.co/datasets/awinml/legalbench-rag/raw/main/README.md`, "as distributed by the authors, without changes"; date not visible; third-party):
  ```
  benchmarks/{privacy_qa,contractnli,maud,cuad}.json     # {"tests":[{"query":..., "snippets":[{"file_path":"<source>/<file>.txt","span":[s,e],"answer":...}]}]}
  corpus/privacy_qa/*.txt   (7 files)
  corpus/contractnli/*.txt  (95)
  corpus/cuad/*.txt         (462)
  corpus/maud/*.txt         (150)
  SHA256SUMS
  ```
  `file_path` is relative to the corpus directory (so the first path segment is the source). Full-set corpus is 80,352,538 bytes per that card (the paper's Table 2 gives 79,704,214 characters, a bytes-vs-characters difference `[INFERENCE]`). Concatenated benchmark JSONs in the order privacy_qa, contractnli, maud, cuad have sha256 `991d965c4e33873cd4db2cfc6e06786a2f313d6d9d310c798e69efde02e0906c` per that card; use this only as an identity check against the Dropbox data (GitHub issue #5 asks the authors to confirm the byte-identity of that HF copy, still open as of 2026-10-22).
- **Offset gotchas from the HF card** (I could not confirm these against the official loader, so verify `text[start:end] == answer` on your own data):
  - Read corpus files as UTF-8 in default text mode (`\r\n` → `\n`). Do not use `newline=""`. Keep the BOM at index 0 (use `utf-8`, not `utf-8-sig`).
  - 16 MAUD file names contain `||`; 16 ContractNLI names contain URL escapes like `%20`; one CUAD `file_path` is NFD while the file name is NFC, so normalize both sides.
  - Spans within a query do not overlap; the dataset has no query IDs.
- **Identify mini at runtime:** run the `benchmark.py` sampling (§A.3) on each JSON, then check (a) 194 queries per source, (b) 72 documents (18/18/29/7) and the Table 3 character totals. There is no `*_mini.json`, no mini flag, and no published list of the 776 queries (GitHub issue #2).
- `benchmarks/` + `corpus/` sit side by side under `./data/`; the benchmark's `file_path` is relative to `./data/corpus/`.

---

## F. What is not verified

1. Dropbox contents and exact file names (not downloaded).
2. Whether the official sampling reproduces Table 3's 72 documents. Two re-implementations got 60; the HF card asserts 72 (no code run by me).
3. PAKTON per-dataset CUAD/ALL row alignment (PDF extraction was shifted).
4. SAC dataset subset and absolute per-k numbers (figures only).
5. StrucChunk: all numbers are from one unreviewed draft. Its chunker description conflicts internally.
6. Dates marked "unknown": the tau-labs leaderboard, the awinml HF card, PAKTON's exact EMNLP date. The ZeroEntropy dates are from the page metadata.
7. Embedding-Free RAG Fig. 2 and 2508.13107v2 Figs. 2, 3, 7 are figures; no numbers extracted.
