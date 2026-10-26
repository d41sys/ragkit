# RAG state of the art, 2023 to 2026, and where ragkit sits

Written 2026-10-26. Scope: retrieval-augmented generation (RAG) for regulation text, with weight on retrieval, reranking and abstention (answering "the corpus does not say"). ragkit is a small, model-free-by-default RAG kit with an EU electricity-regulation corpus; its own numbers come from `README.md`.

## Summary (five findings)

1. **Retrieval:** on BEIR, BM25 is beaten by a base-size dense encoder on most sets (FiQA, ArguAna, NQ, SciDocs, SciFact) and wins only on Touché-2020 in the Pyserini table (section 2a). ragkit has run no neural embedder.
2. **Rerankers:** LLM and cross-encoder rerankers add several nDCG points on BEIR (RankGPT, Qwen3-Reranker), but one legal benchmark (LegalBench-RAG) reported a commercial reranker worse than none (sections 1 and 3).
3. **Abstention:** published work uses LLM generators and LLM judges (UAEval4RAG, Sufficient Context); the closest small-model work is RINSE (2026). ragkit's reader-based gate (AUROC 0.858 on 87 questions) is the same idea at a much smaller scale (section 5).
4. **No number in ragkit's abstention tables can be compared with a published number as it stands**: models, corpora, label sources and metrics all differ (section 7.3).
5. **Largest gaps in ragkit, by expected benefit:** neural retrieval, a sufficiency judge, a human-labelled test set (section 7b).

## 0. How to read this document

- Every source below was opened for this review on 2026-10-26 (arXiv abstract or PDF/HTML, vendor page, model card, or leaderboard backend). A link in a table is a page I read, not a page I was told about.
- "Date" is the arXiv v1 submission date unless I say otherwise. For vendor pages it is the published/modified date in the page metadata. "unknown" means the page did not show one.
- A number appears only if I read it in the cited place. Anything I did not read directly is marked `[INFERENCE]`. A number I could not find is written "not found".
- Numbers from different papers are not comparable unless the table says so: different corpora, metrics, candidate lists, model versions.
- Own = ragkit's measured numbers from `README.md` (EU corpus: 94 articles, 934 chunks; 42 answerable and 45 unanswerable questions, all labels agent-drafted and agent-reviewed, no human expert).

What I could not verify (checked and failed):

| Item | What happened |
|---|---|
| Live MTEB leaderboard tables | `leaderboard.mteb.org` renders its tables client-side; the page text I fetched has no rows. I read one model's score JSON from the leaderboard backend (`mteb-leaderboard-backend.hf.space`) and the model cards instead. I did **not** read the full ranking, so "leader" below means "the model whose own score record says rank 1 of 483 on MTEB(Multilingual, v2) on 2026-10-26", not a verified top-10 list. |
| MTEB English-v2 and Retrieval-only leaders | Not opened. A search-result snippet of the English page showed Qwen3-Embedding-8B at rank #5 with mean 75.22 `[INFERENCE: snippet, page not opened]`. |
| Original BM25F paper | Not needed; Pyserini's "multifield" baseline is the only BM25F-like number used here. |
| Self-RAG, Adaptive-RAG, LightRAG headline numbers | The abstracts have no number; I did not locate the table cell in the PDF text. Marked "not found". |
| Rank1 numeric table | PDF table text was too garbled to read per-system numbers; only the claim is reported. |

## 1. Chunking and context

| System | What it does | Headline result (dataset, metric, where I read it) | Source, date | Difference from ragkit |
|---|---|---|---|---|
| Contextual Retrieval (Anthropic) | An LLM writes a short context for each chunk situating it in its document; the text is prepended before embedding and before building the BM25 index | Top-20-chunk retrieval failure rate (1 minus recall@20), averaged over codebases, fiction, arXiv, science papers, embedding Gemini Text 004: 5.7% to 3.7% (contextual embeddings), 2.9% (plus contextual BM25), 1.9% (plus Cohere reranker, top-150 to top-20). Cost "$1.02 per million document tokens" with prompt caching. Page text: "If your knowledge base is smaller than 200,000 tokens ... just include the entire knowledge base in the prompt" | [anthropic.com/research/contextual-retrieval](https://www.anthropic.com/research/contextual-retrieval), published 2024-09-19, page modified 2026-10-18. Vendor evaluation, own data mix | ragkit's recursive chunks (500 characters) carry no LLM-written context. Own A2 to A4 (structured chunks that keep the article heading, plus heading-field BM25F, plus synonym expansion) did not help: A4 Recall@5 -0.060, CI includes 0 |
| Late chunking (Jina) | Embed the whole document with a long-context embedder, apply chunk boundaries after the transformer, before mean pooling; no training needed | Fixed 256-token chunks, jina-embeddings-v2-small, SciFact nDCG@10 64.2 (naive) to 66.1 (late). Column assignment from my parse of a flattened table `[INFERENCE]`; average cell not used | [arXiv 2409.04701](https://arxiv.org/abs/2409.04701), 2024-09-07 (PDF read) | ragkit has no long-context embedder; its dense leg is a hashing n-gram embedder |
| Summary-augmented chunking (SAC) | Prepend a document-level LLM summary to every chunk to stop retrieval from picking the wrong document | Names the failure "Document-Level Retrieval Mismatch" (DRM): standard RAG on ContractNLI had "DRM rates over 95%" in a pool of 362 documents (text near Fig. 2a). Generic summaries beat expert-guided ones (abstract) | [arXiv 2510.06999](https://arxiv.org/abs/2510.06999), 2025-10-08 | ragkit's corpus has the same shape (near-identical articles across acts), but ragkit labels at article level and measures no DRM rate |
| LegalBench-RAG chunking finding | Benchmark of span-level retrieval on legal contracts; compares naive 500-char chunks with LangChain recursive splitting, with and without Cohere rerank-english-v3.0 | 6,858 query-answer pairs over "79M characters", expert-annotated. Text says recursive splitting without reranker was best and "the performance of the Cohere Reranker was inferior compared to not using a reranker". See the caveat below | [arXiv 2408.10343](https://arxiv.org/abs/2408.10343), 2024-08-19 (PDF read) | ragkit's EU corpus is multi-act legislation, not contracts; ragkit's lexical reranker also did not beat plain BM25 (own: ties) |

Caveat on the LegalBench-RAG tables (my arithmetic, not the paper's claim). The paper's Table 4 (naive) lists "ALL" recall@64 76.39 and precision@1 2.40, but the four dataset rows above it average to recall@64 63.49 and precision@1 9.24. Table 5 (recursive, no rerank) is self-consistent: "ALL" recall@64 62.22, precision@1 6.41. By the recomputed means the naive chunker is ahead of the recursive one at both cutoffs, which contradicts the text. Treat the "recursive is best" claim as unverified until the authors' data are re-scored. [INFERENCE: possible typo in the "ALL" row of Table 4.]

Takeaway: all three context techniques attack the same problem ragkit's own diagnosis found (chunks lose the article heading, near-duplicate articles compete), but none was run in ragkit.

## 2. First-stage retrieval

### 2a. BM25 and the BEIR baselines

Numbers copied from the Pyserini BEIR regression page ([castorini.github.io/pyserini/2cr/beir.html](https://castorini.github.io/pyserini/2cr/beir.html); page HTTP Last-Modified 2026-10-25, a site rebuild, so the date of the numbers is unknown). Lucene BM25 with k1 0.9, b 0.4, Porter stemming and Lucene's English stop list; "flat" concatenates title and text into one field, "multifield" indexes them as two fields and adds their scores with equal weight (parameters and index construction from the Anserini regression docs and Kamalloo et al., arXiv 2306.07471 section 7.1, as collected in `docs/beir-published-numbers.md` section 2). nDCG@10 / recall@100, BEIR test qrels. These are the same datasets ragkit's public-benchmark runs use, so they are the like-for-like reference for a BM25 implementation.

| Dataset | BM25 flat | BM25 multifield | SPLADE++ ED | Contriever (MS MARCO) | BGE-base-en-v1.5 |
|---|---|---|---|---|---|
| trec-covid | 0.595 / 0.109 | 0.656 / 0.114 | 0.727 / 0.128 | 0.596 / 0.091 | 0.781 / 0.141 |
| nfcorpus | 0.322 / 0.246 | 0.325 / 0.250 | 0.347 / 0.285 | 0.328 / 0.301 | 0.373 / 0.337 |
| fiqa | 0.236 / 0.539 | 0.236 / 0.539 | 0.347 / 0.631 | 0.329 / 0.656 | 0.406 / 0.742 |
| arguana | 0.397 / 0.932 | 0.414 / 0.943 | 0.526 / 0.983 | 0.446 / 0.977 | 0.636 / 0.992 |
| scidocs | 0.149 / 0.348 | 0.158 / 0.356 | 0.159 / 0.373 | 0.165 / 0.378 | 0.217 / 0.496 |
| scifact | 0.679 / 0.925 | 0.665 / 0.908 | 0.704 / 0.935 | 0.677 / 0.947 | 0.741 / 0.967 |
| quora | 0.789 / 0.973 | 0.789 / 0.973 | 0.834 / 0.986 | 0.865 / 0.994 | 0.889 / 0.997 |
| webis-touche2020 | 0.442 / 0.582 | 0.367 / 0.538 | 0.246 / 0.471 | 0.204 / 0.442 | 0.257 / 0.487 |
| nq | 0.305 / 0.751 | 0.329 / 0.760 | 0.537 / 0.930 | 0.498 / 0.925 | 0.541 / 0.942 |

BM25 flat has the highest nDCG@10 of the five columns only on Touché-2020; on FiQA, ArguAna, NQ and SciDocs the base-size dense encoder (BGE-base) is ahead of it by 0.07 to 0.24 nDCG@10, on SciFact by 0.06. Other implementations of BM25 give slightly different numbers: the Jina-ColBERT-v2 paper lists SciFact 66.5, NFCorpus 32.5, FiQA 23.6, TREC-COVID 65.6, SciDocs 15.8 for its BM25 ([arXiv 2408.16672](https://arxiv.org/abs/2408.16672) Table 1). ragkit's own measured BEIR numbers on SciFact, NFCorpus, FiQA, ArguAna, TREC-COVID and Quora are in section 8 and in the README, "Comparison with SOTA".

BEIR itself: [arXiv 2104.08663](https://arxiv.org/abs/2104.08663), 2021-04-17: 18 datasets, "BM25 is a robust baseline", re-ranking and late-interaction models best on average zero-shot (abstract).

### 2b. Learned sparse, late interaction, single-vector dense

| System | What it does | Headline result (dataset, metric, where read) | Source, date | Difference from ragkit |
|---|---|---|---|---|
| SPLADE-v3 | Transformer produces a weighted bag of vocabulary terms including expansion terms; searched in an inverted index | MS MARCO dev MRR@10 40.2; mean nDCG@10 over 13 BEIR sets 51.7 (SPLADE++SelfDistil 50.7); per set: SciFact 71.0, NFCorpus 35.7, FiQA 37.4, ArguAna 50.9, SciDocs 15.8, TREC-COVID 74.8, Touché 29.3 (Table 2, column "SPLADE-v3") | [arXiv 2403.06789](https://arxiv.org/abs/2403.06789), 2024-03-11 | ragkit's synonym map (hand-written, 0.4 weight) is a manual stand-in; own: no separated gain (dMRR +0.012, CI [-0.044, +0.066]) |
| ColBERTv2 | One vector per token, residual compression, denoised supervision | Index "6 to 10x" smaller than ColBERT v1 (abstract). Mean nDCG@10 over the 13 BEIR datasets in the Jina-ColBERT-v2 paper's Table 1: 49.6 vs BM25 44.0 | [arXiv 2112.01488](https://arxiv.org/abs/2112.01488), 2021-12-02; Table 1 of [2408.16672](https://arxiv.org/abs/2408.16672) | No multi-vector index in ragkit |
| PLAID | Centroid interaction/pruning engine for ColBERTv2 | Latency down "up to 7x on a GPU and 45x on a CPU against vanilla ColBERTv2"; "tens of milliseconds on a GPU ... at 140M passages" (abstract) | [arXiv 2205.09707](https://arxiv.org/abs/2205.09707), 2022-05-19 | Efficiency work, irrelevant at ragkit's 934 chunks |
| Jina-ColBERT-v2 | Multilingual ColBERT, Matryoshka heads | Mean nDCG@10 over the 13 BEIR datasets of Table 1: 53.1 vs ColBERTv2 49.6, BM25 44.0, answerai-colbert-small 55.7. It is far below ColBERTv2 on ArguAna (36.6 vs 46.5), which the authors attribute to the task. (The paper's text says "14 tasks"; the table shows 13 datasets plus the average) | [arXiv 2408.16672](https://arxiv.org/abs/2408.16672), 2024-08-29 | Same as ColBERTv2 |
| E5 (v1) | Contrastive pre-training on web pairs, single vector | "first model that outperforms the strong BM25 baseline on the BEIR retrieval benchmark without using any labeled data" (abstract); unsupervised E5-PT-base +1.2 points over BM25 averaged over 15 datasets (Sect. results) | [arXiv 2212.03533](https://arxiv.org/abs/2212.03533), 2022-12-07 | ragkit's hashing embedder is not trained on text pairs at all |
| GTE | Multi-stage contrastive training, 30M to 330M | GTE-base (110M) "outperforms the black-box embedding API provided by OpenAI" on MTEB (abstract; no number read) | [arXiv 2308.03281](https://arxiv.org/abs/2308.03281), 2023-08-07 | same |
| Nomic Embed v1 | Open-data, 8192-token English embedder | MTEB 62.39 vs jina-embeddings-base-v2-en 60.39 (Table 1); beats text-embedding-ada-002 and -3-small (abstract) | [arXiv 2402.01613](https://arxiv.org/abs/2402.01613), 2024-02-02 | same |
| BGE-M3 | One model: dense, sparse and multi-vector, 100+ languages, 8192 tokens | MIRACL dev nDCG@10, average over languages: BM25 31.9, mE5-large 66.6, M3 dense 69.2, M3 sparse 53.9, all three combined 71.5 (Table 1) | [arXiv 2402.03216](https://arxiv.org/abs/2402.03216), 2024-02-05 | ragkit is English-only; the EU acts exist in 24 languages `[INFERENCE: not checked for this corpus]` |

### 2c. LLM-based embedders and the MTEB leaders as of 2026-10-26

| System | Headline result | Source, date | Note |
|---|---|---|---|
| Qwen3-Embedding-8B | "ranks No.1 in the MTEB multilingual leaderboard (as of June 5, 2025, score 70.58)". Model-card table, MTEB (Multilingual): mean(task) 70.58, retrieval 70.88. MTEB (Eng v2): mean 75.22, retrieval 69.44. Same table lists gemini-embedding-exp-03-07 at 68.37 / 67.71 (multilingual) and 73.3 / 64.35 (Eng v2) | [HF model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B/raw/main/README.md), snapshot 2025-06-05; paper [arXiv 2506.05176](https://arxiv.org/abs/2506.05176), 2025-06-05 | Card also says instructions give "1% to 5%" |
| Gemini Embedding | MTEB(Multilingual) mean(task) 68.32, retrieval 67.71; MTEB(Eng, v2) mean 73.30 (paper's first table) | [arXiv 2503.07891](https://arxiv.org/abs/2503.07891), 2025-03-10 | API only |
| Gemini Embedding 2 | "69.9 on MTEB multilingual and 84.0 on MTEB Code" (abstract); multimodal | [arXiv 2605.27295](https://arxiv.org/abs/2605.27295), 2026-05-26 | Version of MTEB not stated in the abstract `[INFERENCE]`; do not compare with Qwen's 70.58 |
| harrier-oss-v1-27b (Microsoft) | Model card: "state-of-the-art results on the Multilingual MTEB v2 benchmark as of the release date", score 74.3 (0.6B: 69.0, 270M: 66.5). Leaderboard backend score record: MTEB(Multilingual, v2) rank 1 of 483 models, mean(task) 74.27, retrieval task-type 78.27, release date 2026-03-27, 27.0B parameters, MIT licence | [model card](https://huggingface.co/microsoft/harrier-oss-v1-27b/raw/main/README.md); [leaderboard backend JSON](https://mteb-leaderboard-backend.hf.space/v1/models/microsoft/harrier-oss-v1-27b/scores), both read 2026-10-26 | The same record flags 78% zero-shot on that benchmark (share of tasks not seen in training, per the leaderboard's own field), so part of the lead may be training overlap `[INFERENCE]`. Qwen's 70.58 and harrier's 74.3 may sit on different MTEB versions `[INFERENCE: the Qwen card does not say v2]`; do not compare them |

What I can say about "who leads in 2026": the largest embedders are decoder LLMs of 7B to 27B parameters, and on MTEB(Multilingual, v2) the top-ranked model on 2026-10-26 is a 27B model. What I cannot say: the full top-10, the English-only leader, or whether any model has overtaken harrier since the snapshot. Two caveats from the sources themselves: (1) the leaderboard marks how much of each score is zero-shot; (2) the legal benchmark MLEB (section 6, as of 2025-10-21) ranks models differently from MTEB: there Qwen3-Embedding-8B is 4th, behind a legal-specialised model and two Voyage models. MTEB rank alone is therefore not a safe way to pick an embedder for a legal corpus.

Qwen3-Embedding-8B was rank 1 on the multilingual board on 2025-06-05 (its card). On 2026-10-26 the model ranked 1 in the leaderboard backend for the multilingual v2 board is a different one. The answer to "what is best" moves within months.

## 3. Rerankers

| System | What it does | Headline result | Source, date | Difference from ragkit |
|---|---|---|---|---|
| monoBERT | Cross-encoder over (query, passage) | "outperforming the previous state of the art by 27% (relative) in MRR@10" on MS MARCO passage ranking (abstract). On BEIR-8 average nDCG@10 over BM25 top-100: 47.16 vs BM25 43.42 (RankGPT Table 1) | [arXiv 1901.04085](https://arxiv.org/abs/1901.04085), 2019-01-13 | ragkit's neural reranker (MiniLM cross-encoder) is an optional extra and was not run on the reported EU table; lexical reranker ties BM25 |
| monoT5 | T5 emits "true/false", logit is the score | MS MARCO "at least on par" with BERT rerankers (abstract). BEIR-8 mean: 220M 49.07, 3B 51.36 (RankGPT Table 1) | [arXiv 2003.06713](https://arxiv.org/abs/2003.06713), 2020-03-14 | none run |
| RankGPT | Prompt an LLM with a window of passages, it outputs the permutation | BEIR-8 mean nDCG@10, reranking BM25 top-100: BM25 43.42, Cohere Rerank-v2 49.45, gpt-3.5-turbo 49.37, gpt-4 53.68 (gpt-4 reranks only the top-30 already reranked by gpt-3.5). Per set, gpt-4 vs BM25: TREC-COVID 85.51 vs 59.47, SciFact 74.95 vs 67.89, NFCorpus 38.47 vs 30.75 | [arXiv 2304.09542](https://arxiv.org/abs/2304.09542), 2023-04-19 (PDF, Table 1) | no LLM reranker |
| RankZephyr | Open 7B listwise reranker distilled from RankGPT-4 | "bridges the effectiveness gap with GPT-4 and in some cases surpasses" it on TREC DL and BEIR NEWS/COVID (abstract). Ablation row: DL19 nDCG@10 0.7374 with BM25 candidates, 0.7645 with ADA2 candidates | [arXiv 2312.02724](https://arxiv.org/abs/2312.02724), 2023-12-05 | none |
| FIRST | Listwise reranking from the logits of the first generated identifier only | Faster inference, ranking-aware loss (abstract; numbers not read) | [arXiv 2406.15657](https://arxiv.org/abs/2406.15657), 2024-06-21 | none |
| bge-reranker-v2-m3 and Qwen3-Reranker | Open cross-encoder / LLM pointwise rerankers | Qwen3-Reranker model card, reranking top-100 from Qwen3-Embedding-0.6B: MTEB-R (retrieval subsets of MTEB eng v2) 0.6B 65.80, 4B 69.76, 8B 69.02; BGE-reranker-v2-m3 57.03; first-stage Qwen3-Embedding-0.6B alone 61.82 | [Qwen3-Reranker-8B card](https://huggingface.co/Qwen/Qwen3-Reranker-8B/raw/main/README.md) (date unknown); [arXiv 2506.05176](https://arxiv.org/abs/2506.05176), 2025-06-05 | Vendor evaluation by the model's authors |
| Rank1 | Reasoning reranker distilled from DeepSeek-R1 traces (600k+ examples) | Claims "state-of-the-art performance on the BRIGHT benchmark ... NevIR ... mFollowIR"; numeric table not extracted | [arXiv 2502.18418](https://arxiv.org/abs/2502.18418), 2025-02-25 | none |
| Rank-R1 | RL (GRPO) on relevance labels only, no reasoning supervision | BRIGHT nDCG@10 average, reranking BM25 top-100 (Table 2): BM25 .137, RankZephyr-7b .130, RankGPT-4 .170 (copied from a run on a stronger BM25), Rank-R1-14B .205 | [arXiv 2503.06034](https://arxiv.org/abs/2503.06034), 2025-03-08 | none |
| ReasonIR-8B | Retriever trained on synthetic reasoning-intensive queries with hard negatives | BRIGHT nDCG@10 29.9 without reranker, 36.9 with reranker (abstract) | [arXiv 2504.20595](https://arxiv.org/abs/2504.20595), 2025-04-29 | none; reasoning-heavy queries are not what ragkit's near-miss set tests |

Evidence that a general reranker can hurt on legal text: LegalBench-RAG's own text (section 1). Evidence that LLM rerankers help on BEIR: RankGPT table above. Both are on other corpora.

## 4. Agentic, structured and long-context RAG

| System | What it does | Headline result | Source, date | Difference from ragkit |
|---|---|---|---|---|
| Self-RAG | Fine-tuned LM emits reflection tokens (retrieve? relevant? supported?) | 7B and 13B "significantly outperform" ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, reasoning, fact verification (abstract); number not found | [arXiv 2310.11511](https://arxiv.org/abs/2310.11511), 2023-10-17 | ragkit has no trained generator; its "supported?" step is the support-threshold rule |
| CRAG (corrective RAG) | A retrieval evaluator grades retrieved docs; triggers use, discard, or web search | Table 1, LLaMA2-hf-7b generator, PopQA acc / Bio FactScore / PubHealth acc / ARC-C acc: RAG 50.5 / 44.9 / 48.9 / 43.4; CRAG 54.9 / 47.7 / 59.5 / 53.7 | [arXiv 2401.15884](https://arxiv.org/abs/2401.15884), 2024-01-29 (PDF; header row of the table read) | Closest idea to ragkit's abstention gate, but the web fallback has no meaning for a closed legal corpus |
| Adaptive-RAG | Classifier routes a query to no retrieval, single-step, or iterative retrieval by complexity | "enhances the overall efficiency and accuracy" on open-domain QA (abstract); number not found | [arXiv 2403.14403](https://arxiv.org/abs/2403.14403), 2024-03-21 | No router; the EU set has 8 multi-hop of 42 questions |
| RAPTOR | Recursively cluster and summarise chunks into a tree; retrieve at any level | "improve the best performance on the QuALITY benchmark by 20% in absolute accuracy" with GPT-4 (abstract) | [arXiv 2401.18059](https://arxiv.org/abs/2401.18059), 2024-01-31 | No LLM summarisation. GridCodex (below) uses RAPTOR on grid codes |
| GraphRAG (Microsoft) | LLM builds an entity graph and community summaries; map-reduce over summaries | "substantial improvements over a conventional RAG baseline" in comprehensiveness and diversity for global questions at about 1M tokens (abstract; LLM-judged) | [arXiv 2404.16130](https://arxiv.org/abs/2404.16130), 2024-04-24 | ragkit has no corpus-wide "themes" questions |
| LazyGraphRAG | Defers all LLM use to query time; NLP noun-phrase graph | Indexing cost "identical to vector RAG and 0.1% of the costs of full GraphRAG"; ">700 times lower query cost" than GraphRAG global search at comparable quality; 5,590 AP articles, 100 synthetic queries, LLM-judged head-to-head | [Microsoft Research blog](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/), published 2024-11-25, modified 2025-06-06. Vendor evaluation | same |
| LightRAG | Graph index with dual-level (entity and theme) retrieval, incremental update | "considerable improvements in retrieval accuracy and efficiency" (abstract); number not found | [arXiv 2410.05779](https://arxiv.org/abs/2410.05779), 2024-10-08 | same |
| HippoRAG | Knowledge graph plus personalised PageRank, hippocampus-inspired | Single-step retrieval "comparable or better ... than iterative retrieval like IRCoT while being 10-30 times cheaper and 6-13 times faster"; multi-hop QA improvement "up to 20%" (abstract, sentence start cut off in my copy) | [arXiv 2405.14831](https://arxiv.org/abs/2405.14831), 2024-05-23 | same |
| GraphRAG-Bench | When do graphs pay off | "GraphRAG frequently underperforms vanilla RAG on many real-world tasks" (abstract) | [arXiv 2506.05690](https://arxiv.org/abs/2506.05690), 2025-06-06 | supports ragkit not adding a graph |
| EffiRAG | Cheap graph RAG; graph only locates passages | Preferred over LightRAG-hybrid on 93 of 120 UltraDomain questions (abstract) | [arXiv 2609.18099](https://arxiv.org/abs/2609.18099), 2026-09-16 | same; shows graph cost vs quality is still contested in 2026 |
| Search-R1 | RL-trained LLM issues multiple search queries while reasoning | Abstract on the arXiv listing: "41% (Qwen2.5-7B) and 20% (Qwen2.5-3B)" over RAG baselines on seven QA sets; the abstract in the current PDF says "24% (Qwen2.5-7B)". The introduction in the same PDF still says 41%. Cite either with the version | [arXiv 2503.09516](https://arxiv.org/abs/2503.09516), 2025-03-12 | RL training; ragkit is retrieve-once |
| Long context vs RAG | Put the whole corpus in the prompt; Self-Route sends easy queries to RAG | "when resourced sufficiently, LC consistently outperforms RAG in terms of average performance"; Self-Route cuts cost "by 65% for Gemini-1.5-Pro and 39% for GPT-4O" at comparable performance | [arXiv 2407.16833](https://arxiv.org/abs/2407.16833), 2024-07-23 | ragkit has no full-context baseline. Its corpus is 337 kB, about 85k tokens `[INFERENCE at 4 chars/token]`, under Anthropic's 200k-token line, so "no retrieval" is an available control |

## 5. Abstention and sufficiency

### 5a. The two papers ragkit is closest to

| | UAEval4RAG | Sufficient context |
|---|---|---|
| Source, date | [arXiv 2412.12300](https://arxiv.org/abs/2412.12300) (PDF v3 read), v1 2024-12-16, v3 2025-04-21; ACL 2025 long papers ([aclanthology.org/2025.acl-long.415](https://aclanthology.org/2025.acl-long.415/), pages 8452 to 8472 per the search listing) | [arXiv 2411.06037](https://arxiv.org/abs/2411.06037) (PDF v3 read), v1 2024-11-09, v3 2025-04-23; "Published as a conference paper at ICLR 2025"; Google blog 2025-05-14 ([link](https://research.google/blog/deeper-insights-into-retrieval-augmented-generation-the-role-of-sufficient-context)) |
| Task | Given a knowledge base, synthesise unanswerable requests in six types (underspecified, false presupposition, nonsensical, modality-limited, safety-concerned, out-of-database) and score whether the RAG system rejects them | Classify whether (query, retrieved context) is enough to answer; use that label to analyse errors and to gate answering |
| Metrics | Unanswered ratio, acceptable ratio, ask-for-clarification ratio, plus a joint score w1 x correctness + w2 x acceptable ratio with w1 0.7, w2 0.3. All LLM-labelled | Autorater accuracy vs human labels; correct / abstain / hallucinate shares by sufficiency; selective accuracy vs coverage |
| Validation of the pipeline | 100 generated requests per dataset checked by three authors: 92% accuracy on TriviaQA and MuSiQue, agreement 0.85 and 0.88. LLM labellers vs author labels on 150 pairs: accuracy 81.3% to 84.4% (Table 1) | Autorater on 115 hand-labelled (query, context) pairs from PopQA, FreshQA, NQ, EntityQuestions: Gemini 1.5 Pro 1-shot accuracy 0.930, F1 0.935; FLAMe 24B accuracy 0.878; fine-tuned T5 entailment (TRUE-NLI) 0.826 (Table 1) |
| Systems tested | 3 embedders, 3 retrievers (vector, BM25, vector+RAPTOR ensemble), Cohere and GPT-4 rerankers, HyDE and multi-step rewriting, GPT-4o / Claude 3.5 Sonnet / Gemini Pro, three prompts; knowledge bases are Wikipedia dumps for TriviaQA, MuSiQue, 2Wiki, HotpotQA | Gemini 1.5 Pro, GPT-4o, Claude 3.5 Sonnet, Gemma 27B, Mistral 3; FreshQA (452), Musique-Ans (500 sampled), HotpotQA (500 sampled) |
| Headline numbers | TriviaQA, ensemble retriever, Table 4, all six types: acceptable ratio GPT-4o 52.5%, Claude 3.5 Sonnet 77.0%, Gemini Pro 51.0%; "Underspecified" is the hardest type (GPT-4o 17.0%). Table 5 (ada-002 vector retriever, GPT-4): acceptable ratio TriviaQA 49.0%, MuSiQue 56.8%, 2Wiki 61.5%, HotpotQA 61.6%. Restrictive prompt #1 on TriviaQA (BM25, Cohere rerank, HyDE): acceptable ratio 53.2% to 84.3% with correctness 88.0% to 88.4% (Table 3). Text: "the best prompt boosting unanswerable query performance by about 80%" | 55.4% of Musique-Ans instances have insufficient context (Fig. 1). Claude 3.5 Sonnet abstains on 84.1% of questions without RAG and 52% with RAG; Gemini 1.5 Pro 100% to 18.6%. With insufficient context, SOTA models are still correct "35 to 62%" of the time. Selective generation (autorater label + self-rated confidence, logistic regression) raises the correct share of answered questions by 2 to 10%; "over 5% for Gemini 1.5 Pro" near 70% coverage on HotpotQA |
| Main limit stated by authors or visible in text | Synthetic requests; LLM metrics; single-turn only (their Limitations) | Binary label only; no study of how retrieval choice changes sufficiency (their Limitations); correct-with-insufficient-context is common, so abstaining on every "insufficient" label costs accuracy |

### 5b. Other abstention and uncertainty work

| System | What it does | Headline result | Source, date | Difference from ragkit |
|---|---|---|---|---|
| RGB | Four abilities tested per LLM: noise robustness, negative rejection, information integration, counterfactual robustness | "The highest rejection rates for LLMs in English and Chinese were only 45% and 43.33%" (six LLMs) | [arXiv 2309.01431](https://arxiv.org/abs/2309.01431), 2023-09-04 (PDF) | Rejection is tested with irrelevant documents; ragkit's unanswerables are on-topic near-misses |
| R-Tuning | Refusal-aware instruction tuning: fine-tune the model to say "I don't know" for questions outside its parametric knowledge | Abstract describes the method; number not read | [arXiv 2311.09677](https://arxiv.org/abs/2311.09677), 2023-11-16 | Parametric knowledge, not retrieval sufficiency |
| Trust-Align | Aligns a generator with a "Trust-Score" covering refusal and citation quality | Trust-Aligned LLaMA-3-8b reaches Trust-Score 54.85%, above GPT-4's 54.69% (text near Table 7) | [arXiv 2409.11242](https://arxiv.org/abs/2409.11242), 2024-09-17 | Needs fine-tuning |
| CRAG benchmark | 4,409 QA pairs, 5 domains, 8 question types, mock web/KG APIs | "advanced LLMs achieve at most 34% accuracy ... adding RAG ... only to 44%. State-of-the-art industry RAG solutions only answer 63% of questions without any hallucination" | [arXiv 2406.04744](https://arxiv.org/abs/2406.04744), 2024-06-07 | General web facts |
| Semantic entropy | Cluster sampled answers by meaning, compute entropy over meanings as an uncertainty score | Beats predictive entropy and lexical similarity in AUROC on CoQA and TriviaQA (Figs. 1 and 2; numeric values not extracted) | [arXiv 2302.09664](https://arxiv.org/abs/2302.09664), 2023-02-19 | Needs sampled generations; ragkit has no generator |
| Selective QA under domain shift | Train a calibrator on features of the QA model and its input to decide when to abstain | "answers 56% of questions while maintaining 80% accuracy; ... using the model's probabilities only answers 48%" (SQuAD plus five other datasets) | [arXiv 2006.09462](https://arxiv.org/abs/2006.09462), 2020-06-16 | ragkit's nested-CV logistic combination is the same recipe on 9 to 11 cheap signals |

### 5c. 2026 work found by search (all opened)

| System | What it does | Headline result | Source, date | Relevance to ragkit |
|---|---|---|---|---|
| RINSE ("Relevance Is Not Sufficient Evidence") | Scores sufficiency from question plus evidence only, before any answer exists: question-part coverage, whether any passage offers an answer, and a small LM's sufficiency judgement; equal-weight sum corrected for evidence-set size | "Even when instructed to abstain, 12 generators answer 40.0-99.3% of insufficient-evidence questions." RINSE averages 0.837 over six datasets vs Self-RAG 0.746 and UPR 0.721; 36.5 ms per question on one GPU. The metric name is not stated in the text I read `[INFERENCE: likely a ranking metric such as AUROC]` | [arXiv 2609.37469](https://arxiv.org/abs/2609.37469), 2026-09-27 (HTML read) | Same idea as `qa_answerable`, with a named design. Their stated pitfall matters for ragkit: tests built by "removing relevant evidence or pairing evidence with unrelated questions can reveal labels through lexical overlap or evidence position" |
| GRAB-RAG | Paired benchmark: same question under supportive, degraded, missing, misleading context (NQ, HotpotQA), three frozen 3.8B to 8B models | Under explicit abstention prompting models "still answer 41.6% of misleading questions, with 63% of those answers echoing the planted wrong entity"; a conflict check cuts this to 13.3% but "discards many correct answers"; "Prompt-based abstention asks whether context is sufficient, not whether it is correct" | [arXiv 2608.22228](https://arxiv.org/abs/2608.22228), 2026-08-23 (PDF) | ragkit has no misleading-context type; its `wrong_regulation` type is the nearest |
| DeclineCraft | Abstention benchmark whose unanswerable questions target Minecraft features added after a corpus cut-off; 27 LLMs | Single-shot retrieval raises correct-abstention rate in 38 of 40 configurations vs no retrieval; iterative RAG lowers it in 30 of 40 (mean 46.4%) with accuracy 68.5% to 69.3%; adding rewriting and reranking to hybrid retrieval lowers it for all 12 models tested | [arXiv 2610.28340](https://arxiv.org/abs/2610.28340), 2026-10-20 (HTML read) | Supports the UAEval4RAG finding that rewriting and reranking can reduce rejection of unanswerables |

Honest read of this section: abstention for RAG is a 2024 to 2026 area with no standard benchmark; the leading designs use an LLM as generator or judge. Only RINSE is a small pre-generation scorer comparable in spirit to ragkit's model signals.

## 6. Legal and regulatory RAG

| System | What it does | Headline result | Source, date | Difference from ragkit |
|---|---|---|---|---|
| LegalBench-RAG | Span-level retrieval benchmark from LegalBench queries; also a "mini" version | See section 1 | [arXiv 2408.10343](https://arxiv.org/abs/2408.10343), 2024-08-19 | Contracts, single-document retrieval |
| ObliQA / RIRAG (ADGM financial rules) | 27,869 questions over Abu Dhabi Global Market rules, with RePASs for generated answers (obligations covered, contradictions) | Baseline findings: models trained only on MS MARCO are "worse or on-par" with BM25 on recall@10 and MAP@10; ColBERTv2 is +1.1 recall@10 over BM25; NV-Embed-v2 and BGE-EN-ICL beat BM25 "significantly". Separate 2026 paper on ObliQA: recall@10 BM25 0.678, E5-large-v2 0.758, its staged LegalBERT retriever fused with BM25 0.774 (the staged tuning raised its starting model from 0.256, per the abstract) | [arXiv 2409.05677](https://arxiv.org/abs/2409.05677), 2024-09-09; [arXiv 2609.30009](https://arxiv.org/abs/2609.30009), 2026-09-24 (abstract) | Same "regulation rulebook" shape as ragkit, English, public; has no unanswerable split |
| CLERC | Case-law retrieval and generation from citations | "zero-shot IR models only achieve 48.3% recall@1000"; GPT-4o "hallucinates the most" (abstract) | [arXiv 2406.17186](https://arxiv.org/abs/2406.17186), 2024-06-24 | US case law, not statutes |
| MLEB | Ten expert-annotated legal retrieval sets across US, UK, EU, Australia, Ireland, Singapore | As of 2025-10-21, Kanon 2 Embedder first at NDCG@10 86.03 (task average) of 21 models; Voyage 3 Large 85.71; Qwen3-Embedding-8B 82.96; Gemini Embedding 80.90. On the "Regulatory" domain average: Kanon 2 91.48, Gemini Embedding 91.13, Qwen3-Embedding-8B 88.99, Voyage 3.5 88.83 | [arXiv 2510.19365](https://arxiv.org/abs/2510.19365), 2025-10-22 (Table 2) | I did not check whether the benchmark authors are affiliated with the top model's vendor. No EU electricity set inside |
| Stanford legal-research tools study | Preregistered test of Lexis+ AI, Westlaw AI-Assisted Research, Ask Practical Law AI | These tools "each hallucinate between 17% and 33% of the time" (abstract) | [arXiv 2405.20362](https://arxiv.org/abs/2405.20362), 2024-05-30 | Shows that vendor RAG does not remove hallucination; ragkit's extractive generator avoids it by construction but answers only by copying sentences |
| Large Legal Fictions | Hallucination rates of general LLMs on legal queries | "between 58% of the time with ChatGPT 4 and 88% with Llama 2" (abstract) | [arXiv 2401.01301](https://arxiv.org/abs/2401.01301), 2024-01-02 | Closed-book, motivates retrieval |
| GridCodex | RAG for power-grid-code compliance with multi-stage query refinement and RAPTOR | "26.4% improvement in answer quality and more than a 10 fold increase in recall rate" (abstract, baseline not defined in text I read) | [arXiv 2508.12682](https://arxiv.org/abs/2508.12682), 2025-08-18 | Closest domain to ragkit (electricity regulation). No abstention evaluation seen in the abstract |
| Financial-compliance RAG with domain-adapted retriever | LegalBERT entailment tuning, contrastive tuning, BM25 fusion; 2B to 12B generators with RAFT-LoRA | A closed-book model with no passages scores "within 0.011 RePASs of the full pipeline while producing answers that cite nothing and misstate obligations"; adapted models do not transfer to Australian case law | [arXiv 2609.30009](https://arxiv.org/abs/2609.30009), 2026-09-24 (abstract) | Shows an answer-quality metric can miss grounding; ragkit measures grounding by construction, not answer quality |

EU-specific work: I found no public EU-electricity-regulation RAG benchmark. The searches ("EUR-Lex benchmark question answering", "energy sector regulation RAG") returned survey papers and EUR-Lex documents themselves; the nearest domain hit is GridCodex. MLEB includes EU material but no EU electricity set. This is a negative search result, not proof of absence.

## 7. Positioning: ragkit's EU-regulation abstention work

### 7.1 What ragkit measured

- Task: for a question and its retrieved top-5 chunks, decide answer vs abstain. 42 answerable questions (17 direct, 17 paraphrase, 8 multi-hop) vs 45 on-topic unanswerable questions in four near-miss types (wrong_regulation 9, absent_article 8, missing_number 9, national_rule 7, plus 12 originals). Own, `README.md`.
- Metrics: AUROC of a score (P[answerable scores higher than unanswerable], ties 1/2) with bootstrap CIs; paired AUROC differences; held-out balanced accuracy at a threshold chosen on training folds; risk-coverage ("unanswerable share of answered"). Nested 5-fold CV for the learned combination; five CV seeds.
- Own AUROC, all 45 vs 42: `query_support` (term coverage) 0.587 [0.463, 0.702]; `dense_top` 0.679 [0.558, 0.794] (the best of the nine cheap signals, picked after seeing the 12-question table); nine-signal logistic combination 0.589 [0.467, 0.709]; cross-encoder `ce_top` 0.724 [0.605, 0.826]; SQuAD2-reader `qa_answerable` 0.858 [0.772, 0.928], +0.272 [+0.150, +0.393] over term coverage; two-model combination 0.836; on all 11 signals 0.839.
- Own operating point: `qa_answerable` answers 37/42 answerable and abstains on 32/45 unanswerable (balanced accuracy 0.796); 13 of 45 unanswerable questions are still answered. Risk-coverage at 70% coverage: unanswerable share of answered 0.344 vs 0.517 with no abstention.
- Own controls: scoring each question against the chunks of the next question of its class gives `qa_answerable` 0.752 [0.651, 0.845], so most of its separation (0.752 of 0.858) survives with the wrong passages; matched minus mismatched is +0.106 [-0.015, +0.228]. `ce_top` falls to 0.601 [0.472, 0.725] under the same control. Question length alone: 0.602 [0.48, 0.73]. On the synthetic mostly off-domain set `query_support` is 0.987, so the domain shift is large.
- Cost: about 0.06 s (`ce_top`) and 0.09 s (`qa_answerable`) per question on a laptop CPU.

### 7.2 Same task or not

| Aspect | ragkit (EU abstention) | UAEval4RAG | Sufficient context |
|---|---|---|---|
| What is decided | Answer or abstain before generation, from retrieval and reader signals | Whether the full RAG system's response rejects a synthesised unanswerable request | Whether the context suffices; optionally gate generation |
| Unanswerable data | 45 hand-drafted on-topic near-misses, agent-reviewed; each tied to closest articles | 600 LLM-synthesised requests across six types, checked by LLM verification plus three authors; out-of-database type built from fresh news | None synthesised; sufficiency labels on existing benchmark instances |
| Answerable data | 42 agent-drafted questions with labelled relevant articles | 500 sampled QA pairs from TriviaQA, MuSiQue, 2Wiki, HotpotQA | FreshQA, Musique-Ans, HotpotQA samples |
| Corpus | 94 EU articles, English, 934 chunks | Wikipedia dumps (tens of thousands of passages `[INFERENCE: size not read]`) | Dataset-supplied or retrieved contexts, up to 6,000 tokens |
| Model in the loop | Small encoders (MiniLM cross-encoder, 124M SQuAD2 RoBERTa-base) `[INFERENCE: sizes from model names]`; no generator | GPT-4, GPT-4o, Claude 3.5 Sonnet, Gemini Pro as generator and as labeller | Gemini 1.5 Pro as autorater; FLAMe 24B as cheaper autorater; large generators |
| Metric | AUROC, paired AUROC difference, balanced accuracy, risk-coverage | Unanswered ratio, acceptable ratio, joint score | Autorater accuracy/F1; selective accuracy vs coverage |
| Uncertainty reporting | Bootstrap CIs, Wilson CIs, 5 CV seeds | Single numbers, no CIs in the tables I read | Curves, no CIs in the text I read |

Same task in spirit: both ask whether a RAG system should have answered. Different in what is measured: ragkit scores a pre-generation gate by ranking quality (AUROC); UAEval4RAG scores a finished system's behaviour by LLM-judged ratios; sufficient-context scores a context label and uses it inside a selective-accuracy trade-off.

### 7.3 Which numbers could in principle be compared, and why none can today

| Their number | Comparable to which ragkit number | What would be needed | Why not comparable now |
|---|---|---|---|
| UAEval4RAG "unanswered ratio" / "acceptable ratio" for out-of-database requests (TriviaQA: 52.5% to 77.0% acceptable over all types, by LLM) | ragkit abstain rate on unanswerable (32/45 = 0.71 at the CV threshold for `qa_answerable`) | Run ragkit's gate on UAEval4RAG-generated requests, or run an LLM RAG system on ragkit's 45 and judge with their labeller | Different knowledge base (Wikipedia vs EU law), different decision-maker (an LLM's own refusal vs an external gate), "acceptable" is a six-type LLM-judged criterion, not abstention alone. Their six types include nonsense and safety requests that ragkit never asks |
| UAEval4RAG joint score 0.7 x correctness + 0.3 x acceptable | ragkit "answered 37/42 and abstained 32/45" | Define correctness for ragkit (it has no LLM answers; on the synthetic corpus its extractive answer scores about 0.23 token-overlap F1) | ragkit's generator is extractive and its correctness metric is not answer correctness |
| Sufficient context autorater accuracy 0.930 on 115 pairs | `qa_answerable` AUROC 0.858 and balanced accuracy 0.796 | Same labelled (question, chunks) pairs scored by both | Their label is "context sufficient", ragkit's is "question answerable from the corpus"; for ragkit's near-misses these coincide only if retrieval found the right article `[INFERENCE]`; different scale of label set (115 vs 87) and metric (accuracy vs AUROC) |
| Selective accuracy vs coverage (Fig. 4: HotpotQA, Gemini 1.5 Pro, +5% near 70% coverage) | ragkit risk-coverage at 70%: unanswerable share of answered 0.344 | Both reported as accuracy of answered questions at matched coverage | Their "accuracy" is answer correctness by an LLM grader; ragkit's "bad share" counts unanswerable-answered plus retrieval misses; the base rates differ (ragkit 51.7% unanswerable at full coverage) |
| RINSE average 0.837 over six datasets | ragkit `qa_answerable` 0.858 AUROC | Their metric definition and their datasets | Metric not identified in the text I read; datasets are not near-miss unanswerables from one corpus |
| RGB negative rejection 45% / 43.33% | ragkit abstain rate on unanswerable | Different LLMs on irrelevant-document noise | Different model class and test design |

Conclusion: none of these can be compared as stated. The honest comparison is to re-run: either ragkit's gate on a public set (the UAEval4RAG PDF footnote gives github.com/SalesforceAIResearch/Unanswerability for its generator; I did not open the repository) or a published system on ragkit's 87 questions. Neither has been done.

### 7.4 Where ragkit is weaker (stated plainly)

1. **Tiny, agent-labelled set.** 42 + 45 questions; AUROC CIs about 0.24 wide on the cheap signals; differences below roughly 0.05 to 0.15 cannot be resolved. No human domain expert saw any label.
2. **Labels were filtered against retrieval-style signals.** The 33 added questions were checked against BM25, dense and hybrid top-10 lists (README caveat). That favours neural readers over lexical signals and may overstate `qa_answerable`'s advantage on questions written by other people.
3. **Much of the best signal is question phrasing.** The mismatched control leaves `qa_answerable` at 0.752. RINSE's authors warn that such constructions "can reveal labels"; ragkit's control is the right check, but its result says the reader partly recognises how SQuAD2-style unanswerable questions are written `[INFERENCE]`.
4. **No LLM generator.** The extractive generator copies corpus sentences, so faithfulness is 1.000 by construction (synthetic-corpus pipeline table) and Answer F1 there is about 0.23. Every published abstention number above is measured on LLM-generated answers.
5. **No sufficiency autorater.** The one component that sufficient-context shows matters (a judge reading question plus context) is replaced by a 124M-class reader and a 22M-class cross-encoder `[INFERENCE on sizes]`.
6. **One corpus, one language, original (unconsolidated) Official Journal texts.** No amendments, no cross-act reasoning beyond 8 multi-hop questions.
7. **Two models, one aggregation.** Max over five chunks; nothing else was tried. No independent test set: all 87 questions feed every number.
8. **No neural retrieval in the reported tables.** "Dense retrieval does not help" is true only of the hashing embedder.

What ragkit does that the cited work mostly does not: paired bootstrap CIs and multi-seed nested CV on every abstention number; a matched/mismatched control; per-miss-type breakdown on near-miss unanswerables from a single closed corpus; pre-registered protocols with commit hashes; and a model-free-first ordering that shows how weak lexical signals are on on-topic unanswerables (AUROC 0.587).

## 7b. What ragkit does not have

Ranked by expected benefit, each tied to evidence in this document. "Expected" is my judgement, `[INFERENCE]`; none was tested in ragkit.

| Rank | Missing component | Evidence for benefit | Evidence against or limit | Rough cost |
|---|---|---|---|---|
| 1 | A neural first-stage retriever (BGE-M3, E5 or Qwen3-Embedding) fused with BM25, alpha tuned on held-out labels | Pyserini BEIR: BGE-base vs BM25 flat nDCG@10 +0.170 FiQA, +0.239 ArguAna, +0.068 SciDocs, +0.062 SciFact (section 2a). ObliQA 2026 paper: recall@10 BM25 0.678 vs E5-large-v2 0.758. Own: paraphrase MRR BM25 0.400, hashing dense 0.197; term overlap with the answer chunk 0.259 for paraphrase questions | Touché-2020: BM25 0.442, BGE-base 0.257. ObliQA baselines: MS-MARCO-only models are "worse or on-par" with BM25. Own label review used BM25 and dense lists, which can favour BM25-like systems | Minutes to embed 934 chunks `[INFERENCE]`; a new question set is needed to avoid re-using the 42 |
| 2 | A sufficiency judge (small LLM or fine-tuned model reading question plus context), used as a gate | Joren et al.: autorater accuracy 0.930; models "often output incorrect answers instead of abstaining"; gate gives +2 to 10% on answered. RINSE: 12 generators answer 40.0 to 99.3% of insufficient-evidence questions when told to abstain. Own: `qa_answerable` still answers 13/45 | Sufficiency is not correctness (GRAB-RAG: misleading context still gets answered 41.6% of the time); autorater needs a strong LLM or training | One LLM call per question; labelling effort for a calibration set |
| 3 | A human-labelled evaluation set and one public benchmark run | Every ragkit abstention number rests on 87 agent-reviewed labels; RINSE and the 2026 benchmarks show construction cues leak labels | none; this is a method fix | A domain expert's days, not compute |
| 4 | Contextual or summary-augmented chunks (act + article title at least, LLM summary at best) | Anthropic: top-20 failure 5.7% to 2.9% with contextual embeddings plus BM25; SAC: DRM above 95% on ContractNLI without it | Anthropic found generic summaries "very limited gains"; own A4 (heading-keeping chunks, heading-field BM25F, synonym expansion) lowered Recall@5 by 0.060 (CI includes 0), though the README's post-hoc diagnosis blames the heading field, not chunking | One cheap LLM call per chunk; $1.02 per million doc tokens (vendor figure) |
| 5 | A neural reranker (bge-reranker or Qwen3-Reranker) over top-50 to top-100 | RankGPT table: monoT5-3B BEIR-8 mean 51.36 vs BM25 43.42; Qwen3-Reranker-4B MTEB-R 69.76 vs first stage 61.82 | LegalBench-RAG: Cohere reranker was worse than none; own lexical reranker ties BM25 | CPU feasible for MiniLM-size; GPU for the rest |
| 6 | An LLM generator with a citation or NLI checker | Faithfulness is only "1.000 by construction"; Stanford study: commercial legal RAG hallucinates 17% to 33% | Larger system, new failure modes (CRAG benchmark: best industry answered 63% without hallucination) | API or GPU, plus a checker |
| 7 | Learned sparse (SPLADE-v3) or late interaction (ColBERTv2, Jina-ColBERT-v2) | SPLADE++ ED vs BM25 flat: FiQA +0.111, ArguAna +0.129, TREC-COVID +0.132; Jina-ColBERT-v2 BEIR 53.1 vs BM25 44.0 | Touché: SPLADE++ 0.246 vs BM25 0.442; adds infrastructure for a 934-chunk corpus | GPU encode; index size |
| 8 | Graph or tree structure (GraphRAG family, RAPTOR) | Good on global "themes" questions (GraphRAG, LazyGraphRAG) | GraphRAG-Bench: frequently underperforms vanilla RAG; ragkit has no global questions | LLM over the whole corpus; skip |

## 8. Head-to-head on public benchmarks

Measured by `ragkit beir` and `ragkit legalbench`, no model run or downloaded; protocols in `docs/beir_protocol.md` and `docs/legalbench_protocol.md`, raw output in `docs/sota-runs/`, full tables, sources and the "not comparable" list in the README section "Comparison with SOTA". Published rows with URL, table and date: `docs/beir-published-numbers.md`, `docs/legalbench-rag-published-numbers.md`.

Mean nDCG@10 over SciFact, NFCorpus, FiQA, ArguAna, TREC-COVID, Quora (SCIDOCS not run):

| System | Mean nDCG@10 |
|---|---|
| ragkit `bm25-lucene` / Pyserini BM25 flat | 0.503 / 0.503 |
| ragkit `bm25-ragkit` (default, no stemming) | 0.501 |
| ragkit lexical rerank (`rerank-lucene`) | 0.410 |
| Contriever MS MARCO / ColBERTv2 / SPLADE++ ED | 0.540 / 0.573 / 0.581 |
| BGE-base-en-v1.5 (Pyserini) | 0.638 |
| NV-Embed-v2 / Qwen3-Embedding-8B (MTEB) | 0.731 / 0.742 |

LegalBench-RAG-mini Recall@8, ALL: ragkit BM25 9.89 (Lucene-matched 13.37), paper's dense text-embedding-3-large baseline 31.86 (mean of its four printed dataset rows).

Reading: the Lucene-matched BM25 reproduces Pyserini within 0.007 on every dataset, so the implementation is sound; ragkit's retrieval quality is a BM25 baseline and trails every dense or learned-sparse system listed above on BEIR mean nDCG@10, and the dense baseline on LegalBench-RAG-mini.
