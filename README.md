# ragkit

Retrieval-augmented generation built from first principles and benchmarked on two corpora: a self-written, near-saturated energy-flexibility corpus (25 docs, 40 labelled questions), and a harder real-text corpus of 94 EU electricity-regulation articles on grid congestion and balancing (42 agent-drafted, second-pass agent-reviewed questions with no human review, see [EU corpus](#real-corpus-eu-electricity-regulation-congestion-and-balancing)).

## Problem
Which retrieval stack actually finds the right document? ragkit implements every stage with no retrieval libraries (BM25, hashing embeddings, fusion, reranking, rewriting, metrics) so each stage can be measured in isolation, offline and deterministically.

## Architecture
```mermaid
flowchart LR
  D[Markdown corpus] --> C[chunking: fixed / recursive / sentence / semantic]
  C --> B[BM25 Okapi k1=1.5 b=0.75]
  C --> E[HashingEmbedder char 3-5 grams + TF-IDF + L2] --> V[numpy cosine index]
  Q[query] --> R[rewrite: HyDE / multi-query / decomposition]
  R --> B & V
  B & V --> F[fusion: RRF k=60 / weighted min-max a=0.5]
  F --> K[LexicalReranker: coverage + min-span + BM25]
  K --> S{support ≥ 0.5?}
  S -->|yes| G[extractive generator with chunk_id citations]
  S -->|no| A[abstain]
```
Optional extras: `[st]` swaps in `all-MiniLM-L6-v2` embeddings and the `ms-marco-MiniLM-L-6-v2` cross-encoder; `[llm]` adds `AnthropicLLM` (needs `ANTHROPIC_API_KEY`) for rewriting.

## Quickstart
```bash
uv sync
uv run ragkit index src/ragkit/data/corpus     # writes .ragkit/index.json
uv run ragkit ask "What is imbalance settlement?" --index .ragkit/index.json
uv run ragkit ask "What is imbalance settlement?"              # indexes the bundled corpus on the fly
uv run ragkit eval --strategies all                             # or a subset: bm25,dense,hybrid,rerank
uv run ragkit abstention                                        # refuse-on-low-confidence check
uv run ragkit ablation                                          # staged end-to-end table
uv run ragkit eval --corpus eu                                  # same commands on the EUR-Lex corpus
uv run ragkit abstention --corpus eu
uv run ragkit ablation --corpus eu
uv run ragkit abstention --corpus eu --signals all               # AUROC / risk-coverage / CV and zero-shot thresholds per signal
uv run ragkit abstention --corpus synthetic --signals all        # same table on the synthetic set, for contrast
# any of eval / abstention / ablation also take --corpus-dir, --qa-file, --unanswerable-file
```

## Results
`ragkit eval --strategies all` (recursive chunks of 500 chars, 50 overlap, offline defaults; mean [bootstrap 95% CI over the 40 questions, 2000 resamples]):

| Strategy | Rewrite | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25 | None | 1.000 [1.000, 1.000] | 0.940 [0.877, 0.988] | 0.955 |
| BM25 | HyDE | 0.975 [0.925, 1.000] | 0.935 [0.871, 0.988] | 0.951 |
| Dense | None | 0.975 [0.925, 1.000] | 0.918 [0.844, 0.981] | 0.938 |
| Dense | HyDE | 0.700 [0.550, 0.850] | 0.513 [0.389, 0.646] | 0.581 |
| Hybrid (RRF) | None | 1.000 [1.000, 1.000] | 0.944 [0.887, 0.988] | 0.958 |
| Hybrid (RRF) | HyDE | 0.925 [0.825, 1.000] | 0.758 [0.658, 0.856] | 0.817 |
| Hybrid + Rerank | None | 0.975 [0.925, 1.000] | 0.942 [0.879, 0.988] | 0.956 |
| Hybrid + Rerank | HyDE | 0.800 [0.675, 0.925] | 0.499 [0.391, 0.608] | 0.603 |

Findings, reported as-is (no tuning on the eval set):
- The QA set is near-saturated: BM25 already reaches recall@5 1.000 and MRR 0.94, because the questions reuse each doc's wording. The table therefore can't separate the methods well: every paired difference among the `None` rows has a bootstrap CI that touches 0 (`ragkit eval` prints them). Hybrid+rerank's 0.975 vs 1.000 means one question out of 40, not a meaningful regression. Paired Hybrid+Rerank minus Hybrid (`uv run ragkit eval --strategies hybrid,rerank`): dRecall@5 -0.025 [-0.075, +0.000], dMRR -0.002 [-0.052, +0.048] without rewrite; with template HyDE dRecall@5 -0.125 [-0.250, +0.025] and dMRR -0.258 [-0.385, -0.131].
- The "+HyDE" rows don't measure HyDE. Offline, `TemplateLLM` appends the same fixed domain paragraph to every query, which acts as a constant keyword prior and pulls queries towards the imbalance, day-ahead and congestion docs. Real HyDE needs `[llm]`.

`ragkit ask "What is imbalance settlement?" --index .ragkit/index.json`, verbatim:
```
Question: What is imbalance settlement?

Answer:
The implementation of 15-minute imbalance settlement periods across Europe is coordinated with centralized balancing energy exchange platforms: PICASSO for automatic Frequency Restoration Reserves (aFRR) and MARI for manual Frequency Restoration Reserves (mFRR). [doc_24_subhourly_dispatch#rec_5] Imbalance settlement is the post-delivery commercial accounting framework that financially penalizes deviations between a Balance Responsible Party's (BRP) contracted market position and its actual metered energy flows. [doc_05_imbalance_settlement#rec_0] A BRP's imbalance volume in settlement period t is defined as:
Imbalance_Volume(t) = Metered_Generation(t) - Metered_Consumption(t) + Contracted_Sales(t) - Contracted_Purchases(t). [doc_05_imbalance_settlement#rec_1]

Citations: doc_24_subhourly_dispatch#rec_5, doc_05_imbalance_settlement#rec_0, doc_05_imbalance_settlement#rec_1
Support: 1.00 (threshold 0.50)
```
Markdown headings are stripped before sentence extraction. Remaining limitation: the top-ranked sentence still comes from a neighbouring doc that mentions the term.

### Refuse on low confidence

`support` = share of the query's content terms (stopwords removed) found in the best retrieved chunk. Below `--min-support` the pipeline abstains instead of answering.

`ragkit abstention`:

| Set | Correct | Rate | Wilson 95% CI |
|---|---|---|---|
| answerable (answered) | 39/40 | 0.97 | [0.87, 1.00] |
| unanswerable near_domain (abstained) | 9/10 | 0.90 | [0.60, 0.98] |
| unanswerable off_domain (abstained) | 6/6 | 1.00 | [0.61, 1.00] |

- The threshold 0.5 is the 5th percentile of support on the 40 answerable questions. The unanswerable set was not used to pick it.
- The 16 unanswerable questions (`data/unanswerable.jsonl`) were written by the author. They are checked against the corpus by keyword, not by an independent annotator. 10 are near-domain (energy topics the corpus does not cover).
- The near-domain miss: "What is the typical cycle life of a sodium-sulfur battery?" reaches support 0.60, because "cycle", "life" and "battery" all appear in the battery doc. Term coverage cannot tell "the corpus talks about this" from "the corpus answers this".

### Staged ablation

`ragkit ablation` runs the full pipeline (retrieve top 5, extractive answer) on the same 40 answerable and 16 unanswerable questions, adding one component per row:

| Stage | Recall@5 | MRR | Answer F1 | Faithfulness | Answered | Abstain (unans.) |
|---|---|---|---|---|---|---|
| BM25 | 1.000 [1.000, 1.000] | 0.940 [0.877, 0.988] | 0.238 | 1.000 | 40/40 | 0/16 |
| + dense (hybrid RRF) | 1.000 [1.000, 1.000] | 0.944 [0.887, 0.988] | 0.220 | 1.000 | 40/40 | 0/16 |
| + lexical rerank | 0.975 [0.925, 1.000] | 0.902 [0.827, 0.971] | 0.235 | 1.000 | 40/40 | 0/16 |
| + refuse (support >= 0.50) | 0.975 [0.925, 1.000] | 0.902 [0.827, 0.971] | 0.234 | 1.000 | 39/40 | 15/16 |

Reading, as measured:
- **The composed pipeline does not beat its stages on answer quality.** Course lesson 69 expects it to; on this corpus it does not. Hybrid adds +0.004 MRR over BM25, the lexical reranker loses one question at recall@5 and 0.04 MRR, and answer F1 stays within 0.22–0.24 for every stage. The only stage that changes behaviour is refusal: 15/16 unanswerable questions refused for 1/40 answerable lost.
- The QA set is too easy to separate retrievers (BM25 already at recall@5 1.000). The EU corpus below is the harder set.
- Faithfulness is 1.000 by construction: the extractive generator only copies corpus sentences. It becomes informative with an LLM generator.
- Answer F1 is low (≈0.23) because reference answers are paraphrases and the extractive answer is up to 3 corpus sentences; it measures token overlap, not correctness.
- The rerank MRR here (0.902) is lower than in the `eval` table (0.942) because the pipeline reranks 10 candidates (`2 × top_k`) while `eval` reranks 20.

## Real corpus: EU electricity regulation, congestion and balancing

The synthetic set cannot separate retrievers, so ragkit also bundles a corpus of real legal text: 94 articles from six EU acts on cross-zonal congestion management and balancing (337 kB of text, 934 chunks at 500 characters).

| Act (CELEX) | Articles kept |
|---|---|
| Regulation (EU) 2019/943, electricity market regulation (32019R0943) | 24 of 71 |
| Commission Regulation (EU) 2017/2195, electricity balancing guideline (32017R2195) | 28 of 65 |
| Commission Regulation (EU) 2015/1222, capacity allocation and congestion management (32015R1222) | 15 of 84 |
| Directive (EU) 2019/944, electricity market directive (32019L0944) | 9 of 74 |
| Commission Regulation (EU) 2017/1485, system operation guideline (32017R1485) | 14 of 192 |
| Commission Regulation (EU) 2016/1719, forward capacity allocation (32016R1719) | 4 of 64 |

Articles were picked by hand for relevance to congestion and balancing, plus neighbours that share vocabulary (definitions, capacity mechanisms, cost recovery, the three near-identical European balancing-platform articles 19/20/21 of 2017/2195) so the corpus has real hard negatives. These are the original Official Journal texts, not consolidated versions; later amendments (for example Regulation (EU) 2024/1747) are not reflected.

**Licence basis.** The EUR-Lex legal notice (<https://eur-lex.europa.eu/content/legal-notice/legal-notice.html>, read 2026-10-16) says: "The Commission's document reuse policy is based on Decision 2011/833/EU. Unless otherwise specified, you can re-use the legal documents published in EUR-Lex for commercial or non-commercial purposes." Decision 2011/833/EU (<https://eur-lex.europa.eu/eli/dec/2011/833/oj/eng>) allows reuse for commercial and non-commercial purposes and states, in Art. 6(2), that conditions may include acknowledging the source and not distorting the original meaning. Both are honoured: every file starts with a source comment, the articles are split but not reworded, and the changes made are listed in `src/ragkit/data/corpus_eu/SOURCES.json` (URL, CELEX id, retrieval date, SHA-256 of each download). This is my reading of the notice, not legal advice; two of the six acts (2019/943, 2019/944) are Parliament and Council acts rather than Commission documents, and I relied on the notice's wording for "legal documents published in EUR-Lex". No Dutch national sources are included.

**How it was built.** `eur-lex.europa.eu` answers scripted requests with an empty HTTP 202 (bot challenge), so the text comes from the Publications Office Cellar endpoint `https://publications.europa.eu/resource/celex/<CELEX>` (English XHTML, retrieved 2026-10-16). `src/ragkit/eurlex.py` parses it with the standard library, cuts one markdown file per article, drops footnote markers and flattens point tables. Rebuild (needs network, caches under the gitignored `data/raw/eurlex/`):

```bash
uv run ragkit fetch-eu --retrieved-on 2026-10-16     # rewrites src/ragkit/data/corpus_eu/
```

Regenerating from the cached download reproduced the committed Markdown files byte for byte (checked with `diff -r`). A fresh download could differ if the Publications Office republishes the XHTML; compare the SHA-256 in `SOURCES.json`.

**Questions.** `qa_eu.jsonl`: 42 answerable (17 `direct`, 17 `paraphrase`, 8 `multi-hop` needing two articles) and `unanswerable_eu.jsonl`: 12 on-topic questions the corpus does not answer. Every record has `provenance` (article ids with paragraph) and `status: "reviewed-agent"`; the 28 records that the review changed also carry a `review_note`.

> **Review status: second-pass review by a separate agent; no human domain expert has reviewed these labels, and single-annotator risk remains.** The questions were first drafted by one AI agent. A second, independent agent pass then read every cited article and re-checked each answer, each paraphrase and multi-hop claim, and the relevant-article labels against the BM25 and dense top-10 (and checked that none of the 94 articles answers an unanswerable question). It changed 25 of 42 answerable items (13 questions and 19 reference answers rewritten, 4 retyped, 3 relevant ids added and 1 removed) and 3 of 12 unanswerable ones, and dropped none; the per-item log is in [`docs/qa_eu_review.md`](docs/qa_eu_review.md). The reviewers are agents, probably of the same model family as the drafter, so shared blind spots are possible, and the owner could not audit the result. Offline tests check that every `relevant_doc_id` exists and that every quoted span in a reference answer occurs verbatim in the cited articles; they do not check that the labels are complete. Where the same rule is repeated in two acts, both were labelled only if the question needs both.

The review changed the benchmark, so the numbers below are not comparable one-to-one with the draft-label run. Draft-label run, for comparison (BM25 / Dense / Hybrid / Hybrid+Rerank, no rewrite, recall@5 / MRR): 0.893 / 0.663, 0.750 / 0.639, 0.786 / 0.665, 0.821 / 0.622. After review: 0.857 / 0.676, 0.619 / 0.528, 0.786 / 0.636, 0.821 / 0.591. Part of the movement is the stricter paraphrase questions (less lexical overlap), part is the added labels; the two effects were not separated.

Paraphrase questions avoid the article's wording (for example "A wind farm is switched off by the grid operator without a market mechanism. Is the owner compensated?" for Art. 13 "non-market-based redispatching"). The last row of the by-type table below measures this: the share of a question's content terms found in its best relevant chunk is 0.669 for `direct`, 0.451 for `multi-hop` and 0.259 for `paraphrase` (draft labels: 0.704, 0.548, 0.336). The reviewer retyped three items from `paraphrase`/`multi-hop` to `direct` because the question reused the article's key phrase or one article sufficed.

### Retrieval: `ragkit eval --corpus eu`

Recursive chunks of 500 characters, 50 overlap, 42 questions, mean [bootstrap 95% CI over questions, 2000 resamples]:

| Strategy | Rewrite | Recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25 | None | 0.857 [0.750, 0.952] | 0.676 [0.560, 0.791] | 0.702 |
| BM25 | HyDE | 0.774 [0.655, 0.881] | 0.621 [0.503, 0.736] | 0.650 |
| Dense | None | 0.619 [0.476, 0.750] | 0.528 [0.401, 0.647] | 0.546 |
| Dense | HyDE | 0.298 [0.167, 0.429] | 0.231 [0.142, 0.324] | 0.275 |
| Hybrid (RRF) | None | 0.786 [0.667, 0.893] | 0.636 [0.518, 0.742] | 0.676 |
| Hybrid (RRF) | HyDE | 0.702 [0.571, 0.833] | 0.497 [0.378, 0.625] | 0.539 |
| Hybrid + Rerank | None | 0.821 [0.702, 0.929] | 0.591 [0.483, 0.701] | 0.648 |
| Hybrid + Rerank | HyDE | 0.512 [0.369, 0.643] | 0.327 [0.236, 0.426] | 0.418 |

Paired difference against BM25 / None (same questions; CI of the mean per-question difference):

| Strategy | Rewrite | dRecall@5 | dMRR |
|---|---|---|---|
| BM25 | HyDE | -0.083 [-0.190, +0.000] | -0.055 [-0.127, +0.017] |
| Dense | None | -0.238 [-0.405, -0.083] | -0.147 [-0.279, -0.023] |
| Dense | HyDE | -0.560 [-0.702, -0.429] | -0.445 [-0.572, -0.315] |
| Hybrid (RRF) | None | -0.071 [-0.190, +0.048] | -0.040 [-0.127, +0.050] |
| Hybrid (RRF) | HyDE | -0.155 [-0.274, -0.048] | -0.178 [-0.279, -0.070] |
| Hybrid + Rerank | None | -0.036 [-0.131, +0.048] | -0.084 [-0.172, +0.003] |
| Hybrid + Rerank | HyDE | -0.345 [-0.488, -0.214] | -0.349 [-0.458, -0.245] |

By question type, no rewrite, mean Recall@5 / MRR:

| Strategy | direct (n=17) | multi-hop (n=8) | paraphrase (n=17) |
|---|---|---|---|
| BM25 | 1.000 / 0.828 | 0.750 / 0.938 | 0.765 / 0.400 |
| Dense | 0.853 / 0.726 | 0.688 / 0.812 | 0.353 / 0.197 |
| Hybrid (RRF) | 0.941 / 0.824 | 0.750 / 0.781 | 0.647 / 0.380 |
| Hybrid + Rerank | 1.000 / 0.730 | 0.812 / 0.792 | 0.647 / 0.358 |
| *question terms in best relevant chunk* | 0.669 | 0.451 | 0.259 |

What this does and does not show, as measured:
- **The set is no longer saturated.** BM25 recall@5 falls from 1.000 (synthetic) to 0.857 and MRR from 0.940 to 0.676; the paraphrase subset is where it breaks (MRR 0.400).
- **Separated:** the three template-HyDE rows for Dense, Hybrid and Hybrid+Rerank are clearly below BM25 / None (CIs exclude 0 on both metrics); BM25 + HyDE is not (recall CI upper bound +0.000, MRR CI includes 0). Dense / None is now also separated from BM25 / None on both metrics (recall -0.238, MRR -0.147, CIs exclude 0); on the draft labels it was separated on recall only. These comparisons are against BM25 / None, not against each method's own no-rewrite row.
- **Still not separated:** BM25 vs Hybrid (recall -0.071, CI [-0.190, +0.048]; MRR -0.040, CI includes 0) and BM25 vs Hybrid+Rerank (recall -0.036; MRR -0.084, CI upper bound +0.003). Hybrid fusion with the offline hashing embedder does not add anything over BM25 on this set, and no claim can be made either way about reranking. Hybrid vs Hybrid+Rerank, now computed directly (`uv run ragkit eval --corpus eu --strategies hybrid,rerank`, paired bootstrap over the same 42 questions): dRecall@5 +0.036 [-0.107, +0.167] and dMRR -0.045 [-0.150, +0.059] without rewrite, so the 0.821 vs 0.786 recall gap is a tie. With template HyDE the reranker is worse than Hybrid (dRecall@5 -0.190 [-0.333, -0.060], dMRR -0.170 [-0.287, -0.057]).
- With n=42 the CIs are wide (about ±0.1 on recall), so differences under roughly 0.1 should be read as ties. The "+HyDE" rows measure the constant-prefix `TemplateLLM`, not real HyDE (see above).
- Dense retrieval with the hashing embedder collapses on the reworded paraphrase questions (recall 0.353, MRR 0.197), which is where the lexical overlap was removed.
- Multi-hop MRR is higher than recall because MRR only needs the first relevant article, while recall@5 needs both.

### Abstention transfer: `ragkit abstention --corpus eu`

The refusal threshold 0.5 was set on the synthetic answerable set. Applied unchanged to the EU set:

| Set | Correct | Rate | Wilson 95% CI |
|---|---|---|---|
| answerable (answered) | 20/42 | 0.48 | [0.33, 0.62] |
| answerable direct (answered) | 14/17 | 0.82 | [0.59, 0.94] |
| answerable multi-hop (answered) | 4/8 | 0.50 | [0.22, 0.78] |
| answerable paraphrase (answered) | 2/17 | 0.12 | [0.03, 0.34] |
| unanswerable near_domain (abstained) | 9/12 | 0.75 | [0.47, 0.91] |

It does not transfer: 0.5 refuses 52% of answerable questions (22 of 42), and 88% of the paraphrase ones (15 of 17), because those share few words with the article. The command also prints the 5th percentile of support on the answerable set, which is 0.250 here (0.500 on the synthetic set). Re-calibrating the same way (`--min-support 0.25`, chosen from the answerable set only; reported alongside, not replacing the default):

| Set | Correct | Rate | Wilson 95% CI |
|---|---|---|---|
| answerable (answered) | 40/42 | 0.95 | [0.84, 0.99] |
| answerable direct (answered) | 17/17 | 1.00 | [0.82, 1.00] |
| answerable multi-hop (answered) | 8/8 | 1.00 | [0.68, 1.00] |
| answerable paraphrase (answered) | 15/17 | 0.88 | [0.66, 0.97] |
| unanswerable near_domain (abstained) | 1/12 | 0.08 | [0.01, 0.35] |

Draft-label run, for comparison: at 0.50 answered 22/42 (direct 13/14, multi-hop 4/8, paraphrase 5/20) and abstained 9/12; at 0.25 answered 40/42 (14/14, 8/8, 18/20) and abstained 1/12.

The default threshold stays 0.5. Operating points on this set (`for t in 0.25 0.30 0.40 0.50; do ragkit abstention --corpus eu --min-support $t; done`), shown as a trade-off curve and not a tuned choice: answered / abstained = 40/42 and 1/12 at 0.25, 31/42 and 2/12 at 0.30, 27/42 and 8/12 at 0.40, 20/42 and 9/12 at 0.50 (draft labels: 40/42 and 1/12, 34/42 and 2/12, 30/42 and 8/12, 22/42 and 9/12). None of these four points gives both a high answer rate and a high refusal rate. Query-term coverage is a lexical signal: it cannot tell a paraphrased answerable question from an on-topic question whose terms all occur in the corpus (the EU unanswerable set is on-topic by construction). With 12 unanswerable questions the CIs are wide. The unanswerable questions changed little in the review (3 of 12 edited), so the 9/12 and 1/12 refusal counts are unchanged.

### Abstention signals under domain shift: `ragkit abstention --corpus eu --signals all`

Question: is query-term coverage (the current `query_support`) a good abstention signal, or does another cheap signal from the same pipeline separate answerable from unanswerable questions better? Nine signals, all computed from the existing pipeline (no new dependency), each oriented so that higher = more likely answerable (the sign is fixed in advance, never flipped):

| Signal | Definition |
|---|---|
| `query_support` | share of query content terms in the best of the top-5 reranked chunks (the current rule) |
| `idf_support` | same, weighted by BM25 IDF; unseen terms get the maximum IDF |
| `bm25_top`, `bm25_norm`, `bm25_margin` | top raw BM25 score; top score / its per-query upper bound; top1 minus top2 |
| `dense_top`, `dense_margin` | top hashing-embedder cosine; top1 minus top2 |
| `rerank_top`, `rerank_margin` | top `LexicalReranker` score over the hybrid candidates; top1 minus top2 |

The `[st]` cross-encoder was not run (not installed here), so no neural signal is compared.

Commands (offline, a few seconds each): `uv run ragkit abstention --corpus eu --signals all` and `uv run ragkit abstention --corpus synthetic --signals all`. AUROC = P(answerable scores higher than unanswerable), ties count 1/2. CIs: percentile bootstrap, 2000 resamples, answerable and unanswerable questions resampled separately, seed 0. The paired difference resamples the same questions for both signals. Labels are the agent-reviewed ones described above.

**AUROC, answerable vs unanswerable.** EU = 42 vs 12 questions; synthetic = 40 vs 16 (for contrast):

| Signal | EU AUROC | EU AUROC minus `query_support` | Synthetic AUROC |
|---|---|---|---|
| `query_support` | 0.642 [0.478, 0.796] | baseline | 0.987 [0.951, 1.000] |
| `idf_support` | 0.633 [0.476, 0.776] | -0.009 [-0.071, +0.045] | 0.992 [0.969, 1.000] |
| `bm25_top` | 0.734 [0.567, 0.875] | +0.092 [-0.040, +0.223] | 0.953 [0.897, 0.994] |
| `bm25_norm` | 0.554 [0.367, 0.726] | -0.088 [-0.244, +0.042] | 0.923 [0.847, 0.977] |
| `bm25_margin` | 0.665 [0.500, 0.815] | +0.023 [-0.138, +0.206] | 0.780 [0.645, 0.891] |
| `dense_top` | 0.734 [0.569, 0.879] | +0.092 [-0.117, +0.312] | 0.856 [0.742, 0.952] |
| `dense_margin` | 0.433 [0.240, 0.629] | -0.209 [-0.446, +0.028] | 0.747 [0.614, 0.863] |
| `rerank_top` | 0.562 [0.393, 0.724] | -0.080 [-0.231, +0.051] | 0.759 [0.603, 0.900] |
| `rerank_margin` | 0.393 [0.232, 0.562] | -0.249 [-0.456, -0.037] | 0.702 [0.548, 0.841] |

**Risk-coverage on the EU set.** Answer the top share of all 54 questions by each signal. Cell = "unanswerable share of answered / bad share of answered", where bad = unanswerable, or answerable with no relevant article in the retrieved top 5 (6 of the 42 answerable questions miss). At 100% coverage every signal gives 0.222 / 0.333 (12 of 54 are unanswerable). Ties at the cut-off are split fractionally.

| Signal | 90% | 80% | 70% |
|---|---|---|---|
| `query_support` | 0.214 / 0.306 | 0.233 / 0.302 | 0.202 / 0.281 |
| `idf_support` | 0.204 / 0.306 | 0.233 / 0.302 | 0.211 / 0.289 |
| `bm25_top` | 0.184 / 0.306 | 0.163 / 0.233 | 0.158 / 0.211 |
| `bm25_norm` | 0.224 / 0.327 | 0.209 / 0.279 | 0.184 / 0.263 |
| `bm25_margin` | 0.224 / 0.327 | 0.209 / 0.302 | 0.158 / 0.263 |
| `dense_top` | 0.204 / 0.327 | 0.186 / 0.302 | 0.132 / 0.263 |
| `dense_margin` | 0.224 / 0.327 | 0.209 / 0.302 | 0.211 / 0.289 |
| `rerank_top` | 0.224 / 0.306 | 0.209 / 0.302 | 0.211 / 0.316 |
| `rerank_margin` | 0.245 / 0.327 | 0.256 / 0.349 | 0.289 / 0.395 |

On the synthetic set the same rows for `query_support` fall from 0.286 (100%) to 0.200 / 0.111 / 0.026 at 90 / 80 / 70% coverage; on the EU set they barely move (0.222 to 0.214 / 0.233 / 0.202).

**Threshold transfer on the EU set.** A threshold `t` (answer when score >= t) is chosen on training folds only: 5-fold stratified CV by question (seed 0; folds hold 8 to 9 answerable and 2 to 3 unanswerable questions), maximising balanced accuracy (mean of answer rate on answerable and abstain rate on unanswerable) on the training folds. Held-out decisions are pooled over the five folds. "Always answer" scores 0.500. The last column is the range of the held-out balanced accuracy over CV seeds 0 to 4 (`for s in 1 2 3 4; do uv run ragkit abstention --corpus eu --signals all --cv-seed $s; done`):

| Signal | answered (42 answerable) | abstained (12 unanswerable) | balanced acc. (seed 0) | range, seeds 0-4 |
|---|---|---|---|---|
| `query_support` | 20/42 (0.48) [0.33, 0.62] | 8/12 (0.67) [0.39, 0.86] | 0.571 | 0.548-0.571 |
| `idf_support` | 18/42 (0.43) [0.29, 0.58] | 10/12 (0.83) [0.55, 0.95] | 0.631 | 0.506-0.631 |
| `bm25_top` | 25/42 (0.60) [0.44, 0.73] | 9/12 (0.75) [0.47, 0.91] | 0.673 | 0.560-0.714 |
| `bm25_norm` | 22/42 (0.52) [0.38, 0.67] | 5/12 (0.42) [0.19, 0.68] | 0.470 | 0.429-0.530 |
| `bm25_margin` | 25/42 (0.60) [0.44, 0.73] | 10/12 (0.83) [0.55, 0.95] | 0.714 | 0.613-0.714 |
| `dense_top` | 27/42 (0.64) [0.49, 0.77] | 7/12 (0.58) [0.32, 0.81] | 0.613 | 0.613-0.661 |
| `dense_margin` | 34/42 (0.81) [0.67, 0.90] | 2/12 (0.17) [0.05, 0.45] | 0.488 | 0.405-0.488 |
| `rerank_top` | 12/42 (0.29) [0.17, 0.44] | 12/12 (1.00) [0.76, 1.00] | 0.643 | 0.530-0.643 |
| `rerank_margin` | 7/42 (0.17) [0.08, 0.31] | 6/12 (0.50) [0.25, 0.75] | 0.333 | 0.333-0.429 |

(Brackets are Wilson 95% CIs.) For contrast, the same procedure on the synthetic set gives `query_support` 39/40 answered and 14/16 abstained (balanced accuracy 0.925) and `idf_support` 40/40 and 15/16 (0.969).

**Zero-shot transfer of the synthetic-calibrated threshold.** The synthetic rule is "5th percentile of the signal on the answerable questions" (how 0.5 was set for `query_support`; no unanswerable data used), applied to each signal unchanged on the EU set:

| Signal | synthetic `t` | answered (answerable) | abstained (unanswerable) | balanced acc. |
|---|---|---|---|---|
| `query_support` | 0.500 | 20/42 (0.48) [0.33, 0.62] | 9/12 (0.75) [0.47, 0.91] | 0.613 |
| `idf_support` | 0.431 | 15/42 (0.36) [0.23, 0.51] | 12/12 (1.00) [0.76, 1.00] | 0.679 |
| `bm25_top` | 12.287 | 40/42 (0.95) [0.84, 0.99] | 2/12 (0.17) [0.05, 0.45] | 0.560 |
| `bm25_norm` | 0.184 | 24/42 (0.57) [0.42, 0.71] | 6/12 (0.50) [0.25, 0.75] | 0.536 |
| `bm25_margin` | 0.179 | 38/42 (0.90) [0.78, 0.96] | 1/12 (0.08) [0.01, 0.35] | 0.494 |
| `dense_top` | 0.529 | 42/42 (1.00) [0.92, 1.00] | 0/12 (0.00) [0.00, 0.24] | 0.500 |
| `dense_margin` | 0.001 | 38/42 (0.90) [0.78, 0.96] | 0/12 (0.00) [0.00, 0.24] | 0.452 |
| `rerank_top` | 0.539 | 27/42 (0.64) [0.49, 0.77] | 5/12 (0.42) [0.19, 0.68] | 0.530 |
| `rerank_margin` | 0.010 | 34/42 (0.81) [0.67, 0.90] | 0/12 (0.00) [0.00, 0.24] | 0.405 |

**Verdict.**
- **No signal separates the classes reliably on the EU set.** The best point estimates are `bm25_top` and `dense_top` at 0.734, with CIs [0.567, 0.875] and [0.569, 0.879]: lower bounds only just above 0.5, for 2 of 9 signals, and with 9 uncorrected comparisons one such lower bound is not strong evidence. The current `query_support` (0.642 [0.478, 0.796]) is itself not distinguishable from chance. [INFERENCE] The EU unanswerable questions are on-topic by construction, so term overlap and lexical or hashing-cosine similarity are both high for them.
- **No signal is shown to be better than the baseline.** The paired AUROC difference CI includes 0 for 7 of the 8 alternatives; `rerank_margin` is worse (-0.249 [-0.456, -0.037]). `bm25_top` and `dense_top` are +0.092 above the baseline, but with CIs [-0.040, +0.223] and [-0.117, +0.312] this is a tie as far as this data can say.
- **Domain shift, measured.** On the synthetic set all 9 signals are above chance and `query_support` (0.987) is the best or tied (`idf_support` +0.005 [+0.000, +0.025]); six alternatives are significantly worse than it. On the EU set the same baseline falls to 0.642. The two sets differ in more than corpus (the synthetic unanswerable set includes 6 off-domain questions; the EU one has none), and I did not split AUROC by question kind, so the drop is not attributed to a single cause.
- **Risk-coverage agrees.** Dropping the lowest 30% of questions by `query_support` leaves the unanswerable share at 0.202 (0.222 with no abstention); the best EU row, `dense_top` at 70% coverage, still has 0.132, which is about 5 unanswerable questions among 38 answered.
- **Thresholds do not rescue a weak signal.** Held-out balanced accuracy is 0.33 to 0.71, with three signals at or below the 0.500 of always answering (`bm25_norm`, `dense_margin`, `rerank_margin`). The top held-out value (`bm25_margin`, 0.714) belongs to a signal whose AUROC CI lower bound is 0.500 and which moves between 0.613 and 0.714 over seeds 0 to 4; picking it afterwards would be selection on 12 questions.
- **Raw-score thresholds do not transfer.** BM25 and cosine scales depend on the corpus (IDF, length), so the synthetic 5th percentile either answers nearly everything (`bm25_top`: 40/42 answered, 2/12 abstained; `dense_top`: 42/42 and 0/12) or nothing useful. The two term-coverage signals keep their abstention rate on the unanswerable questions (9/12 and 12/12) but at the cost of refusing 52% and 64% of answerable questions. The best zero-shot balanced accuracy (`idf_support`, 0.679) is a single operating point without an interval.

**CI width caveat.** With 12 unanswerable questions the bootstrap AUROC CIs are 0.30 to 0.39 wide and the paired-difference CIs 0.12 to 0.47 wide, so differences below about 0.2 AUROC cannot be resolved on this set. Percentile bootstrap intervals with 12 negatives are probably too narrow rather than too wide [INFERENCE]. The intervals reflect question-sampling noise only, not label noise: the labels are agent-reviewed, not human-expert-reviewed (see the review note above), and one mislabelled unanswerable question could move an AUROC by roughly up to 0.08 [INFERENCE].

### Staged ablation: `ragkit ablation --corpus eu`

| Stage | Recall@5 | MRR | Answer F1 | Faithfulness | Answered | Abstain (unans.) |
|---|---|---|---|---|---|---|
| BM25 | 0.833 [0.726, 0.929] | 0.671 [0.552, 0.789] | 0.228 | 1.000 | 42/42 | 0/12 |
| + dense (hybrid RRF) | 0.714 [0.595, 0.833] | 0.611 [0.486, 0.726] | 0.218 | 1.000 | 42/42 | 0/12 |
| + lexical rerank | 0.786 [0.679, 0.893] | 0.616 [0.501, 0.726] | 0.219 | 1.000 | 42/42 | 0/12 |
| + refuse (support >= 0.50) | 0.786 [0.679, 0.893] | 0.616 [0.501, 0.726] | 0.117 | 1.000 | 20/42 | 9/12 |

Draft-label run, for comparison (recall@5 / MRR): BM25 0.845 / 0.650, + dense 0.726 / 0.641, + rerank 0.750 / 0.609 and + refuse 0.750 / 0.609, answered 22/42 at the refusal stage.

- Recall@5 here is lower than in the `eval` table (BM25 0.833 vs 0.857) because the pipeline keeps the top 5 *chunks* and collapses them to documents, which can leave fewer than 5 distinct articles, while `eval` collapses the top 10 chunks first.
- Answer F1 (about 0.22) is almost identical across stages and says little: the extractive generator returns up to three sentences and F1 is token overlap with the quoted reference answer, not correctness. It drops to 0.117 with refusal because refused questions score 0. The reference answers were also rewritten in the review (19 of 42), so F1 is not comparable with the draft-label run (about 0.24).
- Faithfulness is 1.000 by construction (extractive generator).
- The only stage that changes behaviour is refusal, and at the default threshold it costs 22 of 42 answerable questions to refuse 9 of 12 unanswerable ones (see above).

### Caveats

- Labels are agent-drafted and second-pass reviewed by a separate agent; no human domain expert has reviewed them and single-annotator risk remains (see the note above and [`docs/qa_eu_review.md`](docs/qa_eu_review.md)); 42 and 12 questions.
- English original texts only; not consolidated; amendments after 2019 are absent, so a question about current law may differ from what the corpus says.
- Article-level documents are long (up to about 11k characters), so the 500-character chunks lose the article heading in most chunks; doc-level labels hide that.
- Abstention-signal comparison: 12 unanswerable questions make the AUROC CIs about 0.3 wide (see that section); the cross-encoder signal was not run.
- The hashing embedder and lexical reranker are the offline defaults. Neural embeddings (`[st]` extra) were not run, so "dense retrieval does not help" applies to the hashing embedder only.

## Design decisions
- **Protocols for every model boundary** (`Embedder`, `Reranker`, `LLM`, `Generator`): offline defaults keep tests deterministic, and neural models plug in through extras.
- **Hashing embedder instead of a vocabulary**: fixed 512-dim, no fitting state beyond IDF, handles unseen words and typos through char n-grams.
- **RRF as the default fusion**: it is rank-based, so BM25 and cosine scores never need calibration against each other.
- **Extractive generator**: every sentence keeps its `[chunk_id]`, so faithfulness is checkable by construction.
- **Doc-level metrics**: chunks are deduplicated to doc ids before computing recall/MRR/nDCG, matching how `qa.jsonl` is labelled.
- **Persisted index = chunks only**: `ragkit index` writes the chunk list to JSON. BM25 statistics and hashing embeddings are rebuilt from it at load time (milliseconds at this scale), which keeps the format trivial and versionable.