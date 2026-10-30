# Retrieval-miss protocol (K-R3): why six gold articles never reach the top-5 chunks (frozen before scoring)

Problem. With the default neural retriever (`hybrid+ce@100`: bge-small + BM25 fused by RRF,
cross-encoder rerank of the top 100) 6 of the 42 answerable original-EU questions have no gold
article among their top-5 chunks. Article-context generation takes its articles from those five
chunks, so that is a ceiling on article-RAG (`docs/generation_protocol.md`, Phase L k = 5 addendum:
gold coverage 36 of 42). This protocol (a) diagnoses the misses and (b) tests four candidate fixes
against the default, retrieval only: **no generation, no LLM call, no download**. Only the pinned
bge-small, MiniLM cross-encoder and the committed corpora are used, offline.

This file is committed before any code of the study runs, before the diagnosis and before any
tuning or held-out number exists. `git log` shows the order: this protocol, the code, the tuning
run (which fixes the chosen settings in `docs/sota-runs/retrieval_miss_tuning.json`), the single
held-out scoring run, the diagnosis report, then the documentation updates. Numbers are reported
whatever they turn out to be; "no detectable effect" means the paired 95 % CI includes 0.

## What was already known (disclosed taint)

- "6 of 42 gold articles are missing from the top-5 chunks" is already printed
  (`docs/sota-runs/article_coverage_hybrid_ce.txt`: 36 of 42 with `hybrid+ce`; the default
  pipeline's table is `article_coverage_default.txt`). **Which** six questions they are, and why, is not
  known to the author of this protocol: no per-question gold rank has been looked at.
- The original 42 questions are therefore the set on which the problem was found. They are also
  where `hybrid-convex`'s alpha was chosen in K-R1 (`docs/neural_protocol.md`). Here they are
  scored once as held-out data, but they are **not clean**: the candidate fixes are motivated by
  their misses. A non-gating sensitivity row drops them (pooled hard 39 + keyword-free 49).
- The hard 39 and keyword-free 49 sets have been scored by earlier protocols (ctxprefix, K-R1)
  but no setting below was ever chosen on them, and none will be.
- The fresh 29 set was scored by the ctxprefix protocol and by K-R1 as a test set. It is **spent
  as a tuning set here** (together with the synthetic set) and is never scored for a decision.

## Data

| role | set | questions | corpus |
|---|---|---|---|
| tuning | fresh (`qa_eu_fresh.jsonl`) | 29 | EU, 94 articles |
| tuning and guard | synthetic (`qa.jsonl`) | 40 | synthetic, 25 documents (each document is its own "article") |
| held-out | original (`qa_eu.jsonl`) | 42 | EU |
| held-out | hard (`qa_eu_hard.jsonl`) | 39 | EU |
| held-out | keyword-free (`qa_eu_nokw.jsonl`) | 49 | EU |

Pooled held-out = 42 + 39 + 49 = 130 questions. Labels are agent-reviewed (README caveat).

## Systems (the baseline B0 is the shipped default of `ragkit answer --retriever hybrid+ce`)

A *unit* is a chunk (recursive 500 characters, 50 overlap, exactly the chunks of every earlier EU
table, no context prefix) or an article. A system is a `Config`:

| field | values | baseline B0 |
|---|---|---|
| `unit` | `chunk`, `article` | `chunk` |
| `instruction` | bge query instruction `Represent this sentence for searching relevant passages: ` on / off (documents never get one) | on |
| `fusion` | `rrf` (k = 60) or convex `alpha * minmax(bm25) + (1 - alpha) * minmax(dense)`, alpha in {0.1, ..., 0.9}, a missing hit scores 0 | `rrf` |
| `depth` D | 50, 100, 200: the first-stage lists are the top D of BM25 and of dense, the fused list is cut to D, and the cross-encoder reranks all D | 100 |
| `stage` | `bm25`, `dense`, `hybrid` (fused list), `hybrid+ce` (cross-encoder rerank of the fused list; the tail is empty because D = depth) | `hybrid+ce` |

B0 is `hybrid+ce@100` of `ragkit.neural` and a test asserts that the new code reproduces it
exactly with fake models.

**Article units (candidate F1).** One unit per article.

- BM25: the whole article text, the shipped `BM25Index` (k1 1.5, b 0.75), no chunking.
- Dense: the article text is cut into windows of at most 1800 characters with 200 overlap (the
  existing `recursive` chunker, `chunk_size=1800`, `overlap=200`; about 450 bge tokens, under the
  512-token limit; 76 of the 94 EU articles and all 25 synthetic documents have more than one
  window). Every window is embedded; an article's dense score is the **maximum** over its
  windows (max-pooling).
- Cross-encoder: the same windows; the article's logit is the maximum over its windows.
- The window size is not tuned.

## Ranking and metrics

Per system and question the ranked units are mapped to article ids. Two article lists:

- **list**: the first 10 units, deduplicated to articles, first occurrence (`rank_documents`,
  the convention of every earlier EU table). MRR, Recall@5, wrong-act and `G@k (list)` use it.
- **prod**: the first 5 units, deduplicated (what `ragkit answer` hands to the article-context
  generator, `parent_articles`). `G@k (prod)` uses it. For chunks this list can hold fewer than 5
  articles; for article units it always holds 5. That asymmetry is part of what F1 changes, so
  each table also gives the chunk baseline under the `list` definition (`G@5 (list)`), which
  separates "deduplication of the same ranking" from "a different ranking".

Metrics (per question, then mean with a 95 % percentile bootstrap over questions, 2000
resamples, seed 0, `ragkit.eval`): `G@3` and `G@5` = 1 when any gold article is among the first
3 / 5 articles of the list; MRR of the list; wrong-act = top-1 article's act (id without `_artN`;
the document itself for synthetic) is none of the gold acts, or nothing retrieved
(`ragkit.ctxprefix.is_wrong_act`). Paired differences (candidate minus B0, same questions) with
the paired bootstrap CI, plus better / worse / tied counts.

## Candidate fixes (all frozen here)

| id | change from B0 | free parameter, chosen on the tuning sets only |
|---|---|---|
| F1 | `unit = article` | none |
| F2 | `depth` D | D in {50, 100, 200} |
| F3 | `fusion` | `rrf` or alpha in {0.1, ..., 0.9} |
| F4 | `instruction = off` | none |
| S | joint stack: `unit`, `instruction`, `depth`, `fusion` together | the best cell of the whole grid (2 x 2 x 3 x 10 = 120 cells) |

**Tuning.** Every cell of the grid is scored with `stage = hybrid+ce` on the fresh and the
synthetic questions. Objective of a configuration: the mean of the two set-level MRR values
(the two sets weigh equally; the synthetic corpus is small, so the average mostly acts as a guard
against settings that break it). F2, F3 and F4 are chosen one factor at a time with every other
field at B0; S is the argmax over the whole grid. Ties (difference below 1e-9) go to the higher
mean `G@5 (prod)`, then to the configuration with the fewest fields different from B0, then to the
smaller depth, then to RRF, then to the alpha closest to 0.5. If the argmax of a factor is B0
itself, "tuning selected the baseline" is reported and that candidate is not scored as a
different system (its held-out row would be B0). The chosen settings are written to
`docs/sota-runs/retrieval_miss_tuning.json` and committed **before** the held-out run; the
scoring command reads them from that file.

**Held-out scoring (once).** B0 and every candidate that differs from B0, `stage = hybrid+ce`, on
the original 42, hard 39 and keyword-free 49, each alone and pooled. One run, no re-run after
looking. Non-gating rows in the same run (nothing is selected on them): the other two depths for
chunks (F2 sensitivity), `hybrid` without the cross-encoder under RRF and under the tuned alpha
(F3 where it acts first), and the `bm25`, `dense`, `hybrid`, `hybrid+ce` stage rows for both unit
types at B0's settings. The cross-encoder logit of a (query, unit) pair is computed once per
process and cached, so every row of one run shares the same scores.

## Decision rule (per candidate; the standing ctxprefix rule, adapted)

The standing rule (`docs/ctxprefix_protocol.md`) adopts a variant iff it gains on a fresh set
(paired MRR CI lower bound above 0) and does not regress on the synthetic guard (paired MRR
difference at least -0.02 and its CI upper bound not below 0). The fresh set is spent on tuning
here, so the gain test moves to the pooled held-out sets. A candidate X is **adopted** iff all
four hold:

1. pooled held-out paired MRR difference (X minus B0): 95 % CI lower bound above 0;
2. pooled held-out wrong-act count of X not above B0's (the K-R1 EU rule);
3. pooled held-out `G@5 (prod)` difference, point estimate at least 0 (the target is fewer
   misses, so a ranking gain bought with new misses does not count);
4. synthetic guard: paired MRR difference X minus B0 on the synthetic questions at least -0.02
   and its CI upper bound not below 0 (the synthetic set is also tuned on, so this is a
   consistency check, not independent evidence).

Otherwise **rejected** when the pooled MRR CI lies entirely below 0 or condition 4 fails, and
**not adopted (no evidence)** when the pooled CI includes 0. If several candidates are adopted,
the one with the highest pooled MRR difference point estimate becomes the default. Five
candidates are five tests with no multiplicity correction (about a 12 % chance that at least one
passes the one-sided lower-bound test under the null). "Not adopted (no evidence)" never means "it
does not help": the pooled set can be at its ceiling or too small.

If adopted, the winner becomes the default of the retrieval that serves `ragkit answer
--retriever hybrid+ce` and the README numbers are updated. If none is adopted the default stays.

## Diagnosis (descriptive, no decision attached)

For every question of the original 42 whose gold articles are all missing from the top-5 chunks
of B0 (and, as a count only, on the other held-out sets): the rank of the best gold article, and
of its best chunk, under `bm25`, `dense`, `hybrid` and `hybrid+ce` (B0 settings, chunks; article
rank = position among distinct articles ordered by best chunk; "not in pool" when the system's
top-100 lists do not contain it), plus the same article's rank under article units (F1).
Mechanical cause flags (all computed from the text, no reading of the question by the author):

- **vocabulary mismatch**: fewer than 50 % of the query's content terms (lowercase alphanumeric
  tokens, minus `ragkit.pipeline.STOPWORDS`) occur anywhere in the best gold article;
- **chunk boundary split**: the share of content terms found in the gold article's best single
  chunk is at least 0.25 lower than the share found in the whole gold article;
- **near-duplicate outranked**: some article that outranks the gold article in the B0 article
  ranking has word-bigram Jaccard similarity of at least 0.30 with a gold article (also
  reported: whether it belongs to the same act);
- **query too short / too long**: word count below the 10th or above the 90th percentile of the
  159 EU answerable questions (all five EU files, fresh included).

A miss may carry several flags; its **primary** cause is the first that applies in the order
vocabulary mismatch, chunk boundary split, near-duplicate outranked, query length, otherwise
"unexplained". The thresholds (0.5, 0.25, 0.30, the percentiles) are fixed here and not tuned.

## Predictions (written before any run)

- D1: at least 3 of the 6 misses carry the near-duplicate flag and at least 2 have vocabulary
  mismatch as their primary cause; at most 1 has a chunk boundary split as primary cause.
- D2: for the majority of misses `dense` ranks the gold article better than `bm25` does.
- P-F1: article units raise pooled `G@5 (prod)` over B0 (partly because `prod` always holds 5
  articles) and reduce the wrong-act count, but the MRR gain is small; the 120-cell argmax S ends
  up with `unit = article`.
- P-F2: depth has no detectable pooled MRR effect (K-R1: depths 20 / 50 / 100 differ by about
  0.013 pooled MRR); tuning picks 50 or 100.
- P-F3: convex versus RRF has no detectable effect on `hybrid+ce` (the cross-encoder decides the
  order); a difference shows only in the no-cross-encoder `hybrid` rows.
- P-F4: instruction off does not help (bge's card recommends the instruction for short queries
  retrieving passages); the point estimate is at or below 0.
- P-adopt: at most F1 and S pass the rule; F2, F3 and F4 are "not adopted (no evidence)" or
  rejected.

## Not tried (would need a fresh protocol)

Window size and overlap of article units, BM25 on windows instead of the whole article, a score
blend of cross-encoder and first stage, other cross-encoders or embedders, query rewriting or
expansion, context-prefixed chunks with the neural stack, a larger top-k of units for the
generator, and per-question depth.

## Outcome (scored runs; raw output in `docs/sota-runs/retrieval_miss_*.txt`)

Runs: `ragkit retrieval-miss tune` (tuning set: fresh 29 + synthetic 40; frozen in
`docs/sota-runs/retrieval_miss_tuning.json`), `ragkit retrieval-miss score` (held-out pooled 130
questions, one run; `docs/sota-runs/retrieval_miss_heldout.txt`) and `ragkit retrieval-miss
diagnose` (`docs/sota-runs/retrieval_miss_diagnosis.txt`).

### 1. Diagnosis of the misses (descriptive)

A miss is an answerable question with no gold article among the first five chunks of B0.
Length bounds (10th / 90th percentile of all 159 answerable EU questions): 14.0 and 31.0 words.

| set | misses | vocabulary mismatch | chunk boundary split | near-duplicate outranked | query too short | query too long | unexplained |
|---|---|---|---|---|---|---|---|
| original 42 (42) | 6 | 4 | 1 | 0 | 1 | 0 | 0 |
| hard 39 (39) | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| keyword-free 49 (49) | 21 | 21 | 0 | 0 | 0 | 0 | 0 |

Primary cause for 29 of the 31 misses across the 130 held-out questions is vocabulary mismatch
(under 50% query content terms in the whole gold article). On the original 42:

1. Q1 (`reg_2019_943_art19`): chunk boundary split (term coverage 0.50 article vs 0.25 chunk;
   13 words, too short). Ranks: bm25 5 (c5), dense 9 (c14), hybrid 7 (c15), h+ce 12 (c24), F1 6 (c6).
2. Q2 (`dir_2019_944_art32`): vocabulary mismatch (term coverage 0.48). BM25, dense and hybrid all
   rank chunk 1 or 2; cross-encoder demotes it to c7 (article rank 6). F1 article units recovers
   it at article rank 3.
3. Q3 (`reg_2019_943_art17`): query too short (13 words, term coverage 0.50). BM25 rank 2, hybrid 2;
   cross-encoder demotes to c8 (article rank 6). F1 recovers at article rank 2.
4. Q4 (`reg_2017_1485_art23`): vocabulary mismatch (term coverage 0.17; 34 words, too long).
   Ranks: bm25 20 (c27), dense 6 (c10), hybrid 4 (c4), h+ce article rank 5 but chunk rank c7
   (5 chunk slots filled by earlier articles).
5. Q5 (`dir_2019_944_art40`): vocabulary mismatch (term coverage 0.25; 13 words). Ranks: bm25 5
   (c5), dense 10 (c18), hybrid 11 (c22), h+ce 7 (c16), F1 6 (c6).
6. Q6 (`reg_2015_1222_art72`): vocabulary mismatch (term coverage 0.25). BM25 rank 2, hybrid 3,
   h+ce article rank 2 but chunk rank c9 (chunk pushed past top-5 chunks by earlier articles).
   F1 article units recovers at article rank 2.

Near-duplicate outranking (max bigram Jaccard with an outranker >= 0.30) fired on 0 of the 6
misses (highest Jaccard was 0.22 on Q2 with a same-act article).

### 2. Tuning outcome (fresh 29 + synthetic 40, hybrid+ce)

Objective: mean of the two set-level MRR values. B0 objective: 0.9765 (fresh MRR 0.966,
synthetic MRR 0.988).

- F1 (article units): objective 0.9478 (-0.0287 vs B0; fresh MRR dropped to 0.908).
- F2 (depth): 50, 100, 200 all tied at 0.9765; tie-break chose 100 (= B0).
- F3 (fusion): RRF and all alphas in {0.1, ..., 0.9} tied at 0.9765; tie-break chose RRF (= B0).
- F4 (no query instruction): tied at 0.9765; fixed to `instruction=False` by definition.
- S (joint stack): all top cells tied at 0.9765; tie-break chose B0.

### 3. Held-out scoring outcome (pooled 130 questions: 42 original, 39 hard, 49 keyword-free)

Mean [bootstrap 95% CI over questions, 2000 resamples, seed 0]; paired differences against B0:

| system | configuration | MRR | Recall@5 | G@3 (prod) | G@5 (prod) | d MRR vs B0 | d G@5 (prod) vs B0 | wrong-act |
|---|---|---|---|---|---|---|---|---|
| B0 | `chunk/inst=on/rrf/D100/hybrid+ce` | 0.664 [0.596, 0.732] | 0.796 [0.727, 0.865] | 0.723 [0.646, 0.800] | 0.762 [0.685, 0.831] | baseline | baseline | 32 / 130 |
| F1 | `article/inst=on/rrf/D100/hybrid+ce` | 0.630 [0.561, 0.700] | 0.788 [0.719, 0.854] | 0.700 [0.623, 0.777] | 0.792 [0.723, 0.862] | -0.034 [-0.089, +0.016] | +0.031 [-0.023, +0.085] | 38 / 130 |
| F4 | `chunk/inst=off/rrf/D100/hybrid+ce` | 0.664 [0.596, 0.733] | 0.796 [0.727, 0.865] | 0.723 [0.646, 0.800] | 0.762 [0.685, 0.831] | +0.000 [-0.004, +0.004] | +0.000 [+0.000, +0.000] | 32 / 130 |

Synthetic guard: B0 0.988 [0.963, 1.000]; F1 dMRR +0.000 [-0.037, +0.037]; F4 dMRR +0.000
[+0.000, +0.000].

Sensitivity rows (pooled 130):
- Depth 50: MRR 0.662, dMRR -0.002 [-0.022, +0.012], wrong-act 32.
- Depth 200: MRR 0.659, dMRR -0.005 [-0.009, -0.001], wrong-act 32.
- Hybrid RRF (no CE): MRR 0.615, dMRR -0.049 [-0.097, -0.002], wrong-act 37.
- Sensitivity without original 42 (88 questions): B0 MRR 0.627; F1 dMRR -0.059 [-0.127, +0.007],
  wrong-act 33 vs 25; F4 dMRR -0.002 [-0.006, +0.000].

### 4. Decision rule verdicts

- **F1 article units**: not adopted (no evidence). Pooled MRR CI lower bound (-0.089) is not above 0;
  wrong-act count increased from 32 to 38.
- **F2 depth**: tuning selected B0 (no candidate to score); sensitivity shows D50/D200 do not help.
- **F3 fusion**: tuning selected B0 (no candidate to score).
- **F4 no query instruction**: not adopted (no evidence). Paired dMRR +0.000 [-0.004, +0.004] (CI
  includes 0; identical on 127 of 130 questions).
- **S joint stack**: tuning selected B0.

**Verdict: default unchanged (B0: chunk/inst=on/rrf/D100/hybrid+ce).**

### 5. Prediction assessment

- D1 partly held: vocabulary mismatch was primary for 4 of 6 (predicted >= 2) and chunk split for
  1 (predicted <= 1); near-duplicate was 0 (predicted >= 3; highest Jaccard 0.22 < 0.30).
- D2 did not hold on the 6 misses: BM25 ranked the gold article better than dense on 4 of the 6
  questions (dense only beat BM25 on Q4: rank 6 vs 20).
- P-F1 partly held: G@5 (prod) rose by +0.031 as predicted, but wrong-act rose (+6) rather than
  falling, MRR fell by -0.034, and S selected B0, not article units.
- P-F2 held: depth had no detectable pooled MRR effect; tuning picked 100.
- P-F3 held: convex vs RRF had no detectable effect on hybrid+ce; tuning picked RRF.
- P-F4 held: instruction off had no detectable effect (+0.000 [-0.004, +0.004]).
- P-adopt held: no candidate passed the adoption rule.
