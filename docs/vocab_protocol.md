# Vocabulary-mismatch protocol (K-R5): larger models and LLM query expansion (frozen before scoring)

Problem. K-R3 (`docs/retrieval_miss_protocol.md`) found that 29 of the 31 retrieval misses of the
default retriever (`hybrid+ce@100`: bge-small + BM25 fused by RRF, MiniLM cross-encoder over the
top 100) on the 130 held-out EU questions are **vocabulary mismatch**: under half of the question's
content terms occur in the gold article. None of its four fixes (article units, depth, fusion,
query instruction) passed. Its "not tried" list named the two remaining kinds of fix: other
embedders / cross-encoders, and query rewriting or expansion. This protocol tests both, retrieval
only: no answer generation.

This file is committed before any code of the study, before any model download and before any
expansion is generated. `git log` shows the order: this protocol, the code, the expansion cache,
the tuning run (which fixes the chosen settings in `docs/sota-runs/vocab_tuning.json`), the single
held-out run, then the documentation and any default change. Numbers are reported whatever they
turn out to be; "no detectable effect" means the paired 95 % CI includes 0.

## What is already known (disclosed taint)

- The 130 held-out questions were scored by K-R3 (MRR, G@5, wrong-act per system) and by the
  packing study (gold in packed context). Their misses are known to be mostly vocabulary
  mismatch: that is the motivation of this study, so they are **not clean**. No setting below is
  chosen on them.
- The fresh 29 and synthetic 40 were the tuning sets of K-R3 and of the packing study; they are
  the tuning sets here too and are never scored for a decision.
- No number of any new model or of any expansion exists on any set; no expansion text has been
  generated.

## Data

| role | set | questions | corpus |
|---|---|---|---|
| tuning | fresh (`qa_eu_fresh.jsonl`) | 29 | EU, 94 articles (`data/corpus_eu`, the corpus of K-R3 and the packing study) |
| tuning and guard | synthetic (`qa.jsonl`) | 40 | synthetic, 25 documents (each its own "article") |
| held-out | original (`qa_eu.jsonl`) | 42 | EU |
| held-out | hard (`qa_eu_hard.jsonl`) | 39 | EU |
| held-out | keyword-free (`qa_eu_nokw.jsonl`) | 49 | EU |

Pooled held-out = 130. Labels are agent-reviewed (README caveat). No question file is edited.

## Systems

Every system ranks the K-R3 chunks (recursive 500 characters, 50 overlap, no context prefix) with
the B0 pipeline: BM25 top 100 and dense top 100 fused by RRF (k = 60), cut to 100, the
cross-encoder reranks all 100. Four fields vary:

| field | values | B0 |
|---|---|---|
| `embedder` | `bge-small` (`BAAI/bge-small-en-v1.5@5c38ec7`, 33 M), `bge-base` (`BAAI/bge-base-en-v1.5@a5beb1e`, 110 M, MIT); both with the bge query instruction `Represent this sentence for searching relevant passages: ` and max sequence length 512 | `bge-small` |
| `reranker` | `minilm` (`cross-encoder/ms-marco-MiniLM-L-6-v2@233902d`, 22 M), `bge-reranker` (`BAAI/bge-reranker-base@2cfc18c`, 278 M, MIT); raw logit, max length 512 | `minilm` |
| `expansion` | off, on | off |
| `ce_query` | `original` or `expanded` (only when expansion is on) | `original` |

Grid: 2 x 2 x (1 + 2) = 12 cells. The two new models were approved for download by the user on
2026-11-17 (about 0.44 GB and 1.1 GB); no other model is downloaded.

**Expansion (query2doc / HyDE style).** For each question one passage is generated once by
Qwen3.6 27B (Ollama tag `qwen3.6:27b-ctx4k`, Q4_K_M, on the lab server; `num_ctx` 4096,
temperature 0, seed 0, `max_tokens` 256, thinking off). System prompt, frozen:

> You write one short passage that answers the question as it would be worded in EU
> electricity-market legislation or in a technical document on energy flexibility. Use the formal
> terms such a text would use. Write only the passage, 80 to 120 words: no preamble, no title, no
> citations, no mention that the passage is hypothetical.

User message: the question text, verbatim. With expansion on, the first stage uses
`q' = q + "\n" + passage`: the BM25 query is the question repeated five times followed by the
passage (query2doc's sparse form, which keeps the question's terms dominant), the dense query is
`q'` with the instruction. `ce_query = original` scores (question, chunk) pairs; `expanded` scores
(`q'`, chunk). An empty reply, or a call that fails after the transport's retries, makes that
question fall back to `q` (the count is reported). Every call is stored in
`docs/sota-runs/vocab_expansions.jsonl` (`ragkit.llm.CachedMessages`), so tuning and scoring replay
offline and the passages are public. The prompt is not tuned and no passage is read before the
held-out run.

## Metrics

Per system and question, as K-R3: the first 10 chunks deduplicated to articles (`list`) give MRR,
Recall@5 and wrong-act; the first 5 chunks deduplicated (`prod`) give `G@3 (prod)` and
`G@5 (prod)`. New here, the serving metric: **pack recall**, 1 when the budget-packed context that
`ragkit.search` builds holds a gold article: parent articles of the first 10 chunks, policy `stop`,
12 000 estimated tokens (the frozen setting of `docs/sota-runs/packing_tuning.json`). Means with a
95 % percentile bootstrap (2000 resamples, seed 0); paired differences against B0 with the paired
bootstrap and better / worse / tied counts.

## Candidates (frozen here)

| id | change from B0 | free parameter (tuning sets only) |
|---|---|---|
| M1 | `embedder = bge-base` | none |
| M2 | `reranker = bge-reranker` | none |
| Q1 | `expansion = on` | `ce_query` |
| S | joint: the best of the 12 cells | the cell |

**Tuning.** All 12 cells are scored on the fresh and the synthetic questions. Objective of a cell:
the mean of its two set-level MRR values (K-R3's objective). Ties (difference below 1e-9) go to the
higher mean pack recall, then to the cell with fewer fields different from B0, then to
`ce_query = original`, then to the smaller models. Q1 picks `ce_query` with the other fields at B0;
S is the argmax of the grid. If S is B0 or equals M1, M2 or Q1, it is reported and not scored
twice. The choices are written to `docs/sota-runs/vocab_tuning.json` and committed **before** the
held-out run, which reads them from that file.

**Held-out scoring (once).** B0 and every candidate, on the original 42, hard 39 and keyword-free
49, each alone and pooled; the synthetic guard for each. One run, no re-run after looking.
Non-gating rows in the same run: all 12 cells pooled, the pooled tables without the original 42
(88 questions), and the count of expansion fallbacks.

## Decision rule (per candidate; K-R3's rule plus the serving metric)

A candidate X is **adopted** iff all five hold on the pooled 130:

1. paired MRR difference X minus B0: 95 % CI lower bound above 0;
2. wrong-act count of X not above B0's;
3. `G@5 (prod)` difference, point estimate at least 0;
4. pack-recall difference, point estimate at least 0 (a ranking gain that loses gold articles from
   the context the agent reads does not count);
5. synthetic guard: paired MRR difference at least -0.02 and its CI upper bound not below 0.

**Rejected** when the pooled MRR CI lies entirely below 0 or condition 5 fails; otherwise **not
adopted (no evidence)**. If several are adopted, the highest pooled MRR difference point estimate
wins. Four candidates, four tests, no multiplicity correction (about a 10 % chance that at least
one passes condition 1 under the null).

**What an adoption changes.**

- A model-only winner (M1, M2, or S without expansion) becomes the default of
  `ragkit.search(retriever="hybrid+ce")` and `ragkit answer --retriever hybrid+ce`; the prebuilt
  index is rebuilt; the `hybrid+ce` answerable threshold is re-derived by the rule that set it
  (`abstention.best_threshold` on the 87 EU questions, `_search.py`). The agent evaluations
  (`docs/agent_eval_*`) are **not** re-run: their numbers stay those of the old models, and the
  README says so.
- Expansion needs an LLM at query time, so it never becomes the default: if Q1 (or an S with
  expansion) is adopted, it ships as an opt-in `expand` argument of `ragkit.search`, documented
  with these numbers. The model part of such an S becomes the default only if its model-only
  candidate (M1 or M2) is itself adopted.
- Nothing adopted: the defaults stay.

## Predictions (written before any run)

- P-M1: bge-base raises pooled MRR by about +0.02 with a CI that includes 0 (the cross-encoder
  re-sorts the first stage, so a better embedder mostly changes the pool, K-R3 found depth and
  fusion do not matter).
- P-M2: bge-reranker-base gives the largest model gain (the MiniLM cross-encoder demoted the gold
  article on 3 of the 6 original-42 misses); about +0.03 to +0.05 MRR, CI lower bound near 0.
- P-Q1: expansion helps most on the keyword-free 49 (the set built to have no shared keywords)
  and least on the original 42; tuning picks `ce_query = original`; pooled MRR gain +0.03 to
  +0.06, the only candidate with a fair chance to pass condition 1.
- P-S: the argmax cell has expansion on and `bge-reranker`.
- Pack recall is near its ceiling (111 of 130 for B0 in the packing study), so condition 4 moves
  by at most a few questions either way.

## Not claimed

Answer quality, latency and cost under the agent are not measured here. Expansion results hold for
this one LLM, prompt and decoding setting. The held-out questions are agent-drafted and
agent-reviewed, and were already used to find this problem.
