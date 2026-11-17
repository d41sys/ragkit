# Contextual retrieval on the shipped chunks (K-R8), frozen before any code or run

Problem. K-R7 (`docs/contextual_protocol.md`) showed that Anthropic's chunk context helps a lot on
long documents (LegalBench privacy policies and NDAs: budget recall +0.145 [+0.103, +0.191]), but on
the EU articles its candidate also enlarged the chunks to 1600 characters, and larger chunks alone
lost EU MRR (-0.039 [-0.079, -0.001]). This study tests the context **on the shipped 500 / 50
chunks**, so the chunk size that works for EU stays, and decides whether it becomes the EU default.

## Disclosed prior knowledge

- The hypothesis comes from K-R7's EU result (chunk size hurt, context partly recovered). The same
  190 EU questions are scored again; no question was used to choose a setting here, and there is
  nothing to tune. They have now been scored in K-R3, K-R5, packing, K-R6 and K-R7.
- B0 on these 190 is known: MRR 0.681, wrong-act 46, pack recall 159 / 190 (K-R7).
- Closest prior: the model-free heading prefix (act title plus article heading prepended to these
  same 500 / 50 chunks, `ragkit.ctxprefix`) was not adopted on three sets and fixed none of BM25's
  wrong-act top-1 errors on the keyword-free set (README, "Context prefix on a keyword-free set").
  It ran only with BM25 and the hashing hybrid, never with `hybrid+ce`.

## Systems

Both are the shipped `hybrid+ce@100` (ragkit BM25 and bge-small top 100, RRF k = 60, MiniLM reranks
the top 100) over `corpus_eu_consolidated`, chunked recursive 500 / 50 (982 chunks).

| id | chunk text |
|---|---|
| B0 (shipped) | the chunk |
| C8 | Qwen context + `"\n\n"` + the chunk, used by BM25, the embedder and the cross-encoder |

Context generation exactly as K-R7 (Anthropic's prompt, `ragkit.contextualbench.CONTEXT_PROMPT`;
the document is the article file; `qwen3.6:27b-ctx4k`, temperature 0, seed 0, at most 200 output
tokens, thinking off), except `num_ctx` 8192: the longest article is 13 253 characters, so every
request fits and the model needs less GPU memory. `num_ctx` is not part of the cache key. Calls are
cached in `docs/sota-runs/contextual_small_contexts.jsonl` (committed). An empty reply leaves the
chunk without context (count reported).

## Decision rule

C8 is **adopted** iff, on the 190 pooled EU questions (`qa_eu` 42, `qa_eu_hard` 39, `qa_eu_nokw` 49,
`qa_eu_confirm` 60):

1. paired MRR difference C8 - B0, 95 % percentile bootstrap (2000 resamples, seed 0), lower bound
   above 0;
2. wrong-act count not above B0's, and pack-recall difference point estimate at least 0.

**Rejected** when the MRR CI lies entirely below 0; otherwise **not adopted (no evidence)**.

There is no check on another collection: an adoption changes only corpora that ship contexts, and
only the EU corpus will. Other corpora are indexed as now.

## What an adoption changes

The EU contexts ship as a data file; `ragkit.search` on the EU corpus indexes context + chunk for
the neural retrievers (no LLM at search time); the prebuilt index is rebuilt; the `hybrid+ce`
answerable threshold is re-derived by its existing rule (`abstention.best_threshold` on the 87 EU
questions). The agent evaluations are not re-run (README says so). Any other corpus is unchanged.

## Also printed (not gating)

Per EU set: MRR, Recall@5, G@5 (prod), pack recall, wrong-act with paired differences; mean context
length; empty contexts; chunk texts over 512 bge-small tokens.

**Amendment (committed while the contexts were being generated, before any score).** A control
arm **P0** = the heading prefix of `ctxprefix` (`build_chunks(..., prefix=True)`) on the same
chunks with `hybrid+ce`, scored alongside and judged by the same rule, but **not gating**: it is
printed to show whether C8's effect, if any, is only the act and heading names, which would need no
LLM to ship.

## Predictions (written before any run)

- P1: C8 gains, mostly on wrong-act (the context names the act for chunks deep in an article):
  pooled MRR +0.01 to +0.05; CI above 0 is a coin flip.
- P2: the gain is largest on the keyword-free set (lay wording, the context adds formal terms).
- P-cost: 982 calls, about 70 to 90 minutes.

## Not claimed

Answer quality through the agent; corpora other than EU; other prompts or LLMs.
