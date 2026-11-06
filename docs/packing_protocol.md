# Budget packing of parent articles (Phase N-1): protocol, frozen before any scoring

Date: 2026-11-06. Retrieval only: **no generation, no LLM call, no API key, no download**. Only the
pinned bge-small and MiniLM cross-encoder from the local Hugging Face cache are used, offline
(`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, `uv run --offline`).

## Problem

`ragkit answer --context article` hands the generator the full text of the top `ARTICLE_K = 3`
distinct parent articles of the top-5 chunks (`src/ragkit/parent.py`), under a budget of
`ARTICLE_BUDGET_TOKENS = 12 000` estimated tokens (characters / 4). The budget is rarely close to
binding (mean context 3 232 to 3 790 tokens on the original 42 questions,
`docs/sota-runs/article_coverage_*.txt`), so when the top articles are short the budget is wasted.
The post-hoc k = 5 analysis (`docs/generation_protocol.md`, Phase L addendum) found k = 5 helped
answers somewhat, and was not adopted by rule. Six of the 42 original questions have no gold
article among the top-5 chunks, which caps every k because the articles are taken from those
chunks.

## Change (opt-in packing mode)

`pack_articles` (a mode of `ragkit.parent.parent_articles`) adds the distinct parent articles of
the ranked chunks **in rank order of their best chunk** until the budget is full:

- an article is added whole if `used + estimate_tokens(text) <= budget`; the budget is never
  exceeded (the sum of `estimate_tokens` of the context texts is at most the budget);
- **overflow policy** (the article that does not fit), one of two:
  - `stop`: the article and all later ones are dropped (the rule of the shipped k = 3 mode);
  - `skip`: only that article is dropped; later, shorter articles that still fit are added;
- the first article is always kept; if it alone exceeds the budget it is cut at the budget and
  packing ends (the shipped rule, unchanged);
- no limit on the number of articles other than the budget and the candidate chunks;
- **candidate chunks**: packing reads the top `D` chunks of the retriever (`D` = pool depth)
  instead of the top 5, otherwise it cannot reach articles beyond the top-5 chunks. For the default
  pipeline a deeper `retrieve(top_k=D)` also widens the candidate list its lexical reranker sees, so
  its first five chunks can differ from `retrieve(top_k=5)`; this is part of the mode and is
  measured, not corrected.

Token measure everywhere: `estimate_tokens` (characters / 4, rounded up), as for the budget.

## Systems compared (per retriever)

Retrievers: `default` (the shipped hashing hybrid + lexical rerank) and `hybrid+ce`
(`NeuralChunkRetriever("hybrid+ce")`: bge-small + BM25 by RRF, MiniLM cross-encoder over the top
100), both through the same `RAGPipeline` path `ragkit answer` uses.

| arm | chunks retrieved | articles | budget (est. tokens) |
|---|---|---|---|
| `k3` (shipped; baseline) | 5 | first 3 distinct, policy `stop` | 12 000 |
| `k5` (the K-R4 setting) | 5 | first 5 distinct, policy `stop` | 20 000 |
| `pack` | `D` | as many as fit, policy `O` | 12 000 |

`O` in {`stop`, `skip`} and `D` in {5, 10, 20} are the only free choices of the study. They are
chosen on dev questions only (next section), once, and then frozen.

## Data

| role | questions | corpus |
|---|---|---|
| dev (choose `O` and `D`) | fresh 29 (`qa_eu_fresh.jsonl`) and synthetic 40 (`qa.jsonl`) | EU 94 articles; synthetic 25 documents (each document is its own "article") |
| held-out (score once) | pooled original 42 + hard 39 + keyword-free 49 = 130 (`qa_eu.jsonl`, `qa_eu_hard.jsonl`, `qa_eu_nokw.jsonl`) | EU |

The fresh 29 and synthetic 40 are the tuning sets of the retrieval-miss study
(`docs/retrieval_miss_protocol.md`); no packing setting was ever looked at on them. The 130 pooled
questions are the held-out sets of that study (M-R). The EU question files are **not edited** (a
human review of the labels is pending); labels are agent-reviewed (README caveat).

Disclosed prior knowledge about the held-out set: for the original 42 only, gold coverage at k = 3
and k = 5 is already printed (`docs/sota-runs/article_coverage_*.txt`: 33 and 36 of 42 default,
34 and 36 of 42 `hybrid+ce`), and the top-5 chunk ceiling is 36 of 42. For the hard 39 and
keyword-free 49 sets no article-context coverage number exists. No per-question result of this
study has been seen.

## Dev selection (before any held-out number)

For every `(O, D)` in the 2 x 3 grid, and for the two retrievers, compute on each dev set the
share of questions whose context holds a gold article (`recall`) and the mean context tokens. The
objective of `(O, D)` is the mean of its four recall cells (2 retrievers x 2 dev sets). Ties
(difference below 1e-9) go to the lower mean of the four mean-token cells, then to `stop` over
`skip`, then to the smaller `D`. The chosen `(O, D)` is written to
`docs/sota-runs/packing_tuning.json` and committed **before** the held-out run, which reads it
from that file.

## Held-out scoring (once)

Both retrievers, 130 pooled questions (and each set alone, descriptive), arms `k3`, `k5`, `pack`
with the frozen `(O, D)`:

1. **Gold article in context**: share of the 130 questions whose context holds any gold article
   (the definition of `render_coverage`).
2. Context tokens (`estimate_tokens` summed over the context): mean, median, max, and the number
   of contexts above their budget (must be 0; any other value is a bug).
3. Paired bootstrap CI of the recall difference `k5 - k3` and `pack - k3` (per-question 0/1
   differences, 2000 resamples, seed 0, `ragkit.eval.paired_bootstrap_ci`), with the counts of
   questions gained / lost / tied. Non-gating, labelled secondary: `pack - k5`.
4. Number of articles in the context, mean and median (descriptive).

One run, no re-run after looking. Command (raw output committed under `docs/sota-runs/`):
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline ragkit packing score --out
docs/sota-runs/packing_heldout.txt` (the `tune` stage writes the JSON the same way).

## Decision rule

`ragkit answer` uses budget packing **by default** only if the recall difference `pack - k3`
(percentile paired bootstrap, 95 %) has a **CI lower bound above 0 on BOTH retrievers**
(`default` and `hybrid+ce`) on the pooled 130 questions, and no context exceeds its budget.
Otherwise packing stays opt-in (`--pack`) and the default stays `k3`.

If the rule passes, that is a statement about **retrieval-side gold coverage only. Answer quality
(correctness, citation precision, cost) was not re-measured**: that needs the API, which is out of
scope here. A larger context holds more distractors and costs more `GEN` tokens, and the k = 5
analysis showed citation `yes` rates did not rise; so a pass here is not evidence that answers
improve. When a CI includes 0 the result is reported as "no detectable effect". Any analysis not
listed above is labelled post-hoc.

## Predictions (before any scored number)

- P1 Packing raises gold coverage over `k3` on both retrievers, with a lower bound above 0 (the
  ceiling moves from the top-5 chunks to the top-`D` chunks).
- P2 Mean context tokens under packing are several times those of `k3` (articles are short, so the
  12 000-token budget admits far more than 5 of them), well above the `k5` mean.
- P3 `pack` and `k5` differ little in gold coverage if `D` is small, and `pack - k5` is positive
  only for larger `D`.

## Not claimed

Answer quality, citation behaviour, cost per question, other budgets, other corpora, any retriever
besides the two above, the correctness of the agent-reviewed labels.

## Outcome (scored once; raw output `docs/sota-runs/packing_heldout.txt`, dev grid `packing_tuning.txt` / `packing_tuning.json`)

Commands (offline, no API key, no download):
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline ragkit packing tune --out docs/sota-runs/packing_tuning.txt`
then `... ragkit packing score --out docs/sota-runs/packing_heldout.txt`. The tuning JSON was committed
(`b4a9f9f`) before the held-out run.

**Dev choice.** The dev sets are at the ceiling (dev recall 0.966 to 1.000 in every cell), so the
choice came from the tie-breaks: overflow `stop` (`skip` never raised dev recall and added tokens
only at depth 20, 10 824 against 10 595 on default/fresh) and pool depth 10 (depth 5 had a lower
recall cell, 0.975 on default/synthetic; depth 20 tied at 0.9914 with about twice the tokens).
Frozen: `PACK_OVERFLOW = "stop"`, `PACK_CHUNKS = 10`, budget 12 000.

**Held-out, pooled 130 questions** (gold article in context, mean / median / max estimated
context tokens; 95 % percentile bootstrap, 2000 resamples, seed 0):

| retriever | arm | gold in context | recall [95 % CI] | mean / median / max tokens | mean articles |
|---|---|---|---|---|---|
| default | k3 | 80/130 | 0.615 [0.531, 0.700] | 3526 / 3380 / 6816 | 2.93 |
| default | k5 | 86/130 | 0.662 [0.584, 0.738] | 4854 / 4756 / 9580 | 3.97 |
| default | pack | 96/130 | 0.738 [0.662, 0.815] | 8502 / 8738 / 11909 | 7.15 |
| hybrid+ce | k3 | 94/130 | 0.723 [0.646, 0.800] | 3229 / 3033 / 7324 | 2.78 |
| hybrid+ce | k5 | 99/130 | 0.762 [0.685, 0.831] | 4082 / 3814 / 9204 | 3.58 |
| hybrid+ce | pack | 111/130 | 0.854 [0.792, 0.915] | 6876 / 6886 / 11712 | 6.11 |

Paired recall difference (gained / lost / tied questions):

| retriever | k5 - k3 | pack - k3 (gating) | pack - k5 (secondary) |
|---|---|---|---|
| default | +0.046 [+0.015, +0.085] (6 / 0 / 124) | **+0.123 [+0.062, +0.185]** (17 / 1 / 112) | +0.077 [+0.023, +0.131] (12 / 2 / 116) |
| hybrid+ce | +0.038 [+0.008, +0.069] (5 / 0 / 125) | **+0.131 [+0.077, +0.192]** (17 / 0 / 113) | +0.092 [+0.046, +0.146] (12 / 0 / 118) |

No context exceeded its budget (0 of 130 in every arm and retriever).

**Rule: PASSES.** `pack - k3` CI lower bound is +0.062 (default) and +0.077 (`hybrid+ce`), both
above 0, and nothing is over budget. Per the rule, `ragkit answer` uses budget packing by default
(`--pack`, `--pack-chunks 10`, `--overflow stop`; `--no-pack` restores the k = 3 context).

**What this does not show.** Answer quality was **not re-measured** (that needs the API). Packing
costs context: mean tokens are 2.4 times (default) and 2.1 times (`hybrid+ce`) those of `k3`, and
more distractors per prompt; the k = 5 analysis found the citation `yes` rate did not rise with
more articles. The gain is retrieval-side gold coverage only. The keyword-free set carries most of
the gain on `default` (14 to 22 of 49) and a large part on `hybrid+ce` (25 to 34 of 49), the hard
set least (35 to 35 and 35 to 37).

Predictions: P1 held (both lower bounds above 0). P2 only partly: packing is 2.1 to 2.4 times `k3`
(1.8 times `k5` on default, 1.7 on `hybrid+ce`), not "several times". P3 held in the direction
stated (`pack - k5` is positive for both retrievers at depth 10; no small-`D` run was made, so
the "differ little at small D" half is untested).
