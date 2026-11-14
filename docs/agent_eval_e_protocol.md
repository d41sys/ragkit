# Coverage check in the agent (issue #12, setup E): protocol, frozen before any scored run

Date: 2026-11-14. Local models only, no paid API. Follow-up to [#7](agent_eval_protocol.md): the
agent answered 12 of the 44 unanswerable questions and did not finish 9 (timeouts, context
overflow, empty answer after a thinking run). This file is committed alone, before the first E
run; the code it describes (`agentbench.py`, setup E) and the ezcross-ems branch are committed
right after it and before the first E run.

## What changes (setup E = A + coverage check)

Everything of setup A in `agent_eval_protocol.md` stays: models and digests (`qwen3.6:27b`,
variant `qwen3.6:27b-ctx64k` with `num_ctx` 65536, temperature 0, seed 0, thinking on,
`max_tokens` 8192, per-question timeout 1800 s), the CLI system prompt, the 87 questions
(`runs/agent_eval/questions.jsonl`, sha256 `6bc34fb1f615013a10a4bcd34ebfc79cec9a20728c82307d56c1de55bf874d40`),
the judge (`gemma3:27b`, same prompts, same cache file), the metrics and the statistics. What
differs, in ezcross-ems branch `feat/eu-law-answerable` (stacked on `feat/search-eu-law`, PR #239):

1. `POST /api/v1/regulation/search` calls `ragkit.search(..., answerable=True)` and adds three
   fields to the response: `answerable_score` (reader score on the top 5 chunks),
   `answerable_threshold` (`ANSWERABLE_THRESHOLDS` of the retriever that ran) and `answerable`
   (`score >= threshold`; `null` when the retriever has no threshold, e.g. the bm25 fallback).
2. `search_eu_law` puts the three fields in its output and a negative check changes the hint, and
   the description gets one sentence. Description as sent (sha256
   `428f0df093697206e489fe9f4e0715df9aa2bb963b2628298e90b809dc8d836d`); the added part is the
   `plus a coverage check` clause and the sentence that starts `If the coverage check is false`:

   > Search EU law (articles of EU regulations and directives on energy, data and AI) and return the best-matching articles with id, act, article number, heading, url, corpus version and text, plus a coverage check (`answerable`). Retrieval only: write the answer yourself from these articles. Cite every claim by article id and url, and quote the corpus version when it matters. If the coverage check is false, the bundled corpus likely does not cover the question: tell the user so and do not keep searching (at most one rephrased search). If no returned article supports an answer, say so plainly; do not answer EU-law questions from memory.

   Hint when `answerable` is `false` (otherwise the hint of A):

   > Coverage check negative: the bundled EU corpus likely does not answer this question. Tell the user it does not cover it; do not keep searching (at most one rephrased search).

   Nothing forces the agent to obey: the check is advice in the tool output, as in production.
3. The harness code is the production one from that branch. The stand-in server
   (`python -m ragkit.agentbench serve --setup E`) returns exactly the endpoint's fields: the same
   `ragkit.search(..., answerable=True)` call, the same `answerable` rule, the same article
   text path; no setup shaping (A's list, as in A).

## Threshold source and what it means

`ANSWERABLE_THRESHOLDS["hybrid+ce"] = -0.7443351745605469` (`_search.py`): the value
`abstention.best_threshold` picks on **all 87 questions of the earlier EU set** (42 answerable,
45 unanswerable; README "Library use"). It is **in-sample**, and it is **unstable across folds**:
the five fold thresholds of the 5-fold CV for `hybrid+ce` range from +0.39 to -2.78. On that set
the reader with this threshold answers 9/45 unanswerable questions and refuses 7/42 answerable
ones in-sample; in the held-out CV rows it is 12/45 answered and 12/42 answerable refused. Today's
question set is the consolidated corpus with re-reviewed labels (43 + 44), so the threshold was not
fitted on it, but the two sets overlap in most of their questions and the numbers below are not
clean held-out numbers. The threshold is not re-tuned for this study.

## Runs

Setup E on all 87 questions, one worker (one stream on the shared server), in file order,
timeout 1800 s. A failed run (timeout, driver error, truncation guard, empty final text) is run
once more with the same settings (`--redo-failed`); if it fails again it stays failed (not
correct, not refused), as in A. Answerable questions are included to see wrongly refused answers.
`runs/agent_eval/E.jsonl`, `E.jsonl.config.json`, `E.server.jsonl` are committed like A's. Judging
is one pass over E with `gemma3:27b` after all E answers exist (never both models at once).

## Metrics

As in `agent_eval_protocol.md`; plus, descriptive only, how many questions got a negative
coverage check on their first search, by question kind.

## Decision rule (E against the existing A run)

E ships (endpoint and tool change stay in the PR for review, issue #12 closes) iff all of:

1. R1: the paired 95 % CI (bootstrap, 2000 resamples, seed 0) of
   `refused(unanswerable, E) - refused(unanswerable, A)` has a lower bound above 0;
2. R2: `correct(answerable, E) >= correct(answerable, A) - 0.05` (point estimates; A has 40 of 43,
   so E must have at least 38 of 43, at most 2 lost);
3. R3: E has at most 9 failed runs (A's 9).

If the rule fails the stacked PR stays a draft with the numbers and issue #12 stays open.
A's rows are the ones already committed (`runs/agent_eval/A.jsonl`, judge cache and verdicts);
nothing of A is re-run or re-judged.

## Predictions (before any E run)

- P1: the check is negative on about 7 of the 43 answerable questions and about 35 of the 44
  unanswerable ones (the in-sample rates above, on a partly different question set).
- P2: refusal on unanswerable rises clearly above A's 0.523 (R1 passes) and failed runs fall,
  because the agent stops searching.
- P3: R2 is at risk: if the agent obeys a negative check on answerable questions that are flagged,
  E loses more than 2 of the 43 correct answers.

## Differences between E and A (and from production)

- **A ran on another day under another server load, with two concurrent streams; E runs one
  stream.** Latency and the number of timeouts are not comparable to the same degree as the
  answers; Ollama batching also means a rerun of A would not be bit-for-bit identical.
- A's per-question timeout was 900 s for part of its rows and 1800 s for the rest (see the A
  write-up); E uses 1800 s throughout.
- The backend stand-in replaces FastAPI (no JWT, no database, no threadpool), as in A; the
  endpoint change itself is tested only with a stub of `ragkit` (`tests/test_regulation.py`), not
  against a live model.
- The numbers are judged by `gemma3:27b`, not comparable with the earlier Sonnet-judged numbers.

## Not claimed

That the threshold is right (in-sample, unstable); that the effect holds for other models; that
E is better at telling the user *why* the corpus does not cover a question (not measured).
