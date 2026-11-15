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

## Results (2026-11-15; raw: `docs/sota-runs/agent_eval_e.txt`, verdicts `agent_eval_e_verdicts.json`)

Protocol commit `6aeea6c`, code `0953d39`, ezcross-ems `feat/eu-law-answerable` at `2ae4af62`
(driver, tool and endpoint; the run used that checkout). Main pass 2026-11-14 22:54Z to
2026-11-15 01:34Z (2 h 40 m, one stream); the redo pass found nothing to redo (no failed row);
judging 01:34Z to 01:59Z (`gemma3:27b`, A's verdicts replayed from the cache). 87 of 87 questions
ran, 0 failed. **Not comparable one-to-one with the earlier Sonnet-judged numbers.** A ran on
another day under another server load.

| metric (mean [95 % CI]) | A | E |
|---|---|---|
| correct (answerable) | 0.930 [0.860, 1.000] | 0.884 [0.791, 0.977] |
| partial (answerable) | 0.047 [0.000, 0.116] | 0.047 [0.000, 0.116] |
| wrongly refused (answerable) | 0.000 [0.000, 0.000] | 0.047 [0.000, 0.116] |
| refused (unanswerable) | 0.523 [0.364, 0.659] | 0.909 [0.818, 0.977] |
| cited articles judged `yes` (answerable, answered) | 0.919 [0.837, 0.977] | 0.976 [0.939, 1.000] |
| articles cited per answer | 1.42 [1.21, 1.65] | 1.41 [1.22, 1.66] |
| answers without citation | 0.047 [0.000, 0.116] | 0.000 [0.000, 0.000] |
| agent called the tool (all) | 0.885 [0.816, 0.943] | 0.920 [0.862, 0.977] |
| prompt tokens of the last call (all) | 13939 [11521, 16704] | 9832 [9122, 10559] |
| latency, s per question (all) | 400 [301, 515] | 110 [99, 124] |
| failed runs | 9 | 0 |

Paired differences, E - A (paired bootstrap over questions):

| metric | E - A |
|---|---|
| correct (answerable) | -0.047 [-0.163, +0.047] (n=43) |
| partial (answerable) | +0.000 [-0.093, +0.093] (n=43) |
| wrongly refused (answerable) | +0.047 [+0.000, +0.116] (n=43) |
| refused (unanswerable) | **+0.386 [+0.227, +0.545]** (n=44) |
| cited articles judged `yes` (answerable, answered) | +0.049 [-0.024, +0.134] (n=41) |

Agent loop (mean / max per question; runaway = a call used all 8192 tokens):

| setup | kind | searches | max prompt tokens | runaways | n |
|---|---|---|---|---|---|
| A | answerable | 1.07 / 2 | 9872 / 22069 | 0 | 43 |
| A | unanswerable | 2.75 / 9 | 22400 / 64896 | 5 | 44 |
| E | answerable | 1.00 / 1 | 9535 / 13703 | 0 | 43 |
| E | unanswerable | 1.00 / 2 | 10122 / 22708 | 0 | 44 |

### Decision

- R1: refusal on unanswerable E - A = +0.386, CI lower bound +0.227 > 0: **pass**.
- R2: correct on answerable 0.884 (38 of 43) vs 0.930 (40 of 43); margin 0.05 (0.880): **pass**,
  with no room left (one more lost answer would fail it).
- R3: 0 failed runs <= 9: **pass**.

**E ships by the pre-written rule.** The endpoint and tool change stays in the stacked PR for
review.

### What the numbers do not say (read before relying on the check)

Post-hoc, descriptive; none of it changes the decision.

- **P1 was wrong: the check is negative on almost every question.** First search of each
  question: `answerable = false` on 40 of 43 answerable and 36 of 44 unanswerable questions
  (predicted about 7 and 35). The reader scores the query the agent writes, a keyword string such
  as "minimum share transmission capacity TSOs cross-zonal trade electricity", while the threshold
  was fitted on whole natural-language questions. Median first-search score: -5.06 (answerable,
  n=43) and -7.27 (unanswerable, n=37 with a search); the threshold is -0.74. The score separates
  the two kinds only weakly (AUC 0.656).
- So the gain does not come from a discriminating signal. The agent read a negative hint on 40 of
  43 answerable questions and still answered 38 of them correctly; it declined on 40 of 44
  unanswerable ones. The hint changes what the agent does after one search (1.00 searches on
  both kinds, against 2.75 on unanswerable questions in A), and it judged the articles itself.
  Whether a calibrated check (for example scoring the user's question, not the model's query)
  would do better is not measured.
- **A's 9 failed rows were unanswerable questions and count as not refused.** On the 35
  unanswerable questions A finished, A refused 23 (0.657) and E refused 31 (0.886): paired
  +0.229 [+0.086, +0.400]. The rule's R1 used all 44, as pre-registered.
- Wrongly refused answerable questions: E `a24`, `a43` (A: none). E did not answer correctly on
  `a06` and `a21` (partial) and `a19` (no); A on `a22`, `a24` (partial) and `a30` (no).
- E did not refuse unanswerable `u09`, `u38`, `u42`, `u44`.
- Latency fell from 400 to 110 s per question and no run failed, because the agent stops after
  one search; the latency is not a controlled comparison (other day, one stream against two).

## Setup F (hint only): protocol, frozen before any F run

Date: 2026-11-15. User decision after the E result: the check was negative on almost every
question, so its score carried little information (see "What the numbers do not say"). F keeps
only the instruction. This section is committed alone, before the F code and before any F run.

**What F is.** A's endpoint, unchanged: no `answerable` fields, no reader model, no third model
in the request path. The tool text changes, in ezcross-ems branch `feat/eu-law-search-hint`
(from `feat/search-eu-law`; tool text only, backend unchanged; E's branch
`feat/eu-law-answerable` stays as it is). The wording follows E's without naming a check.

Description as sent (sha256 `651bac3f3d8fa9256e474362ebfee3ae0f23485dea52a2de959f721f3db025a7`;
the added sentence is `If the returned articles do not answer the question, ...`):

> Search EU law (articles of EU regulations and directives on energy, data and AI) and return the best-matching articles with id, act, article number, heading, url, corpus version and text. Retrieval only: write the answer yourself from these articles. Cite every claim by article id and url, and quote the corpus version when it matters. If the returned articles do not answer the question, tell the user the bundled corpus does not cover it and do not keep searching (at most one rephrased search). If no returned article supports an answer, say so plainly; do not answer EU-law questions from memory.

Hint on **every** tool result (A's hint ended with "If none of these articles supports an
answer, say so instead of guessing."):

> Cite article ids and urls. If the returned articles do not answer the question, tell the user the bundled corpus does not cover it and do not keep searching (at most one rephrased search).

and, when a search returns no article:

> No article found: tell the user the bundled corpus does not cover it; do not keep searching (at most one rephrased search).

Unlike E the hint does not depend on a score: it is the same advice after every search, and the
agent decides from the articles whether they answer the question.

**Runs.** Same as E: all 87 questions, the same models, variant (`qwen3.6:27b-ctx64k`,
num_ctx 64k), temperature 0, seed 0, 1800 s timeout, judge `gemma3:27b`, same metrics, one
stream on the shared server, one redo pass of failed rows (a row that fails again stays failed:
not correct, not refused). Harness: `eu-law-run.ts` from `feat/eu-law-search-hint`; stand-in
`python -m ragkit.agentbench serve --setup F` (A's response, A's list, no shaping). A's rows and
E's rows are the ones already committed. A ran on another day under another server load, and so
did E; F is a third day.

**Decision rule.** *Corrected 2026-11-15 by user decision ("ship F if it matches E"), in the
commit after `adab848`, which had a weaker rule (F against A only); no F code or run existed
before the correction.* F ships iff **both**:

(a) F passes E's rule against A:

1. R1: the paired 95 % CI of `refused(unanswerable, F) - refused(unanswerable, A)` has a lower
   bound above 0;
2. R2: `correct(answerable, F) >= correct(answerable, A) - 0.05` (at least 38 of 43);
3. R3: F has at most 9 failed runs;

(b) F is non-inferior to E on the same questions (paired F - E, point estimates):

4. N1: `refused(unanswerable, F) >= refused(unanswerable, E) - 0.07` (at most 3 of 44 fewer);
5. N2: `correct(answerable, F) >= correct(answerable, E) - 0.05` (at most 2 of 43 fewer; together
   with R2 that means at least 38 of 43);
6. N3: F has at most 3 failed runs (`MAX_FAILED` of #7; E had 0).

If F passes (a) and (b) it ships instead of E (simpler: no third model, no extra latency, no
always-false field): a PR for `feat/eu-law-search-hint` against `feat/search-eu-law`, PR #241
closed with the numbers. If F fails (a) or (b), E ships as measured (PR #241 unchanged). The
paired F - E CIs are reported too; the rule uses the point estimates above.

**Predictions.**

- P1: F refuses more of the unanswerable questions than A and fewer than E (a hint that is the
  same after every search is weaker than one that was negative on 93 % of answerable questions);
  R1 is uncertain.
- P2: F answers the answerable questions as well as A (R2 passes with room, because the hint does
  not tell the agent the corpus fails).
- P3: failed runs fall below A's 9, but stay above E's 0.
