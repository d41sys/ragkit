# Article order (setup G) and the full corpus in the prompt (setup D) against the shipped setup E (issue #11): protocol, frozen before any G or D run

Date: 2026-11-16. User decision on issue #11: "cap", i.e. make the runs finish with a valid
baseline instead of running C and D under the #7 settings, where A had 9 failed runs (rule R3
allows 3) and C plus D needed about 20 h. This file is committed alone, before the code for G and
before any G or D run.

## Why no new turn cap

The cap was meant to stop the search loops that made A's unanswerable questions run 5 to 30 min
and fail. Setup E (`agent_eval_e_protocol.md`), which shipped by its pre-written rule, already
does that in production: its tool text says "do not keep searching (at most one rephrased
search)", and in its run the agent searched at most twice per question, 87 of 87 questions
finished, 0 failed, 110 s per question. A harness-only cap would make the agent differ from what
ships. So the baseline is **E's committed run** (`runs/agent_eval/E.jsonl`, its judge cache and
verdicts; nothing of E is re-run or re-judged), and the two open questions of #7 are asked
against E:

| id | context the model gets | how |
|---|---|---|
| E (baseline, existing run) | `ragkit.search` defaults plus the coverage check | agent loop, tool; E's tool text |
| G (new) | E's response with the article list reversed (worst first) | agent loop, tool; E's tool text; the stand-in reverses `articles`, everything else (coverage fields, hint) is E's |
| D (as frozen in #7) | all 94 articles in the system prompt | one call, no tool, native `/api/chat`, `qwen3.6:27b-ctx128k` |

G is C of #7 moved onto the setup that ships. D is unchanged from `agent_eval_protocol.md`
(`ask-full`, `D_PREAMBLE`, the system prompt of `runs/agent_eval/A.jsonl.config.json`, which E
used too, `num_ctx` 131 072, `num_predict` 8192, temperature 0, seed 0).

## Runs

- Models, variants, judge, question file, metrics and statistics: as in `agent_eval_protocol.md`
  (agent `qwen3.6:27b-ctx64k` for G, `qwen3.6:27b-ctx128k` for D, judge `gemma3:27b`, digests as
  listed there, on the Korea server's Ollama 0.18.1 through the SSH tunnel). No model is
  downloaded or created; the variants exist.
- G: the driver, tool text and endpoint of E's run, ezcross-ems `feat/eu-law-answerable` at
  `2ae4af62` (`eu-law-run.ts`, `--num-ctx 65536`, 1800 s per question), against
  `python -m ragkit.agentbench serve --setup G`. One stream, all 87 questions in file order, then
  one redo pass of failed rows (`--redo-failed`); a row that fails again stays failed (not
  correct, not refused).
- D: `python -m ragkit.agentbench ask-full` after G, one question at a time, 3600 s per call.
  Tried on `a01` first: if it does not run (out of memory, truncation guard, error) D is recorded
  as not run with the measured reason and G is unaffected. Failed rows are redone once like G's.
- GPU out of memory on the shared server is infrastructure: a pass that contains any `out of
  memory` row is moved aside unscored and restarted (as for F).
- Judging: one pass after all G and D answers exist (never agent and judge at once); A and E are
  replayed from the cache. `report --setups A,E,G,D`.

## Questions, predictions and what follows

Nothing ships from this study: a reversed list is not a candidate default, and the corpus cannot
go into every request (D is a reference). Paired 95 % bootstrap CIs (2000 resamples, seed 0)
against E, over the questions where both are defined.

1. **Order (G - E).** Does putting the worst article first change the answers?
   - P1: `correct(G) - correct(E)` CI includes 0 (the model reads every returned article).
   - P2: refusal on unanswerable G - E CI includes 0 (the coverage hint, not the order, drives it).
2. **Full context (D - E).** Does the whole corpus in the prompt beat retrieval?
   - P3: `correct(D)` is within 0.10 of `correct(E)` (point estimates).
   - P4: `refused(unanswerable, D)` is below E's 0.909 (D has no coverage hint and sees every
     near-miss article).
   - P5: D's latency per question is above E's 110 s (a 68 k-token prompt on every question).
3. **Validity.** G or D with more than 3 failed runs (`MAX_FAILED`) is reported as invalid for its
   question; its numbers are still printed.

Follow-ups, decided now: if a CI in 1 or 2 has its lower bound above 0 on `correct` (G or D beats
E), a new issue records it with the numbers; the endpoint does not change in this study. Anything
not listed here is labelled post-hoc.

## Not claimed

As in `agent_eval_protocol.md`: one agent model, an LLM judge, agent-reviewed labels, effects of
about 15 percentage points resolvable. G and D run on another day than E under another server
load, so latency and timeouts are not a controlled comparison; the answers are.

## Amendment 2026-11-16 (G running, before any D row and before any judge call)

- **D's refusal is not a clean contrast with E's.** D's instruction (`D_PREAMBLE`, "If no article
  supports an answer, say so plainly") is neutral like F's tool text, while E's tool text says the
  bundled corpus "likely does not cover" the question; F showed that wording alone moves refusal by
  5 of 44. So D - E on refusal mixes "full context vs retrieval" with "strong vs neutral refusal
  instruction". P4 and the D - E refusal row are descriptive; D's refusal is also read against F's
  0.795 (point estimates, descriptive). The D - E follow-up rule already uses `correct` only and is
  unchanged.
- **Transport.** D calls Ollama's native `/api/chat` with `temperature 0`, `seed 0` in the request;
  E and G go through `/v1` with the variant's `temperature 0`, `seed 0` and the runner's
  temperature 0. Same sampling settings, different API path; recorded, not tested.

## Results (2026-11-17; raw: `docs/sota-runs/agent_eval_gd.txt`, verdicts `agent_eval_gd_verdicts.json`)

Protocol `6f73969`, amendment `1dee20a`, code `149efd8`, script `7a305ea`; ezcross-ems
`feat/eu-law-answerable` at `2ae4af62` for G. G main pass 2026-11-16 22:33Z to 23:44Z (one
stream; the redo pass found nothing); D `a01` 115 s, then D to 01:13Z; judging (`gemma3:27b`, A and
E replayed from the cache) to 01:37Z. An earlier start failed before its first question because
the retrieval models were missing from the local cache (no row was written); they were downloaded
with the user's approval at the pinned revisions and the pipeline restarted from the start.
87 of 87 questions ran in G and in D, 0 failed, no GPU out-of-memory pass.

| metric (mean [95 % CI]) | E (baseline) | G (E, reversed) | D (full corpus) | G - E (paired) | D - E (paired) |
|---|---|---|---|---|---|
| correct (answerable) | 0.884 | 0.930 | 0.953 | +0.047 [-0.047, +0.140] | +0.070 [-0.023, +0.163] |
| wrongly refused (answerable) | 0.047 | 0.023 | 0.023 | -0.023 [-0.070, +0.000] | -0.023 [-0.070, +0.000] |
| refused (unanswerable) | 0.909 | 0.909 | 0.977 | +0.000 [+0.000, +0.000] | +0.068 [-0.023, +0.160] |
| cited articles judged `yes` | 0.976 | 0.988 | 1.000 | +0.012 [+0.000, +0.037] | +0.024 [+0.000, +0.061] |
| prompt tokens of the last call | 9 832 | 9 723 | 67 987 | | |
| latency, s per question | 110 | 49 | 61 | | |
| failed runs | 0 | 0 | 0 | | |

Misses by question: E not correct `a06` `a21` (partial), `a19` (no), `a24` `a43` (refused), not
refused `u09` `u38` `u42` `u44`; G not correct `a21` `a32` (partial), `a43` (refused), not refused
the same four as E; D not correct `a22` (partial), `a43` (refused), not refused `u25`.

**Rule:** G and D are valid (0 failed runs <= 3). Neither beats E on `correct` (lower bounds
-0.047 and -0.023), so no follow-up issue; nothing ships, as decided before the runs.

**Predictions:** P1 holds (G - E on correct includes 0). P2 holds (G refuses exactly the same
unanswerable questions as E). P3 holds (D within 0.10 of E: +0.070). P4 fails: D refuses more,
not less (43 of 44, against E's 40 and F's 35), with the neutral instruction of the amendment, so
the full corpus did not make the model answer from near-miss articles. P5 fails: D took 61 s per
question against E's 110 s and G's 49 s. *[Inference, not measured]* Ollama reuses the cached
68 k-token system prompt across questions (identical prefix; the first question, `a01`, took
115 s); E's and G's latency is not a controlled comparison either (other days, other server load).

**Read as:** for this model and this 94-article corpus, article order does not matter, and putting
the whole corpus in the prompt is at least as good as retrieval on every metric, with point
estimates higher on correct (+3 of 43) and refusal (+3 of 44), resolved on neither. It costs a
68 k-token prompt per question, which a local server with prefix caching absorbs and a paid API
would bill on every call; the corpus is small enough that this is possible at all. The judge is
one local model; the author's read of 15 judged answers (original protocol) is not done yet.
