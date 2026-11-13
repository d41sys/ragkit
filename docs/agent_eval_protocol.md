# End-to-end EU-law answering through the ezcross-ems agent (issue #7): protocol, frozen before any scored run

Date: 2026-11-13. Local models only, no paid API. This file is committed alone, before the first
scored run; the code it describes (`src/ragkit/agentbench.py`, the REFUSE prompt in
`generation.py`, `ollama_transport` in `llm.py`) and the agent driver
(`harness/packages/ezc-cli/src/eval/eu-law-run.ts` in d41sys/ezcross-ems, branch
`feat/search-eu-law`) are committed right after it and before the first scored run. A smoke test
on question `a01` (not scored, not kept) came first and fixed the settings below.

## Problem

Every measured answer number used ragkit's own `GEN` prompt with Sonnet. In ezcross-ems the agent's
model and system prompt write the answer, so those numbers do not carry over. Not measured so far:
answer quality inside the agent with budget packing vs three articles, best-first vs reversed
article order, and the whole corpus in the prompt.

**Comparability.** The models, prompts, judge and question file differ from every earlier study
(Sonnet, `GEN`, 42 + 45 original questions). The numbers below are **not comparable one-to-one**
with those; only the comparisons between the setups of this study are the point.

## Models

| role | model | digest (`/api/tags`) |
|---|---|---|
| agent | `qwen3.6:27b` (Q4_K_M, 27.8 B, capabilities completion, vision, tools, thinking) | `a50eda8ed977ab48a12431878896b27ffd5cef552c17af3317d9623b939a7f1e` |
| judge | `gemma3:27b` (Q4_K_M, 27.4 B; a different family, so the agent is not graded by itself) | `a418f5838eaf7fe2cfe0a3046c8384b68ba43a4435542c942f9db00a5f342203` |

Both run on the Korea server's Ollama 0.18.1 (4 x RTX 4090, shared with other users' jobs), reached
through an SSH tunnel at `http://localhost:21434`. Concurrency is at most 2 requests. All agent
answers are generated first, all judging afterwards; the two models never run at the same time.

### Context length (num_ctx)

Ollama's OpenAI-compatible `/v1/chat/completions` ignores `num_ctx` and silently cuts a prompt
that does not fit the model's context (checked: a 10.6 k-token prompt sent with `num_ctx 4096`
returns `prompt_eval_count = 4096` and loses the first lines). The agent path (`ezc-cli`'s `local`
provider) only speaks `/v1`, so the context is fixed in the model instead: variants of
`qwen3.6:27b` made with `POST /api/create` (`from: qwen3.6:27b`, only parameters added; the base
model and other users' jobs are untouched). The Modelfile of each is the base one plus
`PARAMETER num_ctx N`, `PARAMETER temperature 0`, `PARAMETER seed 0`; the rest is inherited
(`min_p 0`, `presence_penalty 1.5`, `repeat_penalty 1`, `top_k 20`, `top_p 0.95`,
`RENDERER qwen3.5`, `PARSER qwen3.5`, `TEMPLATE {{ .Prompt }}`):

| setups | variant | num_ctx | digest |
|---|---|---|---|
| A, B, C (agent loop, `/v1`) | `qwen3.6:27b-ctx64k` | 65 536 | `c2aa416d32b7f0ef6dbc68ace23aa1442037de3b4b197691ec3c13f99148dd9b` |
| D (no tool, native `/api/chat` with `options.num_ctx`) | `qwen3.6:27b-ctx128k` | 131 072 | `e93fcd88ee5f8007901fb08c291a73b34e2d428e69e2fdaecafbfd97002cd52f` |

64 k for A to C leaves room for a prompt of up to 32 000 estimated article tokens (the largest
`max_tokens` the tool accepts) plus thinking. D's prompt is the corpus, 353 kB, about 68 k tokens
at 5.2 characters per token (measured: 10 649 tokens for 55 444 characters of corpus).
`/api/ps` shows the loaded `context_length` (65 536 in the smoke test).

### Truncation guard

A call fails the run when its recorded prompt token count (`usage.prompt_tokens` /
`prompt_eval_count`; the smoke test showed it counts cached prefix tokens too) is any of: at least
`num_ctx - 8` (the prompt filled the context), prompt plus completion at least `num_ctx`, or below
`0.8 x characters / 5.2` (a prompt far smaller than its size implies). It runs on every model call,
in `eu-law-run.ts` (A to C; characters = serialized messages plus tool schemas) and in
`agentbench.ask_full` (D). A failed run is stored with an `error`, never judged, and counts as
failed (below). Smoke test: with a 4 k variant the guard failed the run (`prompt of 4096 tokens
fills num_ctx 4096`); with the 64 k variant the same question passed, calls of 3 229 and
10 879 prompt tokens.

## Decoding (fixed by the smoke test)

- Temperature 0 (the ezc eval runner's `atTemperatureZero`, plus the variant's default) and
  `seed 0` (variant). Thinking is **on**: it is the model's default and the `local` provider cannot
  switch it off, so the production agent runs with it; `reasoning_effort` is not sent. One
  question takes about 70 s (2 model calls, about 1 k completion tokens including thinking).
- `max_tokens` 8192 per call (the `local` provider's fallback; thinking counts against it).
- Ollama may batch two concurrent requests; byte-for-byte determinism across runs is not claimed.

## Systems (the question set is the same for all)

Questions: `ragkit --corpus eu-consolidated` = `src/ragkit/data/qa_eu_consolidated.jsonl`
(43 answerable, sha256 `d0f4c775ae5643b3f0d664ed8dfdd888d62fba415c5e1d9e177b37cb0a2773fd`) and
`unanswerable_eu_consolidated.jsonl` (44, sha256
`4b113b8d0363bc45896c025b663b2c1698010732dc2e7b1c1cc994c65e1c430d`), corpus
`src/ragkit/data/corpus_eu_consolidated/` (94 articles). Labels are the agent-reviewed ones of
#4 and are not edited. Ids: `a01..a43`, `u01..u44` in file order
(`python -m ragkit.agentbench questions`).

Retrieval is the production one: `ragkit.search` with its default retriever `hybrid+ce` (bge-small
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` + BM25 by RRF, cross-encoder
`ms-marco-MiniLM-L-6-v2` `233902d25c440f23af6f7d6e94d2946bac0bee0a`, local cache, offline),
ragkit 0.2.1. Retrieval is not changed by this study.

| id | context the model gets | how |
|---|---|---|
| A | `ragkit.search` defaults: top 10 chunks, distinct parent articles packed in rank order into 12 000 estimated tokens | agent loop, tool |
| B | the three best articles | agent loop, tool; `search(pack_chunks=5)`, first 3 articles. Equals `ragkit answer --no-pack` (top 5 chunks, `parent_articles(k=3)`, same budget and `stop` policy): `pack_articles` is `parent_articles(k=None)`, and with policy `stop` the packed list is a prefix-superset of the k = 3 list |
| C | A's articles in reversed order (worst first) | agent loop, tool; the list is reversed before it is returned |
| D | all 94 articles, `=== <id> ===` blocks (`generation.corpus_block`), in the system prompt | one call, no tool, native `/api/chat` |

If the model passes `max_tokens` to the tool, it is honoured as in production (`budget_tokens`);
the calls are in the transcripts and the server logs. The agent is not told which setup it is in.

### The agent path and where it differs from production

Real: the `ezc-cli` session (`createCliSession`: shipped extensions, claim check, loop guard,
approval policy, `DEFAULT_SYSTEM_PROMPT`), the `local` provider with the openai-chat adapter, the
forecast tools and `search_eu_law` (all registered, as in production; the forecast tools are never
useful here but their schemas are part of every prompt), the typed backend client with a bearer
token, and `ragkit.search`. One fresh session per question.

Stand-in: `python -m ragkit.agentbench serve --setup A|B|C` replaces the FastAPI backend. It
answers `POST /api/v1/regulation/search` with the same JSON (`SearchResult.to_dict()`), the same
input validation (query 1 to 2000 characters, `max_tokens` 1000 to 32000, 422 otherwise), 401
without a bearer header, and calls `ragkit.search` directly. Skipped, with the reason they cannot
change an answer:

- JWT verification and the `User` row (`get_current_user`): the endpoint ignores the user, the
  result is a function of the query and `max_tokens` only;
- the database and the FastAPI/pydantic app: the validation is replicated, the response body is
  built from the same call;
- `run_in_threadpool`: the same synchronous call; the pipeline lock serialises concurrent calls;
- the backend audit mirror (`audit: false`): it records, it does not change the transcript;
- the setup shaping (B: first three, C: reverse) is the only code the production endpoint does
  not have.

Also different from production: the CLI's system prompt is used, not the ezc server's (which
reads "forecasting agent", "Use tools for all numbers. Never invent measurements or metrics.", the
same `ANSWER_GUIDANCE` and `ASK_USER_GUIDANCE`, and a line naming the user); neither prompt
mentions EU law, the tool description carries the instructions. D has no
act title, url or corpus version in its context (only the article text), and its instruction
(`D_PREAMBLE` in `agentbench.py`) mirrors the tool description's wording.

System prompt used (A to C; D prepends it to its corpus; sha256
`1e952ab04f6e1cb696548bc591ec9ad1281bf7e4e03af44d1b31e748effda72f`):

> You are the ezcross-ems forecasting assistant. Use tools for facts; never invent numbers. Quote numbers exactly as the tools return them: never compute, round, convert or re-express a number yourself. Tools do not report units; say the values are in the target's units and never convert them (for example kWh to MWh), and do not give conversion factors or formulas either. When asked 'how many percent' for the MAE change, quote `mae_delta_pct` from forecast_pipeline; if no tool returned a percentage, say so. Dispatch and actuation (sending a forecast or setpoint to a battery or any other device) is not available: say so plainly, do not ask the user to confirm or approve it, and offer to forecast and report instead. Never decide for the user. When run_forecast or forecast_pipeline is blocked by 'confirm' issues, call ask_user with their ids in confirm_issue_ids and wait for the answer: only the user's Accept lifts them, so never fill `acknowledged` yourself. Ask one question at a time and never answer your own question.

`search_eu_law` description as sent (sha256
`4034f14e28397d71feb1863ecbc225635f97a3d7a6e9a9fbdace283b369e2944`; parameters `query` 1 to 2000
characters, optional `max_tokens` 1000 to 32000):

> Search EU law (articles of EU regulations and directives on energy, data and AI) and return the best-matching articles with id, act, article number, heading, url, corpus version and text. Retrieval only: write the answer yourself from these articles. Cite every claim by article id and url, and quote the corpus version when it matters. If no returned article supports an answer, say so plainly; do not answer EU-law questions from memory.

`D_PREAMBLE` (setup D, after the system prompt, before the corpus):

> Reference corpus: the complete text of 94 EU-law articles (electricity market, energy), each headed by "=== <article id> ===". There is no search tool in this conversation: answer from these articles. Cite every claim by article id. If no article supports an answer, say so plainly; do not answer EU-law questions from memory.

The user turn is the bare question. The tool result the model sees is the production one: a
`<tool_data>` envelope with `outcome`, `data` (`query`, `retriever`, `answerable_score`,
`articles` with `id`, `act`, `article`, `heading`, `url`, `version`, `text`) and a `hint`.

## Runs

`runs/agent_eval/<setup>.jsonl` (force-added to git; `runs/` is otherwise ignored): per question
the id, the whole transcript (messages with thinking, tool calls and results), every model call's
prompt and completion tokens, characters and stop reason, wall latency and the final text;
`<setup>.jsonl.config.json` holds the model, settings and system prompt; `<setup>.server.jsonl`
holds every search request (query, `max_tokens`, returned ids, estimated tokens).

Order: A and B concurrently (one worker each), then C, then D, one question at a time per setup,
timeout 900 s per question. A failed run (timeout, driver error, guard, empty final text) is run
once more with the same settings (`--redo-failed`); if it fails again it stays failed. D is tried
on question `a01` first; if it does not run (out of memory, guard, error) it is recorded as not
run with the measured reason and the other setups are unaffected.

## Metrics

Per question, judged only on the final answer text (never on thinking):

- **refused** (all questions): `REFUSE` (new, below) says the answer declines.
- **correct** (answerable): `CORRECT` verdict `yes` (prompt of `generation.py`, verbatim, sha256
  `1469aa9ca927145f01469f10bb787564408710f937664292c55e5ede059b1a35`) against the reference answer;
  a refused answer is not correct (no `CORRECT` call), as an abstention was not in the earlier
  studies. `partial` is reported separately.
- **wrongly refused** (answerable): `REFUSE` yes. **Refusal on unanswerable**: `REFUSE` yes over
  the 44 unanswerable questions (the higher the better).
- **cite** (answerable, not refused): the cited articles of the answer are the article ids that
  appear verbatim (`reg_2019_943_art16`); if there is none, each `Article N` with a `YYYY/NNN` act
  number within 80 characters after it, else the last one within 80 characters before it, resolved
  to `reg|dir_YYYY_NNN_artN` if that id is in the corpus. Each cited article that is in the corpus
  and was returned by the tool in this run (D: any) gets one `CITE` call (verbatim prompt, sha256
  `fe84e7e4f09d519897c745ad8f875addd029ab7d92b18aeb163a47728cbc82e9`, full article text); any other
  cited id is `invalid` without a call (as in the earlier studies). The per-question score is the
  share of cited articles judged `yes` (0 if none are cited); "answers without citation" is
  reported next to it. Cite is secondary and never enters the decision.
- context: prompt tokens of the last model call (tokens the model saw when it answered); latency:
  wall seconds per question; tool use: share of questions with at least one `search_eu_law` call.

**`REFUSE`** is the only new judge prompt (a free-text answer has no `context_supports` flag; the
earlier studies counted flags). sha256 `288291ec4e490111ea3a68d4cd9cf924f795e656aa923cbc40bcec427c34adf3`:

```
You classify an answer to a question about EU electricity law. Decide whether the answer REFUSES
to answer: whether its message is that the available articles or sources do not answer the
question. Reply with one JSON object and nothing else:
{"verdict": "yes" | "no", "reason": "<one sentence>"}

- "yes": the answer says the articles or sources do not contain what is needed (or that it cannot
answer), even if it adds what they do state on the topic, and it does not commit to an answer to
the question as asked.
- "no": the answer commits to an answer to the question as asked, even with caveats, and even if
that answer is wrong.
```

(Line breaks above are the display's; the string in `generation.py` has the backslash-continued
lines of the other prompts.) The user turn is `Question: ...` and `Answer: ...`.

**Judge calls.** `CachedMessages` (`llm.py`) with the new `ollama_transport`: native `/api/chat`,
model `gemma3:27b`, `temperature 0`, `seed 0`, `num_ctx 32768`, `num_predict` 800 (the judges'
`max_tokens`), the `_json` retry of `generation.py`. A request whose size at 3 characters per token
plus `max_tokens` exceeds `num_ctx` is refused by the transport. Every call is cached in
`data/llm_cache/agent_eval_judge.jsonl` under the SHA-256 of its payload; `python -m
ragkit.agentbench report` replays it offline and prints the tables. Judge model and its
self-preference: the agent and judge are different families, but no human checked a sample of
verdicts. After scoring, the author reads 15 judged answers (a fixed seed-0 sample) and reports
how many verdicts he disagrees with; this is descriptive and changes no number.

## Statistics

Mean per setup with a percentile bootstrap 95 % CI over questions (`ragkit.eval.bootstrap_ci`,
2000 resamples, seed 0); paired differences X - A with `paired_bootstrap_ci` (same questions, same
resamples, seed 0), over the questions where both values are defined (answerable for correct,
wrongly refused and cite; unanswerable for refusal). 43 and 44 questions resolve differences of
roughly 15 percentage points only. A failed run counts as not correct and not refused.

## Decision rule (which setup becomes the agent's default)

The default is **A** (the current endpoint behaviour: `ragkit.search` defaults) unless a candidate
X in {B, C} satisfies all of:

1. R1: the paired 95 % CI of `correct(X) - correct(A)` has a lower bound above 0;
2. R2: refusal on unanswerable of X is not lower than A's by more than 0.10 (point estimates; at
   most 4 of 44 questions);
3. R3: X has at most 3 failed runs.

If both B and C satisfy all three, the larger paired `correct` difference wins. If A has more than
3 failed runs the study is reported as invalid for the default decision (A stays). D is a
reference: the corpus cannot go into every request, so D never becomes the endpoint default and
is reported with the same tables. Two candidates are tested without multiplicity correction;
anything not listed here is labelled post-hoc. If the rule picks B or C, the endpoint and tool
default is changed in the worktree to that setup in a separate commit and the results say so.

## Predictions (before any scored run)

- P1 The agent calls `search_eu_law` on at least 95 % of the questions in A to C.
- P2 B does not differ from A on `correct` (CI includes 0): the gold article is usually among the
  top three, and packing adds distractors.
- P3 C does not differ from A on `correct`: the model reads all articles.
- P4 D's `correct` is within 0.10 of A's.
- P5 Refusal on unanswerable is at most 0.60 in every setup (a model answers from its training
  data or from near-miss articles more often than it declines), so R2 is not the binding rule.

## Not claimed

Labels are agent-reviewed (README caveat), LLM judges are not human labels, one model of one
family is the agent, and the sample resolves only large effects. Results hold for
`qwen3.6:27b` with thinking and this system prompt; they say nothing about Claude or other
models, and the earlier Sonnet-judged numbers are not comparable (see above).
