# Answers from paper contexts at 12 000 and 36 000 tokens (P-3), frozen before any code or run

P-2 ([`open_papers_protocol.md`](open_papers_protocol.md)) measured what reaches the reader: at a
36 000-token budget more evidence paragraphs (QASPER) and more needed papers (MDA-QA) are packed.
The user then made 36 000 the default for paged corpora. Not measured: whether the reader's
**answers** get better, or whether the extra papers distract it. P-3 measures that once, on
questions no earlier phase has touched.

## Data (fixed before any run)

| | QASPER | MDA-QA |
|---|---|---|
| Source | train split, `qasper-train-v0.3.json` (888 papers; P-2 used dev and test) | shuffled ids 200 to 300 (P-2 used 0 to 200) |
| Corpus | paper ids sorted, shuffled with `random.Random(0)`, the first 416 (P-2's test pool size) | the union of the 100 questions' support papers |
| Questions | those of the 416 papers with an answerable annotator answer with text evidence (964), in file order, shuffled with `random.Random(0)`, the first 300 | the 100 |
| Query | `In the paper "<title>": <question>` (as P-2) | the question |
| Reference answer | the first qualifying annotator: `free_form_answer`, else the `extractive_spans` joined by "; ", else "Yes" / "No" | the `answer` field (written by GPT-4o from the support papers) |

## Arms

Both use the shipped `hybrid+ce@100`, a pool of the top 10 chunks, overflow `stop`, whole papers
(`search()` as shipped). **A12** packs into 12 000 estimated tokens, **A36** into 36 000.

## Reader and judge

Both are the Codex CLI (`codex exec`, version 0.156.1), model `gpt-6-sol`, reasoning effort `low`
(the user's Codex default), one fresh session per call. The user's Codex config, hooks, `AGENTS.md`
and web search are off; the session runs in an empty directory with a read-only sandbox, so the
reader sees only the prompt. Codex adds its own instructions (15 693 input tokens in a pilot call)
and has no temperature or output-length setting; replies are cached, so each request is made once.

Reader prompt (`PAPER_INSTRUCTIONS`), followed by `Context passages:`, the packed papers as
`[paper id]` blocks in rank order, and `Question: <query>`:

```
You answer questions about research papers using ONLY the context passages provided. Each passage
is headed by the id of the paper it comes from, in square brackets.

Reply with one JSON object and nothing else:
{"answer": "<answer>", "citations": ["<paper id>", ...], "context_supports": true | false}

Rules:
- "answer": your best answer to the question as asked, in at most 120 words, using only the
passages. Do not use outside knowledge. If the passages do not contain what is needed, say what
they do state and that they do not answer the question.
- "citations": the ids of the papers whose passages you used. Use only ids that appear in the
context.
- "context_supports": true only if the passages contain what is needed to answer the question
definitively, false otherwise (also false if you could only guess, or if the passages merely
concern the same topic).
```

Judge: the `CORRECT` prompt of `ragkit.generation`, unchanged since K-R2 (question, reference
answer, candidate; verdict yes / partial / no). The judge does not see the arm. When both arms
give the same answer text, the request and so the verdict are the same.

## Metrics

- **Correct** (primary): share of questions whose answer the judge marks `yes`.
- Reported: `yes` or `partial`; `context_supports` true; gold paper cited (QASPER) and share of
  support papers cited (MDA-QA); Codex input tokens per reader call.

## Decision rule

Paired bootstrap over questions (10 000 resamples, seed 0, 95 % CI) of correct, A36 - A12, per
benchmark.

1. **Harm.** If the CI upper bound is below 0 on either benchmark, the 36 000 default for paged
   corpora is reverted to 12 000 (36 000 stays available through `budget_tokens`).
2. **Benefit.** The answer gain is confirmed iff the MDA-QA CI lower bound is above 0. Otherwise the
   default stays (the user's decision) and the README says the answer gain is not confirmed.
3. **No harm on single-paper questions** is claimed only if the QASPER CI lower bound is above
   -0.05.

## Cost guard

Planned: 800 reader and at most 800 judge calls, about 40 million input tokens of the user's Codex
subscription. A probe reads Codex's own usage figures every few minutes; the run is stopped (the
cache keeps finished calls) if the weekly usage reaches 45 % (17 % before the study) or the
five-hour usage 85 %, and resumes only on the user's word. The sample is not changed.

## Predictions (written before any run)

- P1: MDA-QA correct rises by 0.10 to 0.25 (P-2: needed papers packed 0.47 to 0.74).
- P2: QASPER correct changes by less than 0.03 in either direction (P-2: evidence recall +0.061).
- P3: `context_supports` true rises on MDA-QA by at least 0.15.
- P-cost: under 1 hour with 4 parallel calls.

## Not claimed

The ezcross agent's own model (`claude-sonnet-4-5`) or an agent that searches several times; long
PDFs with appendices (P-1's corpus has no reference answers); a judge from another model family
(reader and judge are the same model, the same for both arms); MDA-QA reference answers are
machine-written.

## Amendment 1 (before any QASPER answer exists; no answer scored or read)

The cost guard stopped the first run after 340 calls (171 reader, 169 judge, all MDA-QA; 8.7
million input tokens): five-hour usage 90 %, weekly 30 %. One five-hour window of the user's plan
holds about 10 million tokens, not the 40 million the plan needs, and the full plan would end near
78 % of the week. The user decided to cut the sample, against "The sample is not changed" above:

- **QASPER: the first 60 of the 300 shuffled questions** (same corpus of 416 papers). MDA-QA stays
  at 100; its cached calls are kept and the 60 missing ones are made.
- Rules 1 to 3 stay as written. With 60 questions the QASPER interval is about ±0.10 wide, so rule
  3 (lower bound above -0.05) will probably not be met; that outcome is reported as "not shown",
  not as harm. P2 keeps its wording.
- The run resumes when the five-hour window resets; the guard thresholds stay (85 %, 45 %).
- P-cost was wrong (quota, not time, is the limit).

## Interim outcome at 60 QASPER questions (`docs/sota-runs/paper_answers_interim_60.txt`)

| A36 - A12 | MDA-QA (100) | QASPER (60) |
|---|---|---|
| correct (primary) | +0.080 [+0.000, +0.160], 0.290 to 0.370, 13 better / 5 worse | +0.050 [-0.050, +0.150], 0.650 to 0.700, 6 / 3 |
| yes or partial | +0.160 [+0.090, +0.240] | +0.100 [+0.000, +0.200] |
| `context_supports` | +0.160 [+0.090, +0.240] | +0.117 [+0.050, +0.200] |
| gold papers cited | 0.422 to 0.656 | 0.833 to 0.950 |

- Rule 1 (harm): no CI upper bound below 0. **The 36 000 default stays.**
- Rule 2 (benefit): the MDA-QA lower bound is +0.000, not above 0. **Not confirmed.**
- Rule 3 (no harm on QASPER): the lower bound is -0.050, not above -0.05. **Not shown**, as
  amendment 1 expected at 60 questions.
- Predictions: P1 wrong (+0.080, predicted +0.10 to +0.25); P2 wrong as a point estimate (+0.050,
  predicted within 0.03; the interval includes it); P3 met (+0.160); P-cost wrong.
- MDA-QA papers packed per question: 1.09 at 12 000, 2.59 at 36 000.
- Counted after scoring: on 17 MDA-QA and 18 QASPER questions both budgets packed the same
  context, so the same answer and verdict serve both arms (a difference of exactly 0). The
  differences rest on the other 83 and 42 questions (correct: 13 better / 5 worse, 6 / 3).
- Cost: 569 Codex calls (285 reader, 284 judge; equal requests of the two arms are made once), 13.8
  million input tokens (8.1 million cached), two five-hour windows (90 % and 46 % used), weekly
  usage 17 % to 38 % including the usage probes.

## Amendment 2 (before any further answer exists): QASPER back to the planned 300

After the outcome above the user decided to run the sample as first frozen: the remaining 240 of
the 300 shuffled QASPER questions, same corpus, reader, judge and prompts; the 60 cached
questions are its first 60. MDA-QA is complete and is not run again.

- The outcome above becomes an **interim look at 60 questions**
  (`docs/sota-runs/paper_answers_interim_60.txt`); the QASPER verdicts of rules 1 and 3 are taken
  from the full 300. The extension was decided knowing the interim result (+0.050 [-0.050,
  +0.150]), so QASPER has had two looks at an unadjusted 95 % interval; the 300 are the sample
  fixed before any run, not one chosen for its result.
- The MDA-QA verdict (rule 2, not confirmed) is final and does not change.
- Cost guard: five-hour limit 85 % as before; the run waits for the reset and resumes by itself.
  The weekly limit is raised from 45 % to 65 % on the user's word (38 % before this run; about 20
  points expected).

## Final outcome (`docs/sota-runs/paper_answers_score.txt`)

The 320 records of the interim look are unchanged in the full run (compared record by record
with `ce852d0`), so the MDA-QA column is the one above.

| A36 - A12 | MDA-QA (100) | QASPER (300) |
|---|---|---|
| correct (primary) | +0.080 [+0.000, +0.160], 0.290 to 0.370, 13 better / 5 worse | +0.043 [+0.010, +0.077], 0.627 to 0.670, 20 / 7 |
| yes or partial | +0.160 [+0.090, +0.240] | +0.053 [+0.023, +0.087] |
| `context_supports` | +0.160 [+0.090, +0.240] | +0.070 [+0.037, +0.103] |
| gold papers cited | 0.422 to 0.656 | 0.907 to 0.970 |

- Rule 1 (harm): no CI upper bound below 0. **The 36 000 default stays.**
- Rule 2 (benefit): the MDA-QA lower bound is +0.000, not above 0. **Not confirmed.**
- Rule 3 (no harm on QASPER): the lower bound is +0.010, above -0.05. **Shown.** The interval is
  above 0 as well; the protocol set no rule for a QASPER gain, and the comparison had two looks.
- Predictions: P1 wrong (+0.080, predicted +0.10 to +0.25); P2 wrong (+0.043, predicted within
  0.03); P3 met (+0.160); P-cost wrong.
- Both budgets packed the same context on 17 MDA-QA and 110 QASPER questions (a difference of
  exactly 0); the differences rest on the other 83 and 190.
- Cost of the whole study: 1 335 Codex calls (673 reader, 662 judge), 30.7 million input tokens
  (18.9 million cached), four five-hour windows, weekly usage 17 % to 62 % including the probes.
