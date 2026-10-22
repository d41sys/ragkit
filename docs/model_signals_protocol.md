# Protocol: model-based abstention signals (fixed before any model score was computed)

This file is committed before the two models are run on any ragkit question. Nothing below is
changed after seeing a score. If a result is negative it is reported as negative; a different
model, aggregation or threshold afterwards would be a new protocol with its own entry.

## Question

Do signals that read the retrieved text against the question separate answerable (42) from
unanswerable (45) EU questions better than `query_support` (AUROC 0.587 [0.463, 0.702], all 45)?
Nine lexical / hashing-cosine signals and their logistic combination did not (`combined`
0.589).

## Models (2, chosen before scoring; revisions pinned in code)

| Signal | Model | Revision (HF commit) | Licence |
|---|---|---|---|
| `ce_top` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `233902d25c440f23af6f7d6e94d2946bac0bee0a` | Apache-2.0 |
| `qa_answerable` | `deepset/roberta-base-squad2` | `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4` | CC-BY-4.0 |

Revisions and licences read from the Hugging Face API on 2026-10-23 (`HfApi().model_info`).
An NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`) was considered and **not** used: a
question is not a natural NLI hypothesis, while SQuAD2 training is a direct answerability
objective. It is not run, so nothing is claimed about NLI.

## Signal definitions (fixed)

Both are computed over the same top-5 chunks as `query_support` (the pipeline's retrieved
chunks) and aggregated by the maximum, higher = more likely answerable.

- `ce_top`: max over the 5 chunks of the cross-encoder raw logit for `(question, chunk)`.
- `qa_answerable`: for each chunk, `best non-null span score - null score`, where span score is
  `start_logit[i] + end_logit[j]` over context tokens with `i <= j < i + 30`, and the null score
  is `start_logit[0] + end_logit[0]` (the `<s>` token); max over the 5 chunks. Positive means
  the model prefers a span to "no answer".

No variants (mean, top-1 only, softmax, other lengths) are tried afterwards.

## Evaluation (all existing code, no new knobs)

- Data: EU set, 42 answerable, 45 unanswerable (all), plus the original-12 and expanded-33
  cohorts as descriptive cuts.
- AUROC with a stratified percentile bootstrap (2000 resamples, seed 0); paired difference vs
  `query_support` on identical resamples.
- Combined models, nested 5-fold CV by question, CV seed 0 (seeds 1 to 4 as a robustness
  check, not for selection): `combined` (the 9 lexical signals, unchanged),
  `combined_model` (2 model signals), `combined_all` (9 + 2).
- Risk-coverage at 100/90/80/70%; held-out accept/abstain at the CV-chosen threshold
  (balanced accuracy, Wilson CIs).

## Decision rule

Primary comparisons, all on the 45-question set and CV seed 0 (3 of them: `ce_top`,
`qa_answerable`, `combined_all`; each against `query_support`):

- **Improves on `query_support`**: the paired AUROC difference 95% CI lower bound is above 0.
  With 3 primary comparisons and no multiplicity correction, one such pass would be weak
  evidence and the README must say so.
- **Separates the classes**: own AUROC 95% CI lower bound above 0.5.
- **Usable abstention rule**: improves, separates, and held-out balanced accuracy at the
  CV-chosen threshold is at least 0.70.

A signal that fails the first test is reported as "not shown better than `query_support`", even
if its point estimate is higher. The original-12 numbers are not used to choose among signals
(they were already seen to be optimistic). `combined_model` and the cohort / seed cuts are
secondary and descriptive.

## Cost and caching

CPU by default (`--model-device cpu`); scores are cached to the gitignored
`.ragkit/model_scores.json`, keyed by model id, revision and a hash of (question, chunk).
Runtime of the cold-cache run is recorded in the README.
