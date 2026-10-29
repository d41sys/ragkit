# Generation and sufficient-context abstention protocol (K-R2; frozen before scoring)

Goal: replace the extractive generator by an LLM answer with citations, add a sufficient-context
check (Joren et al. 2024, `arXiv:2411.06037`, style) in front of it, combine that check with the
existing SQuAD2 reader signal, and compare with a full-context baseline (the whole EU corpus in
the prompt). Everything is measured on the 42 answerable (`qa_eu.jsonl`) and 45 unanswerable
(`unanswerable_eu.jsonl`) EU questions, the set behind "answers 37/42, abstains on 32/45,
13/45 unanswerable still answered" in the README. This file, the prompts in
`src/ragkit/generation.py` and the decision rule are committed before any scored call. A format
pilot uses only the synthetic corpus (`corpus/`, `qa.jsonl`) and is not scored; no EU question is
sent to the API before the scored run, and no prompt is edited after it.

## Model, cache, budget

- Model `claude-sonnet-5-5` through the Messages API over `urllib` (no SDK; the `llm` extra is
  not needed). `temperature` 0 when the model accepts it, else the API default (recorded in each
  cached line). `max_tokens` 1200 (generation, full context) / 800 (judges).
- Every request is cached in `data/llm_cache/generation.jsonl` under the SHA-256 of its payload
  (system, messages, model, max_tokens); a cached line holds the reply, the usage block and a
  request summary (the corpus block is stored as its hash and length). Scoring replays the cache
  offline; tests need neither key nor network. Live calls are capped at 1500 and about 15 USD.
- Cost uses assumed list prices (not read from the API): 3 USD per million input tokens, 15 per
  million output tokens, cache write 1.25x and cache read 0.1x of the input price.
- Prompt caching: the full-context baseline marks the corpus block `cache_control: ephemeral`
  (supported by the Messages API; `hardset_review` already uses it). The first call runs alone to
  write the cache, the rest read it.

## Systems

Retrieval for the RAG systems is the shipped `RAGPipeline` top-5 chunks (hybrid with the hashing
embedder + lexical rerank), the chunks the reader signal was computed on, so the reader's scores
come from the committed cache (`data/llm_cache/reader_scores.jsonl`, copied from the model-score
cache) with no model load. Each chunk is shown as `[article id] text`.

Calls per question (all prompts are in `generation.py`):

1. `GEN` (RAG): answer only from the context, cite article ids, always give a best-effort
   answer, plus a boolean `context_supports`. JSON `{answer, citations, context_supports}`.
2. `SUFF`: the autorater of the sufficient-context paper adapted to JSON: given only the question
   and the same chunks, is there enough information to give a definitive answer? JSON
   `{reasoning, sufficient, confidence}` with `confidence` an integer 0 to 100 for "sufficient".
   The judge never sees a reference answer or the gold article.
3. `FC`: the whole corpus (337 kB, one `=== id ===` block per article, about 85k tokens) as a
   cached system block, the same answer schema as `GEN`.
4. `CORRECT` (LLM-judged, answerable questions only): reference answer vs candidate answer,
   verdict `yes` / `partial` / `no` for "states what the reference states".
5. `CITE` (LLM-judged): for each cited article of a `GEN` or `FC` answer, the question, the
   answer and the full article text; verdict `yes` / `partial` / `no` for "the article supports
   what the answer says". A cited id that is not among the retrieved articles (`GEN`) or not in
   the corpus (`FC`) counts as unsupported without a call.

Decision systems (answer iff the rule holds; the answer text is always `GEN`'s):

| id | answers when |
|---|---|
| `reader` | `qa_answerable` >= the threshold chosen by the existing 5-fold stratified CV (seed 0) (baseline) |
| `self` | `GEN.context_supports` is true |
| `suff` | `SUFF.sufficient` is true |
| `strict` | `suff` and `reader` both say answer (abstain if either abstains) |
| `lenient` | `suff` or `reader` says answer (abstain only if both abstain) |
| `stack` | logistic regression on `qa_answerable` and `SUFF.confidence`, nested 5-fold CV by question as in `ragkit abstention` (numpy, ridge and threshold chosen inside the training folds) |
| `fullctx` | `FC.context_supports` is true (no retrieval, no reader) |

## Metrics (all with 95 % CIs: Wilson for counts, stratified bootstrap over questions, 2000
resamples, seed 0, for differences)

- Abstention AUROC (answerable vs unanswerable) for `qa_answerable`, `SUFF.confidence`, the
  stacked score and the binary flags.
- Unanswerable answered, `n/45` (the existing figure to beat is 13/45); answerable wrongly
  abstained, `n/42` (the existing figure is 5/42).
- Balanced accuracy of the decision = (answerable answered rate + unanswerable abstained rate) / 2.
- Answer accuracy on the 42 answerable questions: the system answers and `CORRECT` says `yes`
  (an abstention counts as not correct); `partial` is reported separately; accuracy when
  answered is also given. LLM-judged.
- Citation support rate: share of cited articles (over answers the system gave) with `CITE`
  `yes`; `partial` and invalid ids reported separately. LLM-judged; the judge is the same model
  family as the generator, so self-preference is possible and no human checked a sample.
- Cost: calls, tokens by kind (input, cache write, cache read, output), cost at assumed prices.

## Predictions (before any scored call)

- A1 `SUFF.confidence` AUROC is above `qa_answerable`'s 0.858 and at least 0.90.
- A2 `suff` answers at most 6 of the 45 unanswerable questions (reader: 13) and wrongly abstains
  on at most 10 of the 42 answerable ones (reader: 5); most of its wrong abstentions are
  answerable questions whose gold article is absent from the top 5 (retrieval misses, counted
  from the existing table).
- A3 `self` answers more unanswerable questions than `suff` (models answer instead of
  abstaining, as in the sufficient-context paper).
- A4 `GEN` citation support (`yes`) is at least 85 %, `FC` at least 90 %.
- A5 `fullctx` answers at least 5 percentage points more of the answerable questions correctly
  than the RAG `self` system (no retrieval misses) and answers no more unanswerable questions.
- A6 `strict` has the fewest unanswerable answered and the most wrong abstentions of the
  gates.

## Decision rule

The sufficiency check is adopted as the abstention gate of `ragkit answer --llm` iff the paired
balanced-accuracy difference `suff` minus `reader` has a 95 % bootstrap CI lower bound above 0.
`stack`, `strict` and `lenient` are secondary rows with the same paired test and
are not used to pick a winner. If the rule fails, the README reports "no detectable effect" (CI
includes 0) or a loss, and `--llm` keeps the sufficiency check as an option, not the default
gate. No threshold of the LLM path is tuned on these 87 questions: `suff` uses the model's own
verdict, `self` its own flag.

## Secondary (only if the main run lands inside the budget)

The same `GEN`, `SUFF`, `CITE`, `CORRECT` calls with the `hybrid+ce` retriever of
`docs/neural_protocol.md` (top 5 chunks) in place of the shipped pipeline. No reader signal exists
for those chunks (the reader would have to be re-run), so only `self`, `suff` and the AUROC of
`SUFF.confidence` are reported.

## Not claimed

The 87 labels are agent-reviewed (README caveat); 87 questions resolve differences of roughly
10 points only. LLM judges are not human labels. Published sufficient-context and UAEval4RAG
numbers come from other corpora and models and are not compared with ours
(`docs/sota-review.md` section 7.3).
