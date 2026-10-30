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

## Outcome (scored run; raw output `docs/sota-runs/generation_default.txt`, `generation_hybrid_ce.txt`)

Calls are cached in `data/llm_cache/generation.jsonl` (1016 calls for both runs; the cost line of
the raw files counts that whole file); `ragkit generation` replays it offline and prints exactly
the committed tables. Prices are assumed list prices (3 / 15 USD per M input / output, cache
write 1.25x, cache read 0.1x), not read from the API: 9.72 USD, 1.27 M input, 0.10 M cache-write,
8.54 M cache-read and 0.20 M output tokens. Four more pilot calls (synthetic corpus, about
0.03 USD) are in the gitignored `.ragkit/llm_cache.jsonl`.

Deviations, all outside the frozen prompts: the API rejects `temperature` for `claude-sonnet-5-5`
(HTTP 400 in the pilot), so no temperature is sent; two `SUFF` replies of the secondary run were
cut at 800 output tokens and failed to parse, so `_json` got a third attempt with doubled
`max_tokens` (the first two attempts and their cache keys are unchanged).

Default retriever (shipped pipeline, top 5 chunks; reader replayed from
`data/llm_cache/reader_scores.json`, 37/42 answered and 32/45 abstained as in the README):

| System | unanswerable answered (of 45) | answerable wrongly abstained (of 42) | balanced acc. [95% CI] | paired d vs reader | correct (LLM-judged, of 42) |
|---|---|---|---|---|---|
| `reader` | 13 | 5 | 0.796 [0.715, 0.875] | baseline | 12 |
| `self` | 0 | 25 | 0.702 [0.631, 0.774] | -0.094 [-0.195, +0.013] | 11 |
| `suff` | 0 | 24 | 0.714 [0.643, 0.798] | -0.082 [-0.186, +0.031] | 11 |
| `strict` | 0 | 25 | 0.702 [0.631, 0.774] | -0.094 [-0.191, +0.011] | 11 |
| `lenient` | 13 | 4 | 0.808 [0.726, 0.887] | +0.012 [+0.000, +0.036] | 12 |
| `stack` (nested CV) | 6 | 6 | 0.862 [0.790, 0.931] | +0.066 [-0.025, +0.150] | 12 |
| `fullctx` | 0 | 0 | 1.000 [1.000, 1.000] | +0.204 [+0.125, +0.285] | 41 |

AUROC: `SUFF.confidence` 0.967 [0.933, 0.990], `qa_answerable` 0.858 [0.772, 0.928], `stack`
0.955 [0.911, 0.986] (no paired AUROC test was pre-registered, so the first two are compared by
interval only). Citation support (LLM-judged, per cited article, over all GEN answers): `yes`
0.587; FC 0.592. Per answering system, for example `self` 0.800 and `suff` 0.762: see the raw
file.

Predictions: A1 holds. A2 does not hold (0/45 answered, but 24/42 wrongly abstained against the
limit of 10; 6 of the 24 are retrieval misses). A3 does not hold (`self` and `suff` both answer
0 of 45). A4 does not hold (GEN 0.587 against 0.85, FC 0.592 against 0.90; most non-`yes`
verdicts are `partial`: answers cite two articles and each supports only part). A5 holds
(`fullctx` 0.976 correct against 0.262 for RAG `self`, 0 unanswerable answered). A6 holds.

Decision rule: `suff` minus `reader` balanced accuracy -0.082 [-0.186, +0.031], lower bound not
above 0: **the sufficiency gate is not adopted**; no detectable effect (CI includes 0, point
estimate a loss). `ragkit answer --llm` therefore defaults to `--gate none`; `--gate suff` and
`--gate self` are options. The binary gates are conservative on the 500-character chunks of the
shipped pipeline (they answer none of the unanswerable questions and refuse 24 of 42 answerable
ones); the continuous confidence carries the signal (AUROC 0.967), and a gate fitted on it with
the reader (`stack`) halves the reader's unanswerable-answered count to 6 at a similar
wrongly-abstained count, but its CI also includes 0.

Secondary run (`hybrid+ce` chunks, no reader): `SUFF.confidence` AUROC 0.987 [0.963, 0.999];
`suff` answers 0/45 and wrongly abstains on 17/42 (balanced accuracy 0.798 [0.726, 0.869]);
`self` 0/45 and 19/42; RAG answers 14/42 correct (default retriever: 11 to 12); 6 of 42 gold
articles are still missing from the top 5.

Full context (the whole 94-article corpus, about 85 k tokens, prompt-cached): 41 of 42
answerable questions correct, 0 of 45 unanswerable answered, 0 wrong abstentions, 87 calls for
3.29 USD. This holds for a corpus that fits in a prompt; it says nothing about corpora that do
not. Judge: the same model family as the answerer (`CORRECT`, `CITE` run by `claude-sonnet-5-5`),
no human check, so a shared bias cannot be excluded.

## Phase L: parent-document (small-to-big) context (frozen before any scored call)

Why: in the scored K-R2 run, 22 of the 25 `no` and 20 of the 32 `partial` `CORRECT` verdicts (all
`CORRECT` records of the cache, three runs) are on answers that say the passages were cut off, not
provided or do not state the fact, while only 6 of 42 gold articles are missing from the top-5
chunks. The 500-character chunks hand `GEN` fragments of the right article; full context (41/42)
does not. This phase tests that cause by changing only what `GEN` is shown.

Method (`--context article`, `src/ragkit/parent.py`): retrieve the top-5 chunks exactly as in K-R2
(the same pipeline, the same ranking), then pass the FULL text of the distinct parent articles of
those chunks to `GEN`, in rank order of their best chunk, each as `[article id]` followed by the
whole article. The `GEN` prompt and schema are unchanged (byte for byte); only the context
differs.

- **k = 3** articles (the three best-ranked distinct articles; fewer if the top 5 chunks come from
  fewer articles).
- **Token budget 12 000** estimated tokens (characters / 4; the 94-article corpus is 337 kB and
  the K-R2 protocol counts it as about 85 k tokens, the same ratio). Articles are added whole in
  rank order; the first article is always kept (cut at the budget only if it alone exceeds it);
  a later article that would exceed the budget is dropped, as are all after it. The longest
  article is 12.1 kB (about 3 k tokens), so on this corpus 3 articles are at most about 9 k tokens
  and the budget never binds; it is there so the same code is safe on other corpora.
- k and the budget were fixed here, before any article-context call; they are not tuned on the 87
  questions.
- The decision gates are not re-run: `reader` is the K-R2 `qa_answerable` rule on the retrieved
  chunks and `stack` the K-R2 nested-CV logistic gate on `qa_answerable` and the chunk-context
  `SUFF.confidence` (cached K-R2 calls, no new `SUFF` call). Each gate answers the same questions
  as in K-R2; Phase L changes only the text of the answer, its correctness and its citations.
  For `hybrid+ce` no reader scores existed, so the reader (`roberta-base-squad2`, the pinned
  revision of K-R2) is run once locally on the `hybrid+ce` top-5 chunks and its scores are
  committed as `data/llm_cache/reader_scores_hybrid_ce.json`.

Calls (new, on top of the K-R2 cache; cap 600 new calls, the cost cap rises from 15 to 25 USD
cumulative): `GEN` for all 87 questions per retriever, `CORRECT` for the 42 answerable, `CITE`
for every article an answer cites (only an id in the context counts; others are `invalid` without a
call) on all 42 answerable questions and on the unanswerable questions that `reader` or `stack`
answers (the only unanswerable answers a reported row counts). Retrievers: the shipped `default`
pipeline ("BM25 default") and `hybrid+ce`. No other call is made; chunk-context `GEN`, `FC` and
their judge calls are replayed from the cache.

Reported (per retriever, LLM-judged, no human check):

1. Always-answer correctness on the 42 answerable questions (`CORRECT` = `yes`, strict; and
   `yes` + `partial`, lenient) for chunk-context RAG, article-context RAG and full context (FC),
   each with a Wilson 95 % CI, and paired differences (paired bootstrap over questions, 2000
   resamples, seed 0) article minus chunk and article minus FC, at both levels.
2. The `reader` and `stack` gates with article-context answers: unanswerable answered,
   wrongly abstained, balanced accuracy, correct (strict and lenient) with Wilson CIs, and the
   citation table (yes / partial / no / invalid per cited article) next to chunk context and FC.
3. Context-gap answers: among `CORRECT` verdicts other than `yes`, the share whose answer matches
   a fixed phrase pattern (`cut off`, `truncat`, `incomplete`, `not provided`, `not included`,
   `cannot confirm|determine`, or `passage(s)|context|text|excerpt|fragment` followed within 40
   characters by `do|does|did not`), for chunk and article context. The pattern is fixed here
   and is a text match, not a judgement.
4. Retrieval misses: answerable questions whose gold article is not in the article context
   (k = 3 can miss articles that five chunks reached).
5. FC citation audit (cache only, no calls): per cited article of the 87 FC answers, the verdict
   against whether the id is a gold article of the question; plus a manual read of 10 randomly
   chosen (seed 0) non-`yes` FC `CITE` records, classified as wrong article id cited / correct
   article judged partial because the article is the citation granularity / other.

Predictions (before any scored call):

- L1 Article-context RAG, always answering, is correct (strict) on at least 30 of 42 (default)
  and at least 32 of 42 (`hybrid+ce`), against 12 and 14 for chunk context.
- L2 The paired difference article minus chunk has a 95 % CI lower bound above 0 for both
  retrievers (strict).
- L3 Article-context RAG stays below full context (point difference article minus FC negative,
  and at least -0.25): the remaining gap is retrieval misses within k = 3.
- L4 The context-gap share of non-`yes` verdicts falls from about 0.8 (chunk, on the K-R2 records)
  to at most 0.3 (article).
- L5 The citation `yes` rate of article-context `GEN` is higher than 0.587 (chunk) and is at
  least 0.70, yet A4's 0.85 is still not reached (answers cite several articles and each supports
  only part).
- L6 Most non-`yes` FC citations are `partial` on a gold article (the answer combines claims from
  several articles, so one whole article supports a part); at most 2 of the 10 read records are
  a wrong article id.

Decision rule: article context becomes the default of `ragkit answer --llm` iff the strict paired
difference article minus chunk has a CI lower bound above 0 for both retrievers; otherwise
`--context chunk` stays the default. If the rule passes, that is the answer to "why did RAG get
12/42": the chunks, not the retriever or the generator.
