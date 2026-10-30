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
0 of 45). **A4 failed** (GEN 0.587 against the 0.85 target, FC 0.592 against 0.90; most non-`yes`
verdicts are `partial`: answers cite several articles and each supports only part; Phase L below
reads the FC citations). A5 holds (`fullctx` 0.976 correct against 0.262 for RAG `self`, 0
unanswerable answered). Added after the run, from the cached records and confirmed by Phase L:
the 12/42 of RAG is mostly the 500-character chunks, not retrieval or generation. Across all
`CORRECT` records of the cache (124 parsed; the Phase L phrase pattern), 22 of the 25 `no` and 25
of the 34 `partial` verdicts are on answers that say the passages were cut off, not provided or
do not state the fact (an earlier scan with another pattern found 22 and 20); only 6 of 42
gold articles are missing from the top 5 chunks; giving `GEN` the full parent articles lifts the
same retrievers from 12 to 28 and from 14 to 30 of 42. A6 holds.

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

### Phase L outcome (scored; raw output `docs/sota-runs/generation_article_default.txt` and `generation_article_hybrid_ce.txt`, replayed offline by `ragkit generation --context article [--retriever hybrid+ce]`)

Protocol commit `b8b09c9`, code commit `840db6c`, both before any article-context call. 416 new
cached calls (209 default, 207 `hybrid+ce`; cap 600), 4.53 USD more at the assumed prices (cache
total 1432 calls, 14.25 USD; the code caps rose to 1616 records and 25 USD for this phase). The
reader for the `hybrid+ce` chunks was run locally once (`roberta-base-squad2`, the K-R2 revision;
on 6 default-retriever questions it reproduced the committed scores exactly) and committed as
`data/llm_cache/reader_scores_hybrid_ce.json`. Everything is LLM-judged by the generator's model
family, 42 answerable questions, no human check.

Always-answer correctness on the 42 answerable questions (strict = `yes`; lenient = `yes` +
`partial`; Wilson 95 % CIs; paired differences are bootstrap CIs over questions):

| Retriever | Context | strict | lenient | paired d strict vs chunk | paired d strict vs FC |
|---|---|---|---|---|---|
| default | chunk RAG | 12/42 [0.17, 0.44] | 28/42 [0.52, 0.79] | baseline | -0.690 [-0.833, -0.548] |
| default | article RAG | 28/42 [0.52, 0.79] | 35/42 [0.69, 0.92] | +0.381 [+0.214, +0.548] | -0.310 [-0.452, -0.167] |
| `hybrid+ce` | chunk RAG | 14/42 [0.21, 0.48] | 31/42 [0.59, 0.85] | baseline | -0.643 [-0.786, -0.500] |
| `hybrid+ce` | article RAG | 30/42 [0.56, 0.83] | 33/42 [0.64, 0.88] | +0.381 [+0.238, +0.524] | -0.262 [-0.405, -0.119] |
| both | full context | 41/42 [0.88, 1.00] | 42/42 [0.92, 1.00] | | |

Lenient paired differences, article minus chunk: default +0.167 [+0.024, +0.333], `hybrid+ce`
+0.048 [-0.048, +0.143] (includes 0); article minus FC: -0.167 [-0.286, -0.071] and -0.214
[-0.333, -0.095].

Gates (the `reader` and `stack` decisions are K-R2's and do not change; only the answer text does).
Correct answers, strict, of 42, and the paired difference to the chunk-context answers of the
same gate:

| Retriever | Gate | unanswerable answered | wrongly abstained | chunk | article | paired d |
|---|---|---|---|---|---|---|
| default | `reader` | 13/45 | 5/42 | 12 | 27 | +0.357 [+0.190, +0.524] |
| default | `stack` | 6/45 | 6/42 | 12 | 26 | +0.333 [+0.167, +0.500] |
| `hybrid+ce` | `reader` | 12/45 | 12/42 | 13 | 24 | +0.262 [+0.143, +0.405] |
| `hybrid+ce` | `stack` | 2/45 | 4/42 | 14 | 30 | +0.381 [+0.238, +0.524] |

`hybrid+ce` `stack` (balanced accuracy 0.930 [0.871, 0.977]) is a new row: the reader scores for
those chunks did not exist in K-R2. Its nested-CV gate is not tuned on anything beyond K-R2's
protocol, but it is the best row of the table and was not chosen in advance.

Citation support, per cited article (LLM-judged), always answering on the 42 answerable
questions: default chunk 0.424 (28/66), article 0.661 (41/62), FC 0.506 (40/79); `hybrid+ce`
chunk 0.507 (34/67), article 0.662 (43/65). Gated: `reader` article 0.690 and `stack` 0.656
(default), 0.667 and 0.655 (`hybrid+ce`). The `CITE` calls cover the 42 answerable questions and
the unanswerable questions that `reader` or `stack` answers (55 questions default, 54
`hybrid+ce`); on those the chunk-context yes rate is 0.488 / 0.576 and article context 0.688 /
0.678. Every row is below A4's 0.85 (and FC's 0.506 / 0.592 below 0.90).

Predictions (the report prints each; `holds` means the pre-registered inequality held):

| | default | `hybrid+ce` |
|---|---|---|
| L1 article strict >= 30 / >= 32 | 28: does not hold | 30: does not hold |
| L2 paired d article - chunk lower bound > 0 | +0.214: holds | +0.238: holds |
| L3 article - FC in [-0.25, 0) | -0.310: does not hold | -0.262: does not hold |
| L4 context-gap share of non-`yes` <= 0.30 | 0.86 (chunk 0.87): does not hold | 1.00 (chunk 0.75): does not hold |
| L5 citation yes in [0.70, 0.85) and above chunk | 0.688 vs 0.488: does not hold | 0.678 vs 0.576: does not hold |
| L6 (manual read, below) | first part does not hold, second holds | same records |

L1, L3 and L5 miss by a small margin on the side of "worse than predicted": article context
recovers 18 (default) and 15 (`hybrid+ce`) of the 29 and 27 questions that full context answers
`yes` and chunk context does not, not all (2 and 0 questions go the other way). L4 was badly
specified: after the change the non-`yes` verdicts fall from 30 to 14 (default) and from 28 to 12
(`hybrid+ce`), but most of the remaining ones (9 of 14 and 8 of 12) are questions whose gold
article is not in the context at all (next paragraph), and then the model rightly says the
passages do not state the fact, which the phrase pattern counts as a context gap.

Where the remaining failures are (post-hoc, not pre-registered; `CORRECT` verdicts of the
always-answer rows by whether a gold article is in the context `GEN` sees):

| Retriever | Context | gold in context | questions | yes | partial | no |
|---|---|---|---|---|---|---|
| default | chunk | yes | 36 | 11 | 16 | 9 |
| default | chunk | no | 6 | 1 | 0 | 5 |
| default | article | yes | 33 | 28 | 4 | 1 |
| default | article | no | 9 | 0 | 3 | 6 |
| `hybrid+ce` | chunk | yes | 36 | 14 | 17 | 5 |
| `hybrid+ce` | chunk | no | 6 | 0 | 0 | 6 |
| `hybrid+ce` | article | yes | 34 | 30 | 3 | 1 |
| `hybrid+ce` | article | no | 8 | 0 | 0 | 8 |

So with a gold article in the context, article context is `yes` for 28 of 33 (default) and 30 of
34 (`hybrid+ce`) against 11 of 36 and 14 of 36 for chunks (different question sets: the article
context loses 3 and 2 gold articles that five chunks reached, because k = 3 distinct articles
is fewer than the articles behind five chunks). The remaining gap to FC is retrieval: 9 and 8
questions have no gold article in the three articles (6 had none in five chunks). k was fixed
in advance and was not re-tuned; a larger k or a chunk-to-article fallback would address these
but is untested.

**Judge check: strict against lenient** (the `partial` verdict is frequent, so the strict number
alone understates chunk RAG): default chunk 12 `yes` + 16 `partial` = 28 lenient, article 28 + 7
= 35, FC 41 + 1 = 42; `hybrid+ce` chunk 14 + 17 = 31, article 30 + 3 = 33. At the lenient level
chunk RAG is 0.67 (default) and 0.74 (`hybrid+ce`), not 0.29 and 0.33; the article-minus-chunk
gain is still above 0 for the default retriever (+0.167 [+0.024, +0.333]) and includes 0 for
`hybrid+ce` (+0.048 [-0.048, +0.143]). The strict gap is mostly the difference between a
complete and an incomplete answer, which is what the chunk truncation produces. No human
re-graded a `partial`.

**Why FC is correct 41/42 and its citations are `yes` only 0.51** (cache only, no new call; the
0.51 is FC's rate over the 79 articles cited for the 42 answerable questions, the 0.592 of the
K-R2 outcome is over all 87 answers): the judge is given one cited article at a time and asked
whether it supports the answer, and FC answers cite every article they used.

- All 42 FC answers cite at least one gold article, so the wrong-id explanation is excluded as the
  main cause: of the 39 non-`yes` citations, 13 are `partial` on a gold article, 23 `partial` and
  3 `no` on an article that is not in the gold set.
- The yes rate depends on how many articles the answer cites: 17 answers cite one article (17 of
  17 citations `yes`); 25 answers cite two or more (62 citations, 23 `yes`, 0.37).
- Manual read of the 10 pre-registered records (random, seed 0, over all 87 FC answers, the judge's
  reason beside each in the raw file): wrong article id 0; correct article judged `partial`
  because the answer combines claims from several articles, so one whole article supports only
  part (citation granularity) 3 (records 3, 4, 10); something else 7. The seven are citations
  attached to FC answers that say the corpus does not state the fact (the question is one of the
  45 unanswerable ones, FC abstains in substance), citing the nearest article; the judge calls it
  `partial` because the article supports only the side remark. They are not in the 0.51
  population (answerable questions only), so a second read was added post-hoc: 10 random
  non-`yes` citations on the answerable questions (same seed): wrong article id 1 (a `no`: article
  39 of 2017/2195 is cited for an answer whose claims come from Articles 40 to 42); correct
  article, `partial` because the answer spans several articles, 9 (3 on a gold article, 6 on a
  non-gold article that carries one real sub-claim of the answer); something else 0.
- L6 as pre-registered had two parts. "At most 2 of 10 are a wrong id" holds (0 and 1). "Most
  non-`yes` FC citations are `partial` on a gold article" does not hold: 13 of 39 on the answerable
  questions are on a gold article, 23 on a non-gold one that the answer really draws on. The cause
  is the granularity of the citation check (an article against a multi-article answer), not
  FC citing wrong articles; a per-claim citation check would score FC higher. This was not run.

**Decision rule**: the strict paired difference article minus chunk has a lower bound above 0 for
both retrievers (+0.214 and +0.238), so **article context is adopted as the default of `ragkit
answer`** (`--context chunk` keeps the old behaviour). `ragkit generation` keeps `--context chunk`
as its default because it is the K-R2 replay. Answer to "why 12/42": mostly the chunks. With the
whole parent article the same retriever and the same generator answer 28/42 (default) and 30/42
(`hybrid+ce`) completely, and most of what is still not `yes` (9 of 14, 8 of 12) are questions whose
gold article is not in the three articles. Full context stays ahead (41/42), and these results say nothing about corpora
too large for a prompt.

Not claimed: one model family does the answers and the judges; 42 questions resolve differences of
about 15 points; the three post-hoc tables (gold in context, the answerable-only FC sample, the
`hybrid+ce` `stack` row) were not pre-registered.

## Phase L addendum: k = 5 articles, 20 000-token budget (POST-HOC; frozen before any k = 5 call)

**Status: post-hoc.** k = 3 was frozen in advance (Phase L above). k = 5 is not: it was chosen
after the Phase L result showed that 9 (default) and 8 (`hybrid+ce`) of the 42 gold articles are not
among the three articles, i.e. it is motivated by misses on the very 42 answerable questions it is
scored on. There is no held-out set for this question (the other EU question files have no
unanswerable counterpart and no reader scores), so the result cannot be a clean test. It is
reported as a **post-hoc sensitivity analysis**: it can say whether k = 5 would have helped on
these questions, not that it generalises, and a CI that excludes 0 here would still be selected on
the test set. Only k = 5 (with the budget below) is run: not k = 4, not k = 6, so no further
choice is made on these questions. 5 is the number of chunks retrieved, so it is the largest k that
can still add an article (a context of every distinct article behind the top-5 chunks).

Setting: `--context article --articles 5 --budget 20000`. Everything else is Phase L (same
questions, retrieval, `GEN`/`CORRECT`/`CITE` prompts, `reader` and `stack` gate decisions, judge,
bootstrap). The budget is 20 000 estimated tokens (chars / 4); it does not bind here (largest
context 10 403 estimated tokens). For every question the k = 3 articles are the first three
articles of the k = 5 context (same rank order), so k = 5 only appends articles.

**Retrieval-only coverage, computed before any LLM call** (`ragkit generation --context article
--coverage-only [--retriever hybrid+ce]`, raw output `docs/sota-runs/article_coverage_default.txt`
and `article_coverage_hybrid_ce.txt`). Answerable questions whose context holds a gold article:

| Retriever | top-5 chunks | k = 3 | k = 4 (not run) | k = 5 | contexts that change k = 3 to 5 (answerable / all 87) | mean context tokens k = 3 to 5 |
|---|---|---|---|---|---|---|
| default | 36/42 | 33/42 | 35/42 | 36/42 | 26 / 58 | 3 790 to 4 966 |
| `hybrid+ce` | 36/42 | 34/42 | 34/42 | 36/42 | 14 / 38 | 3 232 to 3 858 |

So k = 5 can add a gold article for at most 3 (default) and 2 (`hybrid+ce`) questions: the ceiling
is the top-5 chunks, 36/42, because articles are only taken from those chunks. Even if every one of
them were answered `yes`, the strict gain is at most +3 and +2 questions; 6 questions have no gold
article at any k and a retrieval-side change (not a larger k) would be needed for them. The paired
bootstrap CI of a gain of 3 of 42 with no losses has a lower bound of 0 (4.5 % of resamples
contain none of the three questions, above the 2.5 % tail). Checked with the report's own
bootstrap on 0/1 differences (gains / losses: lower bound): 3 / 0: 0.000; 4 / 0: +0.024; 4 / 1:
-0.024; 5 / 1: 0.000; 6 / 2: -0.024. So **the adoption rule below cannot pass with a net gain of 3
or fewer questions**; it needs 4 gains and no loss, or more (default). For `hybrid+ce`, where a
gold article can be added for only 2 questions, it needs gains from questions whose gold article
was already in the context. This was worked out before scoring and is stated so the rule is not
read as a real test of a likely effect.

Calls (new, on top of the 1 432-record, 14.25 USD cache; hard budget 4 USD at assumed list prices,
enforced in code as a cumulative cap of 18.10 USD): `GEN` only for questions whose context differs
from k = 3 (the other `GEN`, and the `CORRECT` and `CITE` of an unchanged answer, are cache hits);
`CORRECT` for the changed answerable questions; `CITE` as in Phase L (every answerable question
and every unanswerable question that `reader` or `stack` answers, only for changed answers).
Estimated from the k = 3 call prices: `GEN` 58 + 38 calls about 2.3 USD, `CORRECT` about 0.1 USD,
`CITE` about 0.5 USD. Two stages: `GEN` and `CORRECT` first (`--no-cite`); `CITE` only if the spend
so far plus its estimate stays under 4 USD. Skipped `CITE` is reported as skipped, not imputed. The
unanswerable questions' `GEN` answers matter only through `CITE`; `unanswerable answered` is
decided by the gates, not by the context, so it equals the k = 3 figure by construction
(default 13/45 `reader` and 6/45 `stack`; `hybrid+ce` 12/45 and 2/45).

Reported (per retriever, both gates, LLM-judged, no human check):

1. Strict and lenient correct of 42 (Wilson CI), `reader` and `stack` gates and always answering,
   for k = 3, k = 5 and full context.
2. Unanswerable answered under both gates (k = 3 vs k = 5; equal by construction, printed as a
   check).
3. Cost per question: marginal spend of this run in total and per question, and the recorded
   `GEN` call cost per question at k = 3 and k = 5 (list prices, from the cached usage).
4. Paired differences (paired bootstrap over questions, 2000 resamples, seed 0) k = 5 minus k = 3
   and k = 5 minus full context, strict and lenient, with the counts of questions gained and lost.
5. Gold coverage and, among the strict-`yes` k = 3 answers, how many are lost at k = 5.

**Adoption rule (fixed here):** k = 5 (budget 20 000) becomes the default of `ragkit answer` only if
for **both** retrievers (a) the always-answer strict paired difference k = 5 minus k = 3 has a 95 %
CI lower bound above 0, and (b) `unanswerable answered` under `reader` and `stack` does not rise
above its k = 3 value. Otherwise `ARTICLE_K` stays 3 and k = 5 is a documented option
(`--articles 5 --budget 20000`). Gate-level differences are reported but do not enter the rule.

Predictions (before any k = 5 call):

- M1 The net strict gain k = 5 minus k = 3 is at most +4 questions for both retrievers (the
  gold-coverage ceiling is +3 and +2).
- M2 The lower bound of the strict paired difference is not above 0 for at least one retriever,
  so k = 5 is not adopted.
- M3 k = 5 stays below full context: the strict difference k = 5 minus FC is negative with a CI
  upper bound below 0 for both retrievers.
- M4 At least one question that k = 3 answers `yes` is not `yes` at k = 5 for at least one
  retriever (more articles, more distractors).
- M5 The recorded `GEN` cost per question rises by 15 to 30 % (default) and 8 to 20 %
  (`hybrid+ce`), following the token counts above.
