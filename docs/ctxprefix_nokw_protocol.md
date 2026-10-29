# Context prefix on a keyword-free EU set (rule frozen before any question was written)

Third and last test of the context prefix (`<act title> — Article N: <heading>` on every chunk;
`docs/ctxprefix_protocol.md`, `docs/ctxprefix_hard_protocol.md`). Both earlier sets were saturated
for BM25 (Recall@5 0.948 and 0.897): the discriminating words of the right article were in the
question, so plain BM25 already found it. This set is written to remove exactly those words.

This file is committed in parts. Part 1 (this commit) fixes the overlap rule and its test, before
any question exists. Part 2 (the frozen set, the Sonnet review, the guards and predictions) is
committed before the first retriever is run on the set.

## The mechanical rule (fixed now, before writing)

Implemented in `src/ragkit/nokw.py`, enforced by `tests/test_ragkit_nokw.py::test_nokw_set` on the
committed file, and runnable on any file with `python -m ragkit.nokw FILE --corpus-dir DIR`.

1. **Terms.** `ragkit.analysis.lucene_english`: lowercase alphanumeric tokens, Lucene's English
   stop words removed, Porter stems. A question word and an article word match when their stems
   are equal, so "reviews" overlaps a top term "review".
2. **Discriminating vocabulary of the gold article.** Its **top k = 20** stems by
   `tf * ln(N / df)`: `tf` is the stem count in the article, `df` the number of the N = 94 corpus
   articles (metadata comment stripped) that contain the stem; stems with weight 0 are dropped,
   ties broken alphabetically.
3. **Overlap.** Number of distinct question stems in that top-20 set. A question with several
   gold articles is judged against each, and the largest overlap counts.
4. **Threshold.** At most **1** overlapping stem (overlap <= 1). For reference, with the same
   measure the earlier sets have 3 of 39 (hard), 0 of 29 (fresh) and 6 of 42 (`qa_eu.jsonl`)
   questions at or below that threshold; the medians are 5, 4 and 3. These numbers were
   computed to calibrate the rule; no retriever was run.
5. **Names.** A question may not contain an act number (`\d{4}/\d{1,4}`), an article number
   (`Article 5`, `art. 5`), the words regulation / directive / annex, or the gold article's heading
   title as a phrase (for example `Bidding zone review`).

The rule looks at the question text and the gold article's term statistics only. It runs no
retrieval, so it can be used while writing without touching the scores.
