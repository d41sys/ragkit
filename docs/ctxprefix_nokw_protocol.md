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

---

# Part 2: set, review, guards, predictions (committed before the first scored run)

Everything in `docs/ctxprefix_protocol.md` that is not restated here is unchanged: the recursive
500/50 chunks, the `base` and `prefix` texts, `BM25Index` and the pinned hybrid, metrics (Recall@5,
MRR, nDCG@10), the paired bootstrap (95% percentile interval, 2000 resamples, seed 0), the
wrong-act top-1 count and the adoption rule. The "not tried" list of that protocol still stands.

## The set

`src/ragkit/data/qa_eu_nokw.jsonl`: 49 answerable questions over the same 94-article EU corpus
(`status: draft-agent`; review and cost in `docs/qa_eu_nokw_review.md`). 28 `near-duplicate`
(parallel articles inside one act), 21 `cross-act` (the same matter in another act). Every
question satisfies the part-1 rule (29 have overlap 0, 20 have overlap 1; enforced by
`tests/test_ragkit_nokw.py::test_nokw_set`), names no act, article number or heading, and is a
lay paraphrase. Clusters: 2017/2195 arts 19-22 and 38-42, 2017/1485 arts 146/147/166/176,
directive 2019/944 arts 31/40 and 36/54, bidding zone review (2015/1222 art 32 vs 2019/943
art 14), 2019/943 art 5(4) vs 12(5) vs 2015/1222, 2016/1719 firmness and curtailment (2015/1222
art 72, 2016/1719 arts 25/53/56), congestion income (2015/1222 art 73, 2019/943 art 19), and the
imbalance settlement period (2019/943 art 8(4) vs 2017/2195 art 53).

### Order of work (`git log`)

1. rule, its test and part 1 (`bbe1b8f`), before any question existed;
2. 54 candidates (`c37af66`), all passing the rule, before any reviewer or retriever saw them;
3. Sonnet review (59 live calls), fixes and a rule-based drop of 5 sibling-less `paraphrase`
   items, the frozen 49 questions, this part 2, then the `--nokw` code, then the one scored run.

### Disclosed taint and bias

- The drafter (an AI agent, the same family as the reviewer) knew the aggregate outcome of both
  earlier runs and the texts of the 42 + 29 + 39 earlier questions, and read the corpus articles to
  write these. It saw **no per-question retrieval output**. No retriever was run on any candidate;
  the only computation over the corpus while writing was the TF-IDF overlap check of the rule.
- The rule removes words that are frequent *and* distinctive in the gold article. It does not
  remove every word BM25 can use: other content words ("grid", "price", "areas", "weeks", "who")
  remain, and many are shared by the gold article with its siblings. So the set is lexically
  harder, not lexically blind; how much harder is what the floor guard measures.
- The writing task rewarded questions that pass the rule, so unusually indirect phrasing
  ("blueprint", "takings", "next-day border entitlements") appears; it is not how a user would ask.
- 28 of 49 questions have every drafter-listed decoy inside the gold article's own act, where the
  wrong-act count cannot register a confusion; 21 have a decoy in another act.
- Labels are agent-made and agent-checked; Sonnet answered "yes" for 52 of 54 gold checks on the
  first pass (2 partial, fixed) and found no article that fully answers instead after the fixes
  (it found one before: item 28).

## Guards (pre-registered, thresholds fixed here)

1. **Saturation guard, unchanged.** Baseline BM25 Recall@5 above **0.85**: the set is "saturated"
   and a null on it is "no detectable effect on a saturated set".
2. **Floor guard, new.** Baseline BM25 Recall@5 below **0.15**: the report says first **"too hard
   for lexical retrieval"**. The prefix is a change to the text of chunks that are retrieved; if
   BM25 does not put the gold article in its top 5 for almost any question, there is nothing for
   the prefix to promote, and a null says nothing about the prefix. A set between the two guards is
   a valid test; the guard is on BM25 only, as the saturation guard was.
3. `ragkit ctxprefix --nokw` prints both guards and the synthetic regression guard; both are
   computed from the baseline, never from the prefix variant.

## Decision rule (unchanged)

Per retriever: **adopt** iff the paired dMRR (`prefix` - `base`) on the keyword-free set has a 95%
CI lower bound above 0 **and** the synthetic dMRR is at least -0.02 with CI upper bound not below
0. **Rejected** when the keyword-free CI is entirely below 0 or the synthetic check fails;
otherwise **not adopted (no evidence)**. Two retrievers, two tests, no multiplicity correction. The
fresh, old-42 and hard-set results are not re-reported as gating. The guards do not change the
rule: it is computed and reported whatever they say, and a guard hit changes only what the report
may claim.

## Predictions (written before the run)

1. BM25 baseline Recall@5 on this set is between 0.35 and 0.75 (point guess 0.55). Probability
   above 0.85 (saturated): about 3%. Probability below 0.15 (too hard): about 4%.
2. BM25 wrong-act top-1 is at least 8 of 49 on the base text.
3. BM25 dMRR is between -0.03 and +0.06; probability its CI lower bound is above 0 (adoption for
   BM25): about 15%. The author's reason for any gain: with the heading words absent from the
   question, the prefix matters only through shared act-title words and article numbers, which the
   questions do not contain either, so I expect it to help little even here.
4. Hybrid dMRR within +-0.05 with a CI that includes 0; probability of adoption: about 8%.
5. Recall@5 changes by less than 0.06 for both retrievers.

## After the verdict

- **Neither adopted, or a guard fired:** report plainly; change no default; the README states what
  the set could and could not show.
- **A retriever adopted:** the prefix becomes the default chunk text for that retriever in the
  pipeline and the EU headline paths; every EU headline number that changes is re-run and reported
  old vs new in the README, losses included; a lone pass for one retriever is reported as such. The
  caveats above stand (set built to separate the prefix from lexical cues; 49 questions; agent
  labels).
- Label corrections found later are reported next to the committed numbers; the numbers are not
  re-run to look different.

---

# Outcome (`runs/ctxprefix_nokw.txt`, the one scored run)

Guards: baseline BM25 Recall@5 = 0.347 -> not saturated (0.85) and not too hard (0.15); a valid
test. All 49 questions pass the overlap rule (29 at 0, 20 at 1).

| Retriever | base R@5 / MRR / nDCG@10 | prefix R@5 / MRR / nDCG@10 | dMRR [95% CI] | wrong-act top-1 base -> prefix (fixed / broken) | verdict |
|---|---|---|---|---|---|
| BM25 | 0.347 / 0.259 / 0.300 | 0.327 / 0.238 / 0.284 | -0.022 [-0.056, +0.005] | 29 -> 32 of 49 (0 / 3) | not adopted (no evidence) |
| Hybrid | 0.306 / 0.206 / 0.243 | 0.327 / 0.216 / 0.256 | +0.010 [-0.056, +0.069] | 30 -> 33 of 49 (1 / 4) | not adopted (no evidence) |

Other paired CIs: BM25 dRecall@5 -0.020 [-0.082, +0.041], dnDCG@10 -0.016 [-0.047, +0.014]; hybrid
dRecall@5 +0.020 [-0.041, +0.082], dnDCG@10 +0.013 [-0.053, +0.072]. Synthetic guard: BM25 dMRR
+0.017 [+0.000, +0.046], hybrid +0.013 [+0.000, +0.037] (no regression). Nothing was adopted, so no
default changed and the EU headline numbers were not re-run.

Predictions: 1 missed (Recall@5 0.347, just under the predicted 0.35 to 0.75); 2 held (wrong-act 29
of 49); 3 held (BM25 dMRR inside -0.03 to +0.06, CI lower bound not above 0); 4 held; 5 held (Recall@5
moved by 0.02 for both).

# Post-review note (2026-11-06, after the scored run)

The `draft-agent` statement above was true when this protocol and its one scored run were written.
Afterwards the owner reviewed Q01 and Q03 and delegated the other 47 questions to an AI agent of the
same model family (`docs/qa_eu_nokw_review.md`, "Owner-delegated review"). Two items changed: the Q29
gold answer (no effect on retrieval) and the Q46 query wording (still passes the overlap rule, at 1).
The recorded outcome above was measured on the earlier Q46 wording and was not re-run; any re-run on
the edited set would be post-hoc.
