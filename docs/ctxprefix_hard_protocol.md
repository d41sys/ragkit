# Context prefix on a hard held-out set (frozen before the first scored run)

Follow-up to `docs/ctxprefix_protocol.md`. Everything that protocol defines (chunks, prefix
format, retrievers, metrics, bootstrap, wrong-act count, adoption rule, what was "not tried") is
**unchanged**. This file adds one question set, a saturation guard, predictions and a rule for what
happens after a verdict. It is committed before any retriever has been run on the hard set.

## Why a second set

The first run (`runs/ctxprefix_protocol.txt`, 29 fresh questions) gave a null result, and the
baseline there was already close to the ceiling: BM25 MRR 0.891, Recall@5 0.948, and only 3 of 29
top-1 documents in the wrong act (1 for the hybrid). A prefix that fixes wrong-act retrieval cannot
show a gain when there is almost nothing to fix. So the right reading of that run is **"no
detectable effect on that set"**, not "the prefix does not help": the set could not tell. This
protocol builds a set in which the failure is common, so that the same question can be asked where
it is able to show up.

## The hard set

`src/ragkit/data/qa_eu_hard.jsonl`: 39 answerable questions on the same 94-article corpus
(`status: draft-agent`; design, self-review and independent Sonnet review in
`docs/qa_eu_hard_review.md`). 20 `cross-act`, 12 `near-duplicate`, 7 `paraphrase`. Each targets
parallel articles inside one act (2017/2195 art 19-22 and 38-42, 2017/1485 art 146/147, directive
art 36/54) or across acts (2015/1222 art 32 vs 2019/943 art 14, 2015/1222 art 72 vs 2016/1719
art 56, 2019/943 art 5(4) vs 12(5), directive art 31 vs 40, two definitions articles). No question
contains an act number; they are paraphrases that avoid the heading wording where that was
possible. The drafted set had 56 candidates, 17 single-fact items without a sibling were dropped by
a rule fixed before any retrieval (details and the list are in the review document), and 8 were
reworded after the Sonnet pass. All of this is in git before the first score:
candidates (`8ec1598`), then the frozen set, review and this protocol.

### Disclosed taint and bias

- The drafter had seen the **aggregate** result of the first run (the null result, the ceiling,
  the by-type pattern for BM25) and the 42 old and 29 fresh questions' texts. It did not see any
  per-question retrieval output. No retriever was run on any candidate.
- The drafter read the corpus articles to write the questions, and chose clusters because they
  have parallel text. That is the intended design, and it also means the set is **built to favour
  an effect** of a prefix that carries the act and article heading: a sibling-confusion set is the
  best case for it. A gain here is not evidence of a gain on natural queries.
- The 17-candidate drop removed the easiest items; see the review document.
- Labels are agent-made and agent-checked only. Sonnet (same model family) found no fully
  competing article for any item; that does not prove none exists.
- 19 of 39 questions have all look-alike articles inside the gold article's own act, where the
  wrong-act count cannot register a confusion.

## Saturation guard (pre-registered)

If baseline **BM25 Recall@5 on the hard set is above 0.85**, the set is called **saturated**, and
the report says so first. A saturated set is not a valid test of a failure it does not exhibit:
a null on it is "no detectable effect on a saturated set", not evidence of no effect, and it is not
used to argue the prefix is harmless. The adoption rule below is still computed and reported
unchanged. The threshold is fixed here; `ragkit ctxprefix --hard` prints the guard and its value.

## Run

`uv run ragkit ctxprefix --hard` scores BM25 and the hybrid, `base` vs `prefix`, on the hard set,
and on the synthetic set (the regression guard), with the unchanged bootstrap (95% percentile
interval over questions, 2000 resamples, seed 0) and paired intervals for dRecall@5, dMRR,
dnDCG@10, the better / worse / tied counts, the wrong-act top-1 count with fixed / broken, and the
by-type breakdown. Output goes to `runs/ctxprefix_hard.txt`.

## Decision rule (the one in `docs/ctxprefix_protocol.md`, applied unchanged)

Per retriever: **adopt** iff (1) the paired dMRR (`prefix` - `base`) on the **hard** set has a CI
whose lower bound is above 0 and (2) on the synthetic set dMRR is at least -0.02 with CI upper bound
not below 0. The hard set takes the place the fresh set had in that rule. Otherwise **rejected** when
the hard-set dMRR CI is entirely below 0 or the synthetic check fails, **not adopted (no evidence)**
when the CI includes 0. Two retrievers, two tests, no multiplicity correction. The fresh-29 and old-42
sets are re-reported next to the result for context and do not gate. The saturation guard does not
change the rule.

## Predictions (written before the run)

1. BM25 base Recall@5 on the hard set is between 0.65 and 0.90; probability that it exceeds 0.85
   and the set is saturated: about 25%. The author's expectation is that paraphrasing away the
   heading words lowers lexical overlap with the gold chunk more than it helps the prefix.
2. BM25 wrong-act top-1 is at least 6 of 39 on the base text.
3. BM25 dMRR is positive, between +0.02 and +0.10; probability that its CI lower bound is above 0
   (adoption for BM25): about 30%.
4. Hybrid dMRR is within +-0.05, CI includes 0; probability of adoption: about 10%. The old-42
   result (the hybrid loses with the prefix) is the reason for the low value.
5. Recall@5 changes by less than 0.05 for both retrievers.

These are recorded so that a surprise can be called a surprise. They are not part of the rule.

## After the verdict

- **Neither adopted** (or the set is saturated): report plainly, change no default, README states
  "no detectable effect on this set" and, for the hard set, what the set could and could not show.
- **A retriever adopted**: the prefix becomes the default chunk text for that retriever where the
  pipeline and the EU headline paths chunk the corpus (`ragkit eval --corpus eu` first), and every
  EU headline number that changes is re-run and reported **old vs new** in the README, losses
  included. A lone pass for one retriever is reported as such. Adoption does not remove the caveats
  above (the set favours the prefix, agent labels, 39 questions, two tests).
- Corrections to the labels later are reported next to the committed numbers; the numbers are not
  re-run to make the result look different.

## Not tried

Unchanged from `docs/ctxprefix_protocol.md`: shuffled-prefix control, prefix on structured chunks,
prefix in only one index, field weighting, model-generated chunk context, prefix-format tuning. In
addition: a blind (non-agent) relabelling of the hard set, and a per-decoy confusion metric that
would also count same-act mistakes.
