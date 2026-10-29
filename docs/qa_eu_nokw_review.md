# Review of `qa_eu_nokw.jsonl`

## What this is

`src/ragkit/data/qa_eu_nokw.jsonl` is a 49-question retrieval set over the 94-article EU corpus
(`src/ragkit/data/corpus_eu/`) written to **avoid the discriminating vocabulary of the gold
article**. The mechanical rule (top-20 TF-IDF stems of the gold article, at most 1 shared stem,
no act, article number or heading named) is fixed in `docs/ctxprefix_nokw_protocol.md` part 1 and
enforced by `tests/test_ragkit_nokw.py`. The questions target the same near-duplicate and
cross-act clusters as the hard set (the act or article decides the answer) but are phrased in lay
words. Use: `docs/ctxprefix_nokw_protocol.md`. It is a stress set, not a sample of natural
questions.

Provenance: every item has `"status": "draft-agent"`. An AI agent wrote the 54 candidates after
reading the cited articles; an independent Sonnet pass (`claude-sonnet-5-5`) checked them against
the whole corpus. **No human reviewed any item**; drafter and reviewer share a model family. No
retriever was run on any candidate (the rule's TF-IDF check reads the gold article and the question
only).

## Review procedure

Tool: `src/ragkit/hardset_review.py` (unchanged; standard-library `urllib`). Per question the
model sees the whole corpus as a cached prompt prefix plus the question, the claimed gold ids and
the reference answer; it is not told the drafter's decoys. It returns whether the gold article
supports the answer, which other articles also answer, and whether the question is ambiguous.
Every response is cached under the SHA-256 of its request in `runs/nokw_review_cache/`
(committed), so the review reproduces without a key. Raw outputs: `runs/nokw_review_round1.md` /
`.jsonl` and `runs/nokw_review_round2.md` / `.jsonl`.

### Round 1 (54 candidates, 54 live calls)

Gold supports the answer: 52 yes, 2 partial (items 0 and 34), 0 no. Ambiguity: 9 none, 45 minor,
0 major. Other articles listed: 24 items had at least one; **one "fully"** (item 28: Article 15(5)
of 2019/943 gives the same last-resort answer as Article 14(8)); all others "partially".

Changes before any retrieval:

| Item (candidate index) | Change | Reason |
|---|---|---|
| 0, imbalance netting platform | answer reworded as a contrast with Articles 19-21 | "no merit order lists" is not in Article 22 itself (partial) |
| 27, who decides on the method | question names how the examination is carried out | could be read as the later Article 14(8) decision |
| 28, last-resort decision | Article 15 added to the relevant ids; answer cites both | Article 15(5) fully answers |
| 34, older renewable plant | answer corrected to "capacity is increased"; question names the pre-mid-2019 plants | the reference said "modified" (partial) |
| 45, unavailable volumes | question names the provider as the reporter | could be read as the TSO-side declaration of Article 29(14) |

### Round 2 (5 reworded items, 5 live calls)

All 5: gold supports the answer ("yes"), ambiguity minor, no article that fully answers instead.
Several minor-ambiguity suggestions (name the regulation, use the official term) were not taken:
they would break the naming rule, which is the point of the set.

### Drop of 5 candidates, by a rule fixed before any retrieval

Candidates of type `paraphrase` (single-fact lookups with no parallel article: indices 45, 50, 51,
52, 53 of the candidate file) were dropped: the set exists to put near-duplicates in competition,
and these have none. They remain in commit `c37af66`.

Final: 49 questions (28 near-duplicate, 21 cross-act). Verdicts are in the `sonnet` field of
each item (round 1 for unchanged items, round 2 for the 5 reworded). 22 of the 49 have at least one
article Sonnet marked as partially answering; none has one that fully answers. 21 have a drafter
decoy in another act; for 28 all decoys sit in the gold article's own act, where the wrong-act
count cannot register a confusion. Gold articles by act: 2017/2195 15, 2019/943 11, directive
2019/944 9, 2017/1485 6, 2015/1222 5, 2016/1719 3.

## Cost

59 live calls (cap 150). Token usage over the cache, from the API `usage` fields: 7,309 input,
99,476 cache-creation, 5,769,608 cache-read, 19,151 output tokens. At assumed list prices for a
Sonnet-class model ($3 / $3.75 / $0.30 / $15 per million tokens) that is about $2.4; the price
table is an assumption, not read from the API.

## Limits

- Same model family drafted and reviewed; Sonnet was lenient (52 of 54 "yes" on the first pass),
  saw the gold ids and the answer, and is unlikely to see a competing article that is phrased very
  differently. No human labels.
- Heavy paraphrase makes several questions loose ("scheme", "back-up", "takings"); Sonnet rated 42
  of the 49 as minor-ambiguity, usually for that reason. A retriever that fails on such a question
  may be failing the question, not the corpus.
- Selection: 54 -> 49 by the stated rule; 5 items were reworded or re-labelled after review.
- The overlap rule measures the gold article only; shared words with siblings are not constrained.
