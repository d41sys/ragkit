# Review of `qa_eu_hard.jsonl`

## What this is

`src/ragkit/data/qa_eu_hard.jsonl` is a 39-question retrieval set over the 94-article EU corpus
(`src/ragkit/data/corpus_eu/`). It was built to stress the failure the context prefix is meant to
fix: the right article has near-identical siblings, in its own act or in another act, and the
question does not name its act. It is a stress set, not a sample of natural questions, so its
absolute scores say nothing about everyday queries. Its use is in `docs/ctxprefix_hard_protocol.md`.

Provenance: every item has `"status": "draft-agent"`. An AI agent drafted all 56 candidates after
reading the cited articles, then audited them (below), and an independent Sonnet pass
(`claude-sonnet-5-5`) checked them against the whole corpus. **No human reviewed any item.** Drafter
and reviewer are the same model family, so shared blind spots are possible. Treat the labels as
"agent-checked", not as gold.

No retriever, BM25 index, embedder or `ragkit` scoring command was run on any candidate before the
final file was committed. `git log` shows the order: candidates (`8ec1598`), then this review and
the frozen set with the protocol, then the first scored run.

## Design

The drafter walked the corpus for parallel text and wrote one question per distinction that a
reader could only get right by knowing which article or act applies:

| Cluster | Parallel text | What tells the answers apart |
|---|---|---|
| Balancing platforms | 2017/2195 art 19, 20, 21, 22 (four near word-for-word platform articles) | who develops the proposal, whether merit order lists exist, the extra "principles of paragraph 2" in art 21(5) |
| Reserve capacity allocation | 2017/2195 art 38, 39, 40, 41, 42 (co-optimised, market-based, economic-efficiency allocation) | contracting horizon, single optimisation, avoided-cost valuation, DC-link exception |
| Netting vs cross-border activation | 2017/1485 art 146 and 147 | manual activation and control program (147(4)(b)), cap on netting power (146(6)) |
| Firmness, curtailment, income | 2015/1222 art 70-73, 2016/1719 art 25/53/56/57, 2019/943 art 19, 2017/1485 art 22 | emergency vs force majeure, explicit vs long-term rights, who is asked to start a curtailment calculation |
| Bidding zone review | 2015/1222 art 32, 2019/943 art 14 | 15 months from the decision to launch vs 12 months from methodology approval; single-TSO launch |
| Storage and operator tasks | directive 2019/944 art 36 vs 54, art 31 vs 40, art 15/16/17 | DSO vs TSO deadline and grounds, cybersecurity task only in art 40, loss-energy facilitator only in art 31 |
| 2026 thresholds | 2019/943 art 5(4) vs art 12(5) | same 200 kW number, one for balance responsibility and one for priority dispatch |
| Parallel definitions and rules | 2017/2195 art 2 vs 2019/943 art 2; 2019/943 art 8(4) vs 2017/2195 art 53 | the wording that exists in only one of them |

Questions avoid act numbers (none matches `\d{4}/\d{1,4}`) and are written as descriptions, not
as the article's heading. Composition of the final 39: 20 `cross-act`, 12 `near-duplicate`,
7 `paraphrase`; gold articles by act: 2017/2195 14, 2019/943 8, 2017/1485 6, 2015/1222 5,
directive 2019/944 5, 2016/1719 1. For 19 of the 39 every drafter-listed decoy sits in the gold
article's own act, so the wrong-act count cannot flag a confusion that stays inside the act.
The `decoys` field is the drafter's list of look-alike articles; it is documentation and is not
used by any scoring code.

## Pass 1: adversarial self-review (drafter, before any reviewer)

For every candidate the drafter re-read the cited paragraph against the answer, checked every
number, actor and deadline, confirmed that a question about one article could not be answered from
its sibling, and grepped the corpus for unique terms (`cybersecurity`, `use-it-or-sell-it`,
`neutral market facilitator`, `30 seconds`, `eight hours`, `market spread`, `separate internal
account`, `bid size`, `200 kW`, `4 July 2019`). A script checked that every relevant id and decoy
exists, that provenance ids equal relevant ids, that each cited paragraph number exists, that every
quoted phrase in an answer occurs in the cited articles, and that no question repeats an item of
`qa_eu.jsonl` or `qa_eu_fresh.jsonl`. Result: one answer was incomplete (2017/2195 art 54(3)(c) only
holds "until the implementation of the proposal pursuant to Article 52(2)"), fixed before commit.

## Pass 2: independent Sonnet review

Tool: `src/ragkit/hardset_review.py` (standard-library `urllib`, no SDK). For each question the
model gets the **entire corpus** as a cached prompt prefix plus the question, the claimed gold ids
and the reference answer. It is not told the drafter's decoys and never sees any retriever output.
It returns whether the gold article(s) support the answer, which other articles also answer it,
and whether the question is ambiguous. Every response is cached under the SHA-256 of its request in
`runs/hardset_review_cache/` (committed), so the review reproduces without a key; the unit tests do
not need one. Raw outputs: `runs/hardset_review_round1.md` / `.jsonl`, `runs/hardset_review_round2.md` / `.jsonl`.

**Round 1 (56 candidates, 56 live calls).** Gold supports the answer: 56 yes, 0 partial, 0 no.
Ambiguity: 27 none, 29 minor, 0 major. Other articles that "partially" answer: 22 candidates;
**none that "fully" answers**. Examples of the partial matches: 2017/1485 art 146 for the netting
platform (art 22 of 2017/2195), 2019/943 art 17(3) vs 2017/2195 art 38(9) for free cross-border
capacity, 2015/1222 art 72 and 2019/943 art 16 for curtailment of allocated capacity.

Changes made after round 1, before any retrieval:

| Candidate | Change | Reason |
|---|---|---|
| 3 | "reserve power" -> "frequency restoration power" | loose wording flagged |
| 13 | added "Day-ahead or intraday" | the forward-capacity regime (2016/1719, market spread) could be read instead |
| 18 | question now asks for the *further* condition besides emergency and no redispatch; answer states all three | art 72 of 2015/1222 covers the first two conditions partially; only 2017/1485 art 22(1)(i) has "all TSOs at the interconnector agree" |
| 24 | "may rely on" -> "accepted ground"; answer notes that points (a) and (c) of art 54(2) also apply | necessary-but-not-sufficient condition |
| 25 | "Which grid operators" -> "Which type of system operator ... in the list of its tasks" | plural read as several operator types |
| 32 | "From 2026" -> "commissioned from 2026" | the 200 kW limit applies by commissioning date |
| 51 | "Can a single TSO..." -> "Under what conditions can a single TSO..." | a bare yes/no omits the conditions |
| 53 | "a party" -> "a balance responsible party" | the defined term |

**Round 2 (8 reworded items, 8 live calls).** All 8: gold supports the answer; ambiguity none for
5, minor for 3; none has a fully answering other article.

**Drop of 17 candidates, by a rule fixed before any retrieval.** The target is the
near-duplicate failure, so a candidate was kept only if the corpus holds a sibling that has
parallel wording (a decoy), either flagged by Sonnet or by the drafter's cluster map. Seventeen
candidates are single-fact lookups with no sibling (the 30-second activation delay, the 5 % share of
containment reserve, the 2-hour reservoir recovery, the 500 kW bid size, the 8-hour scheduling
closure, the four directive items on the date for schemes without separate accounting, delegation, communities and aggregators, the compensation payer for hosted cross-border flows, the 30-minute
publication delay, the 12-month contract extension, the 15(4) date, the definitions of priority
dispatch and reserve capacity, the UIOSI treatment, the 146(9) block question, and the second of two
questions on 2019/943 art 19). They remain in commit `8ec1598`. The drop cut the easiest items and
so makes the set harder; this is disclosed, and it is why a result on this set cannot be read as a
rate on natural questions.

Final: 39 questions (verdicts in the `sonnet` field of each item: round 1 for unchanged items,
round 2 for the 8 reworded ones), 22 of which have at least one article that Sonnet marked as
partially answering.

## Cost

64 live calls (cap 150; one earlier request was rejected with HTTP 400 for a deprecated
`temperature` parameter and produced no response). Token usage over the cache, from the API
`usage` fields: 9,839 input, 99,476 cache-creation, 6,266,988 cache-read, 17,111 output tokens.
At assumed list prices for a Sonnet-class model ($3 / $3.75 / $0.30 / $15 per million tokens) that
is about $2.5; the price table is an assumption, not read from the API.

## Limits

- **Same model family** drafted and reviewed. Sonnet was lenient: it returned "yes" for all 56 gold
  checks and found no fully competing article, so it confirms that nothing it could find is
  mislabelled but cannot prove that no sibling exists. It saw the gold ids and the answer, which
  makes confirmation easier than blind labelling.
- **No human labels.**
- **Selection.** 56 -> 39 by a stated rule that removes questions with no sibling. The set is
  harder than a natural sample by construction.
- **Wrong-act blind spot** for confusions inside one act (see Design).
- If a later reviewer finds an error, the scored numbers stay as committed and the correction is
  reported next to them.
