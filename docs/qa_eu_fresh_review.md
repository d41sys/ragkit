# Review of `qa_eu_fresh.jsonl`

## What this is

`src/ragkit/data/qa_eu_fresh.jsonl` is a 29-question retrieval set over `src/ragkit/data/corpus_eu/` (94 EU electricity-law articles). An agent drafted it before any prefix variant was scored. A separate agent pass then reviewed it, as described below. No human reviewed it. Both passes come from the same model family, so shared blind spots are possible: both could misread an article the same way, or miss the same sibling article. Treat the labels as "agent-checked", not as gold.

Every item now has `"status": "reviewed-agent"`. Items changed in review also carry a `review_note`.

## Method

The reviewer read each cited article in full and did three checks per question. No retriever, BM25, embedder or `ragkit` command was run, and the older `qa_eu.jsonl` and any run outputs were not read. Everything was done by reading and by text-grepping the corpus files.

1. **Support.** Every number, deadline, actor and exception in the answer was compared with the article text, and the cited paragraph was confirmed to contain it.
2. **One reading.** The question must have a single reasonable interpretation and must not copy the article's key sentence.
3. **Hidden-relevant search.** Distinctive phrases and numbers from each answer (`one hour before`, `market spread`, `550 g`, `15 minutes`, `double charges`, `force majeure`, `HVDC`, `reliability margin`, `curtail`, `congestion income`, `phase-out`, etc.) were grepped across the whole corpus. Sibling articles were read explicitly: 2017/2195 art 19-22, directive 2019/944 art 33/36/54, 2016/1719 art 25/53/56/57 against 2015/1222 art 70/72/73 and 2019/943 art 19, 2017/1485 art 143-147/166/176, 2015/1222 art 32/33 against 2019/943 art 14, 2019/943 art 21/22, and directive 2019/944 art 15-17.

## Counts

| Outcome | Items |
|---|---|
| Kept unchanged | 23 |
| Fixed | 6 |
| Dropped | 0 |
| Relevant doc ids added | 2 (`reg_2017_1485_art176` on item 16, `reg_2019_943_art17` on item 23) |
| Provenance paragraphs added | 3 (item 12: `art53#1`; item 16: `art176#4`; item 23: `art17#5`) |

All answers were found factually supported by the cited paragraphs. The six fixes concern ambiguity, missing sibling articles, or provenance. After the edits, every relevant id exists as a corpus file and every provenance doc id is in `relevant_doc_ids`.

## Changed items

Index is the 0-based line number in the file.

| Index | What changed | Why |
|---|---|---|
| 11 | Question scoped to "the day-ahead and intraday capacity allocation and congestion management regulation (2015/1222)". Ids and answer unchanged. | 2016/1719 art 57(3) applies the requirements of 2015/1222 art 73 by reference. An unscoped "methodology for sharing congestion income" question fits both, but art 57 does not itself state the art 73(2)(e) content. |
| 12 | Provenance now `art53#1` and `art53#2`. | The reporting and publication duty in the answer is in paragraph 1; compensation with the market spread is in paragraph 2. |
| 16 | Question reworded to "frequency restoration reserve activation". Added `reg_2017_1485_art176` to relevant ids, with provenance `art176#4`. Answer extended. | The old wording ("reserve power ... different synchronous areas") also matched 1485 art 146(5), which has the same HVDC sentence for imbalance netting power interchange. Art 176(4) (HVDC operators control the flow for FRR exchange between synchronous areas) also supports the answer. |
| 23 | Added `reg_2019_943_art17` (provenance `art17#5`). Answer extended. Type changed from near-duplicate to cross-act. | 2019/943 art 17(5) repeats the "shall not increase the reliability margin" rule and covers sharing of reserves too. It answers the question independently. |
| 24 | Question scoped to "the electricity balancing guideline (Regulation 2017/2195)", with the deadline "counted from entry into force" and the exemption actor stated as "which operators may jointly request". | 2019/943 art 8(4) also sets a 15-minute settlement period, but with a different regime (1 Jan 2021, derogations to end-2024, 30 minutes from 2025) and no statement of who may request an exemption. Unscoped, two articles gave different answers. |
| 26 | Question now asks about "reductions that TSOs made when validating cross-zonal capacity" by "each coordinated capacity calculator". | "Units computing cross-border capacity" could be read as the regional coordination centres in 2019/943 art 16(3), which also report every three months, but on coordinated reductions and deviations, to regulators and ACER. |

## Judgement calls

Item numbers below are 0-based line indexes, as in the table above.

- **Items 0-4 (2017/2195 art 19-22).** The sibling platform articles are near word-for-word parallel, so the questions were kept. The type is `near-duplicate` because each question names its platform (automatic activation, replacement reserves, manual activation, imbalance netting) and each answer depends on a platform-specific word or number. Art 20(5) and art 21(5) differ slightly (art 21(5) also mentions "the principles set in paragraph 2"). Item 2 asks for the minimum synchronous area ("Continental Europe"), which appears only in art 22(5).
- **Items 5-8 (directive 2019/944 art 33/36/54).** Art 36(3) and art 54(4) both contain the 18-month phase-out, and art 36(4) and art 54(5) both contain the battery depreciation carve-out. The questions name the operator type (distribution or transmission), so each has one answer. The notification to the Commission and ACER (item 5) appears only in art 54(3).
- **Item 11 vs 2016/1719 art 57.** Scoping the question was chosen over adding art 57 as relevant, because art 57 does not contain the content itself.
- **Item 14.** Both 2015/1222 art 72(5) and 2016/1719 art 56(5) were kept as relevant; their wording is identical. 2019/943 art 16(2) only mentions compensation, not who assesses force majeure, so it was not added.
- **Items 12 and 13.** Art 53 (curtailment before the firmness deadline, market-spread compensation) and art 56 (force majeure, refund of the price paid) are different cases, and each question states which case applies. 2015/1222 art 72(3) covers day-ahead capacity, not long-term transmission rights, so it was not added to item 13.
- **Item 15.** 2019/943 art 17(1) says unallocated capacity goes into the next allocation. It was not added: it speaks of gate closure, not the day-ahead firmness deadline, and says "allocate", not "adjust".
- **Item 10.** "Scarcity income" is a loose paraphrase of "congestion income". Kept as a deliberate paraphrase test; only 2019/943 art 19(5) has the 1 March reporting date.
- **Item 22 (art 33 vs art 35).** 2017/2195 art 33(2) also says capacity exchange is TSO-TSO "except where TSO-BSP model is applied", but it has no deadline and not the replacement-reserve exception. Art 35 only was kept.
- **Item 28.** "Household with its own battery" is taken to mean an active customer under directive 2019/944 art 15(5)(b). No other article contains the double-charge rule.
- **Compound questions** (items 0, 1, 3, 9, 14, 17, 24 and others) were left compound where every part is answered in one article; the answer covers all parts.
