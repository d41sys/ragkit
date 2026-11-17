# Review of `qa_eu_confirm.jsonl`

## What this is

`src/ragkit/data/qa_eu_confirm.jsonl` holds the 60 EU questions of study K-R6 (rules:
`docs/confirm_protocol.md`, "EU-new questions"): 30 natural and 30 keyword-free, gold in the 94-article
`src/ragkit/data/corpus_eu_consolidated/`. One AI agent wrote them; a separate AI agent (this
review, same model family) checked every item. **No human reviewed any item. No retriever, search,
eval or benchmark was run, and no system output was read.**

## Procedure

Per item (0-based line index): read the gold article; compared the answer with its text; grepped
the whole corpus for the key concepts and read every candidate article; compared the question with
the 199 questions of `qa.jsonl`, `qa_eu.jsonl`, `qa_eu_fresh.jsonl`, `qa_eu_hard.jsonl` and
`qa_eu_nokw.jsonl`. Accepted items got `"status": "reviewed-agent"`; REWRITE items stay
`"draft-agent"` and are not rewritten here. After editing, `python -m ragkit.nokw` was re-run: all
30 keyword-free items pass (overlap <= 1); the 29 natural items that exceed the overlap are allowed
to; no question names an act, a year or an article number.

## Counts

| | n |
|---|---|
| gold supports the answer: yes | 58 |
| gold supports the answer: partial (answer text imprecise, fixed) | 2 |
| gold supports the answer: no | 0 |
| ids added to `relevant_doc_ids` | 0 |
| answers fixed | 2 |
| REWRITE (status stays `draft-agent`) | 2 |
| accepted (`reviewed-agent`) | 58 |

No other article fully answers an accepted question. Partial overlaps, noted and not added:
22 (`reg_2015_1222_art72`(2) gives the same notice duty for day-ahead and intraday capacity),
25 (`reg_2015_1222_art25`(2) only requires coordination of real-time use of remedial actions, not
consistency), 26 (the criteria question of `qa_eu.jsonl` has the same gold and its answer lists the
risk criterion among four; the new item asks only for the kinds of risk), 42 (`reg_2017_1485_art153`(1)
sets an annual review for FCR per synchronous area, not per LFC block), 40 (`reg_2017_2195_art24`(7)
covers use of integrated scheduling bids before gate closure, a different object).

## Changed or REWRITE items

| index | change | reason |
|---|---|---|
| 4 | answer extended | "No" alone overstated: Article 17(4) lets Member States require compensation strictly limited to the supplier's activation costs; the framework bars only undue payments (Article 17(3)(e)). |
| 23 | REWRITE | "what must the operator do about the impact" has two readings with different answers in the same article: compensate the rights holders (Article 56(3), already asked in `qa_eu_nokw.jsonl`) or limit consequences and duration (Article 56(4), intended). |
| 24 | REWRITE | Same article and paragraphs as `qa_eu_fresh.jsonl` ("which earlier methodology must they take into account"); the new "Yes" differs only by Article 57(3) "apply" versus 57(2) "take into account". A different fact of the article is needed. |
| 30 | answer fixed | Article 156(6)(a) limits the share to 5 % of the FCR required for each of the CE and Nordic areas; the answer read it as one combined "whole CE and Nordic area". |

## Round 2

The writer replaced items 23 and 24; both were reviewed by the same procedure. Verdicts:

| index | new question (short) | verdict | notes |
|---|---|---|---|
| 23 | with whom must an operator liaise before cutting multi-year entitlements on an unforeseeable outside event | accept | Article 56(1) supports the answer. `reg_2015_1222_art72`(1) has the same sentence but for day-ahead and intraday capacity, so it does not answer a question about multi-year entitlements (not added). It is a different paragraph from the compensation and impact questions in `qa_eu_nokw.jsonl` and `qa_eu_fresh.jsonl`; item 22 of this file asks about the notice (56(2)). Gold `reg_2016_1719_art56` is used twice (items 22, 23), within the limit. |
| 24 | may a power store that a local grid firm has been permitted to run be used for trading | accept, answer extended | Gold `dir_2019_944_art36`. `dir_2019_944_art54`(2)(b) has the same condition but for transmission operators, so it does not answer for a distribution operator; "local grid firm" is the distribution operator (the `qa_eu_nokw.jsonl` convention: local-level versus high-voltage), so art36 alone is gold and art54 is not added. Answer was partial: the trading ban in 36(2)(b) belongs to the derogation conditions and Article 36 sets none for fully integrated network components; the answer now says so. The new fact (no trading) is not asked in the storage items of `qa_eu_nokw.jsonl`, `qa_eu_fresh.jsonl` or `qa_eu_hard.jsonl` (they ask notification, deadlines, tendering review, ancillary-service ground). |

Result: both set to `reviewed-agent`; all 60 items are now `reviewed-agent` and REWRITE is empty. Gold support over the 60 items: 57 yes, 3 partial (items 4, 24, 30; all answers fixed), 0 no. Nokw checker re-run: all 30 keyword-free items pass, 29 natural items exceed the overlap (allowed), no naming violation. Per-act questions: 2019/943 13, 2017/2195 12, 2019/944 11, 2017/1485 10, 2015/1222 9, 2016/1719 5; no article is gold for more than 2.
