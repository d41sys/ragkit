# EU QA set: second-pass review

**What this is.** The 42 answerable and 12 unanswerable questions in `src/ragkit/data/qa_eu.jsonl` and `unanswerable_eu.jsonl` were drafted by an AI agent. The repo owner has no domain knowledge of EU grid regulation and could not review them, so a separate agent pass was run as a substitute: an adversarial second annotator. **No human reviewed these labels, and no human domain expert has seen them.** The reviewers are separate agent instances, probably of the same model family as the drafter (not verified), so errors such a model makes consistently can survive both passes (single-annotator risk remains).

**Method.** For every answerable question a reviewer agent read the cited article(s) in full and checked (a) that the reference answer is correct and fully supported (every number, deadline, actor and exception), (b) that the question has one reading, (c) for `paraphrase` items, lexical overlap with the key sentence, (d) for `multi-hop` items, that both articles are really needed. It then read every article in the BM25 top-10, dense top-10 and hybrid top-10 for that question (computed with the library, recursive 500/50 chunks, chunks collapsed to articles) plus corpus greps for restated rules, and added to `relevant_doc_ids` any article that independently answers the question. For every unanswerable question the reviewer grepped the corpus for the distinctive terms and read the cited articles and the retrieval candidates. I (the coordinating agent) re-read the corpus text behind the less obvious verdicts (Art. 13(3), 16(3)/(8), 37(1), 54(6), 55, 72, 157(2), 1719 Art. 25/53, the price-limit articles) and re-ran the candidate retrieval for every rewritten question. The existing test that every quoted span in a reference answer is a verbatim substring of the cited articles passes on the new answers.

**Counts.**

| | Kept (status only) | Fixed | Dropped | Relevant ids added |
|---|---|---|---|---|
| Answerable (42) | 17 | 25 | 0 | 3 added, 1 removed |
| Unanswerable (12) | 9 | 3 | 0 | n/a |

"Kept" items were confirmed and had only `status` changed to `reviewed-agent`; "fixed" items also carry a `review_note`. No item was dropped: every defect could be repaired by rewriting. After the review the answerable set is 17 `direct`, 17 `paraphrase`, 8 `multi-hop` (was 14 / 20 / 8). No unanswerable question turned out to be answerable from the 94 articles.

Tally of what changed among the 25 fixed answerable items (an item can appear in several): question rewritten 13, reference answer rewritten 19, type changed 4, relevant ids changed 4.

## Answerable changes

| Item | Changed | Reason |
|---|---|---|
| 1 | answer | Answer omitted the rider on condition (d): Art. 13(3)(d) also requires an action plan or Art. 16(8)-compliant minimum capacity. |
| 2 | answer, +ids `reg_2017_2195_art30` | Answer now mentions the technical price limits Art. 10(1) carves out for the balancing timeframe; Art. 30(2) of 2017/2195 (harmonised balancing price limits) added as relevant. |
| 3 | answer | Answer presented the 10 % cap as unconditional; added the Art. 41(2) exception (contracting up to two days ahead, DC borders). |
| 11 | answer | Answer tightened: para 7 applies only outside the regions in paras 2 to 4 and the request goes to the regulatory authorities. |
| 12 | +ids `reg_2017_2195_art2` | Art. 2(27) definition independently states that bids can no longer be submitted or updated after gate closure; added as relevant. |
| 14 | question, answer | Question said 'country' (rule is about the bidding zone) and reused 'capacity/inside/own'; rewritten. Answer now notes the para 3/9 derogations and the 70 % compliance rule. |
| 16 | answer | Answer over-claimed 'at least the higher of'; Art. 13(7) also allows a combination of both. |
| 17 | answer | Answer said 'and' where Art. 19(2) says 'or' and omitted that the priority uses must be adequately fulfilled first. |
| 18 | question | Question reused 'independent aggregator', 'demand', 'supplier'; rewritten without them. Answer unchanged. |
| 19 | answer, type paraphrase→direct | Question reuses 'regional security coordinators replace'; retyped paraphrase to direct. Answer adds that the date follows regulatory approval of the proposal. |
| 20 | question, answer | Question reused 'framework/distribution/flexibility' and the answer ignored the 'instead of upgrading' half; both rewritten. |
| 21 | question, answer | Answer applied the 2026 six-month cap (para 11) to the default rule; rewritten to cover the default (para 9) and the derogation regime (paras 10 to 11). |
| 25 | question, answer, type paraphrase→multi-hop, +ids `reg_2017_2195_art54` | Art. 55 alone does not say a surplus is a positive imbalance (Art. 54(6)(b) does); Art. 54 added, retyped multi-hop, question de-copied from the article wording. |
| 26 | type paraphrase→direct | Question reuses 'same scheduling area'; retyped paraphrase to direct. Answer unchanged. |
| 28 | question, answer | Question copied Art. 18(3); answer lacked the 'where appropriate' hedge. Both rewritten. |
| 29 | question | Question near-copied Art. 23(5) and omitted its second condition; rewritten to include both. Answer unchanged. |
| 32 | question, answer | Answer led with the wrong actor and dropped the 48-hour scope of Art. 25(1); question lifted the key sentence. Both rewritten. |
| 33 | question | Question copied 'explicit allocation / curtailed / emergency / compensation'; rewritten. Answer unchanged. |
| 34 | question | 'then' left open which branch the 31 December 2025 date applies to; question limited to the action-plan branch. |
| 35 | question, answer | Art. 16(3) already says RCCs carry out coordinated capacity calculation, so Art. 37 was decorative; first half now asks about Art. 37(1)(k), which only Art. 37 answers. |
| 36 | question, answer | Art. 13(2) and Art. 30(1)(b) state the same rule, so either alone answered the question; each half now rests on one article only. |
| 37 | answer | Answer omitted the new-interconnector variant and the two-day / DC exception in Art. 41(2); 'TSOs may apply' corrected to 'two or more TSOs may propose'. |
| 39 | answer | Answer omitted the Art. 23(3) duty to coordinate with affected SGUs and DSOs. |
| 40 | answer, type multi-hop→direct, −ids `reg_2019_943_art5` | Dir. Art. 17(3)(d) alone answers both halves (Art. 5 only elaborates); retyped multi-hop to direct and Art. 5 dropped from the labels. |
| 41 | question, answer | Answer omitted the FRR sharing-agreement reduction in Art. 157(2)(j); question tightened to the Art. 157(2)(e) lower bound ('at least cover' also fits para 2(h)). |

Items kept unchanged: 0, 4, 5, 6, 7, 8, 9, 10, 13, 15, 22, 23, 24, 27, 30, 31, 38.

## Unanswerable changes

| Item | Changed | Reason |
|---|---|---|
| 5 | question | Original question presupposed ACER can fine, which is false (Art. 19(5) only has regulators informing ACER); rewritten. Penalties are left to Member States (Art. 66, not in the corpus). |
| 8 | reason | Reason text corrected: Art. 8(3) says bid sizes of 500 kW or less, for day-ahead and intraday products only. |
| 9 | question | 'Grid congestion charges' was undefined and loaded; question sharpened. Still no national data in the corpus. |

Unanswerable items kept unchanged: 0, 1, 2, 3, 4, 6, 7, 10, 11. Items 6 (gate closure minutes), 7 (harmonised maximum clearing price) and 8 (FCR minimum bid size) were the closest calls: the corpus names the quantity but states no value (Arts. 21/24 of 2017/2195 require a harmonised gate closure time with no number; Art. 10(2) of 2019/943 allows limits without values; Art. 8(3) gives 500 kW or less for day-ahead and intraday products only and Art. 157/158 of 2017/1485 give FRR, not FCR, sizes).

## Judgement calls a human should know about

- Item 2: Art. 30(2) of 2017/2195 was added as relevant because Art. 10(1) itself carves out technical price limits in the balancing timeframe and Art. 30(2) is where those limits are provided for. A stricter annotator might label only Art. 10.
- Item 12: Art. 2(27) is a definition, not an operative rule; it was added because it states that bid submission or update is no longer permitted after gate closure.
- Item 25 was made `multi-hop` because mapping "surplus" to "positive imbalance" is only stated in Art. 54(6)(b); this makes it a weaker multi-hop item than the others.
- Item 36's second half is Art. 30(3)(c) (no additional charges for cross-border exchange of balancing energy); the first draft's second half duplicated Art. 13(2) word for word, so the item could not need both articles.
- Rewriting paraphrase questions to avoid the article wording makes them harder for lexical retrievers. Part of the drop in retrieval numbers in the README is therefore the stricter set, not a change in retrievers.
