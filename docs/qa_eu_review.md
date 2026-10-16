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

# Expanded unanswerable set: 33 added questions, adversarial review

**Why.** With 12 unanswerable questions every abstention AUROC interval was too wide to rank signals (see README). `unanswerable_eu.jsonl` now holds 45 on-topic unanswerable questions: the original 12 (`"cohort": "original-12"`, unchanged apart from that tag, still first in the file) and 33 new ones (`"cohort": "expanded"`). No human reviewed these either, and the reviewer is again an agent (probably the same model family as the drafter, not verified).

**Protocol, fixed before any score was computed on the new questions.** (1) Draft 33 candidates in four near-miss types, tagged `miss_type`: `wrong_regulation` (9: a real EU act that the corpus only cites by number, such as the generator, HVDC, demand-connection, REMIT, cybersecurity, emergency-restoration, gas-storage, energy-efficiency and renewables acts), `absent_article` (8: a plausible article of an included act that is not among the 94), `missing_number` (9: the corpus names the quantity but states no value) and `national_rule` (7: Member State or non-EU detail). (2) For each, run BM25, dense and hybrid top-10 (recursive 500/50 chunks, collapsed to articles) and grep the corpus for the distinctive terms and numbers; `provenance` lists the closest articles found. Questions were not kept or dropped by looking at any abstention signal: no signal was computed on the new questions until the review below was applied. (3) Status `draft-agent`. (4) A separate agent, instructed to prove each claim wrong, re-ran the retrieval for every question, read every article in the union of the top-10 lists, grepped at least three patterns per question, and checked every factual statement in the `reason` text. (5) Fixes applied, status set to `reviewed-agent`; changed items carry a `review_note`.

**Result of the review.** Reviewer verdicts, counted from its table: 23 OK (five of them with a wording suggestion that was applied), 7 `reason` factually wrong, 2 partial (35 and 38 below), 1 ambiguous (43). No question was found answerable from the corpus, and none was dropped. 15 of 33 items changed: 4 questions reworded (items 29, 35, 38, 43 by 0-indexed line in the file), 11 `reason` texts corrected or extended, 2 provenance lists extended (31, 44; 31 also reworded in reason).

| Line | Changed | Why |
|---|---|---|
| 12, 14, 18 | reason | Citations were described wrongly (2016/631 is cited for LFSM/FSM in the FCR K-factor; 2017/2196 is also cited in 2017/2195 Art. 2; Art. 21 mentions short-circuit as a failure risk of topology changes, not levels). |
| 15 | reason | "Electricity storage" corrected to "energy storage": Arts. 36 and 54 do not say which. |
| 17 | reason | Art. 12(3)(c) does refer to the Union's binding renewable target, without a value; its 50 % is a Member State derogation condition. |
| 22 | reason | Art. 32(5)'s 100 000 customers exempt from the network development plan duty in para 3, not from unbundling. Closest call: a loose grader could mistake 100 000 for the answer. |
| 25 | reason | The reserve-price rule had been stated backwards: Art. 16(6) bars reserve prices except for exempt new interconnectors. Borderline. |
| 29 | question, reason | FRR is dimensioned per LFC block, not per synchronous area (false premise). |
| 31 | reason, provenance | 2015/1222 Art. 73(3) gives a one-week distribution deadline, but no interest or late-payment rule. |
| 33 | reason | The 1,5 MW in 2017/1485 Art. 158(1)(e)(ii) is a monitoring threshold, not a prequalification minimum. |
| 35 | question, reason | Art. 18(6)(h) of 2017/2195 requires T&Cs to contain a maximum settlement period without a value. Closest call; the question now asks for an EU-wide number. |
| 38 | question, reason | Art. 13(7) of 2019/943 states a compensation floor that applies in Germany, so the compensation question was partly answerable; replaced by one only German law answers. |
| 42 | reason | "GB" appears in 2017/1485 only as a synchronous-area label. |
| 43 | question, reason | Compound question (law and legal form) split to a single ask. |
| 44 | reason, provenance | The articles say "the relevant regulatory authorities" (plural); 2017/2195 Art. 4 added. |

Unchanged after review: 13, 16, 19, 20, 21, 23, 24, 26, 27, 28, 30, 32, 34, 36, 37, 39, 40, 41.

## Judgement calls a human should know about

- Lines 22, 35 and 25 are deliberate lures: the corpus has a nearby threshold, deadline or pricing statement that does not answer the question. A stricter or looser annotator could disagree; they were kept because the benchmark is meant to contain near-misses.
- Some `wrong_regulation` items (gas storage, energy efficiency, renewables) are less on-topic for congestion and balancing than the rest. They are tagged so the AUROC can be recomputed without them (`miss_type`), and they were kept for the same reason as the REMIT or cybersecurity items: the corpus mentions their vocabulary.
- The corpus holds the original 2015 to 2019 texts only, so a question answered by a later amendment is unanswerable here by construction.
