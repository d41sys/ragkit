# EU corpus: amendments, consolidated versions and re-reviewed labels (issue #4)

The bundled EU corpus (`corpus_eu/`) is the **original Official Journal text** of six acts, retrieved 2026-10-16. Two of them were amended in 2024 by the electricity market design reform, and the network codes were aligned with Regulation (EU) 2019/943 in 2021. This page records what changed, what it does to the 94 articles and the 5 question files, and the scores on both texts.

Everything below was read on 2026-11-13. `corpus_eu/` is unchanged apart from an added `amended_by` list in its `SOURCES.json`, so every published EU number still refers to the original text. The consolidated texts are in `corpus_eu_consolidated/` (`--corpus eu-consolidated`).

## 1. Amendment history (EUR-Lex "Modified by" relations)

Source for each act: the EUR-Lex document-information page `https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:<CELEX>` (links in the table). Only "Modified by" relations are counted as amendments. "Corrected by" corrigenda, "Derogated in" decisions and "Completed by" acts are not amendments and are not listed in `SOURCES.json`.

| Act | Document information page | Amended by (date in force) | Consolidated versions | Version used |
|---|---|---|---|---|
| Regulation (EU) 2019/943 | [32019R0943](https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32019R0943) | [Reg. (EU) 2022/869](https://eur-lex.europa.eu/eli/reg/2022/869/oj/eng) (2022-06-23, Art. 48(1)); [Reg. (EU) 2024/1747](https://eur-lex.europa.eu/eli/reg/2024/1747/oj/eng) (2024-07-16, 45 amendments to 20 provisions incl. Arts. 1, 2, 7, 7a, 7b, 8, 9, 18, 19, 21, 22, 37) | 2019-06-14, 2022-06-23, 2024-07-16 | `02019R0943-20240716` |
| Directive (EU) 2019/944 | [32019L0944](https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32019L0944) | [Reg. (EU) 2022/869](https://eur-lex.europa.eu/eli/reg/2022/869/oj/eng) (2022-06-23, Art. 59); [Dir. (EU) 2024/1711](https://eur-lex.europa.eu/eli/dir/2024/1711/oj/eng) (2024-07-16, 29 amendments incl. Arts. 2, 4, 11, 27, 31, 33, 59, 66, 69) | 2019-06-14, 2022-06-23, 2024-07-16, 2025-10-12 | `02019L0944-20251012` |
| Regulation (EU) 2017/2195 | [32017R2195](https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32017R2195) | [Impl. Reg. (EU) 2021/280](https://eur-lex.europa.eu/eli/reg_impl/2021/280/oj/eng) (2021-03-15, Arts. 4 to 7); [Impl. Reg. (EU) 2022/828](https://eur-lex.europa.eu/eli/reg_impl/2022/828/oj/eng) (2022-06-19, Art. 12, Polish language version only) | 2021-03-15, 2022-06-19 | `02017R2195-20220619` |
| Regulation (EU) 2015/1222 | [32015R1222](https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32015R1222) | [Impl. Reg. (EU) 2021/280](https://eur-lex.europa.eu/eli/reg_impl/2021/280/oj/eng) (2021-03-15, Art. 9) | 2015-07-25, 2021-03-15 | `02015R1222-20210315` |
| Regulation (EU) 2017/1485 | [32017R1485](https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32017R1485) | [Impl. Reg. (EU) 2021/280](https://eur-lex.europa.eu/eli/reg_impl/2021/280/oj/eng) (2021-03-15, Arts. 5 to 8) | 2021-03-15 | `02017R1485-20210315` |
| Regulation (EU) 2016/1719 | [32016R1719](https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32016R1719) | [Impl. Reg. (EU) 2021/280](https://eur-lex.europa.eu/eli/reg_impl/2021/280/oj/eng) (2021-03-15, Art. 4) | 2016-09-27, 2021-03-15 | `02016R1719-20210315` |

Notes:

- Commission Implementing Regulation (EU) 2021/280 ("in order to align them with Regulation (EU) 2019/943") amends four of the six acts: 2015/1222, 2016/1719, 2017/2195 and 2017/1485. The issue named only the two 2024 acts; the other four were checked here and all four turn out to be amended.
- The 2019/944 consolidated version of 2025-10-12 differs from the 2024-07-16 one because Directive (EU) 2023/1791 partially repeals Art. 70 (EUR-Lex relation "Repealed by"). Art. 70 is not among the 94 articles, so nothing in the corpus changes because of it.
- The amended provisions that are kept in the corpus match the diff in section 2 exactly: the 12 articles that differ are the kept articles that EUR-Lex lists as touched (2019/943: Arts. 2, 7, 8, 9, 18, 19, 21, 22, 37; 2019/944: Arts. 31, 33; 2017/2195: Art. 4). The kept articles of 2015/1222, 2017/1485 and 2016/1719 are not among the provisions amended by 2021/280 (Arts. 9, 5 to 8 and 4 respectively), and the 2022/869 amendments (2019/943 Art. 48, 2019/944 Art. 59) are in articles that are not kept.
- New articles inserted by the amendments (for example Arts. 7a, 7b, 69a of Regulation 2019/943 and Arts. 15a, 18a, 66a of Directive 2019/944) are **not** added: the consolidated corpus keeps the same 94 article numbers, so it can be compared article by article with the original.

`SOURCES.json` of both corpora carries this as `amended_by` (celex, title, URL) per act; `corpus_eu_consolidated/SOURCES.json` also has `consolidated_celex`, `consolidated_date` and a consolidated `url`. Rebuild: `uv run ragkit fetch-eu --consolidated --retrieved-on 2026-11-13`. The amendment lists and consolidated CELEX numbers are pinned in `src/ragkit/eurlex.py`.

## 2. Articles that differ (original vs consolidated)

Method: both corpora were built with the same article selection; for each of the 94 article files the text after the metadata comment (heading included) was whitespace-normalised and compared. 12 articles differ, 82 are identical. The consolidated texts come from a different XHTML layout than the Official Journal, so the identical 82 are also a check that the new parser reproduces the original parser's text. Descriptions are from a word-level diff.

| Article | What changed |
|---|---|
| `reg_2019_943_art2` (definitions) | (22) 'capacity mechanism' is no longer "a temporary measure"; new definitions (72) peak hour, (73) peak shaving, (74) peak-shaving product, (75) regional virtual hub, (76) two-way contract for difference, (77) power purchase agreement, (78) dedicated measurement device, (79) flexibility |
| `reg_2019_943_art7` (day-ahead and intraday markets) | Adds subjection to transparency and anti-manipulation supervision (Regulation (EU) No 1227/2011); market rules also cover intra-zonal trade; new point (ca): liquidity sharing between all NEMOs, with day-ahead and intraday orders submitted to single coupling; adds information by generation unit |
| `reg_2019_943_art8` (trade on day-ahead and intraday markets) | From 1 January 2026 intraday cross-zonal gate closure at most 30 minutes before real time; new paragraphs 1a to 1c (derogations until 1 January 2029, a further one of up to 2.5 years, Commission report by 1 December 2027); **minimum bid size 500 kW → 100 kW**, "including through aggregation"; paragraph 4 (imbalance settlement period) unchanged |
| `reg_2019_943_art9` (forward markets) | Rewritten: long-term transmission rights allocated on a regular basis, frequency and maturities must support forward markets; Commission impact assessment by 17 January 2026 and implementing act by 17 July 2026; regional virtual hubs; legal form of the single allocation platform; regulators may require market-making; hedging products by market operators |
| `reg_2019_943_art18` (network charges) | Paragraph 2 gains a lettered list of tariff-methodology aims (capital and operational expenditure, anticipatory investment, flexibility and flexible connections, storage and demand response, NECP objectives, interconnection target); paragraph 3 now speaks of "locational investment signals" with examples; paragraph 8 "may" → "shall, where appropriate"; ACER report items (f) and (i) |
| `reg_2019_943_art19` (use of congestion income) | Paragraph 2 gains priority objective (c): compensating offshore renewable generators in an offshore bidding zone whose access to interconnected markets was reduced, with the compensation rules; paragraphs 3 to 5 unchanged |
| `reg_2019_943_art21` (capacity mechanisms) | Paragraph 1 loses "to eliminate residual resource adequacy concerns … as a last resort"; paragraph 7 (phase-out provision after three years without new contracts) is deleted; paragraph 8 loses "Capacity mechanisms shall be temporary" and the phase-out sentence, keeps the 10-year approval limit |
| `reg_2019_943_art22` (design principles of capacity mechanisms) | Paragraph 1(a) "be temporary" deleted; the CO2 limits in paragraph 4 are unchanged |
| `reg_2019_943_art37` (tasks of regional coordination centres) | Point (a) now refers to the forward capacity allocation, capacity allocation and congestion management and electricity balancing guidelines by number; points (j) and (k) unchanged |
| `dir_2019_944_art31` (tasks of DSOs) | Paragraph 2 names energy communities; paragraph 3 adds publication of available connection capacity (quarterly update), a three-month duty to report the status of connection requests, new 3a (digital requests) and 3b (100 000-customer exemption); paragraphs 4 to 10 unchanged and numbered as before |
| `dir_2019_944_art33` (recharging points) | Paragraph 1 adds smart and bidirectional charging functionalities (Art. 20a of Directive (EU) 2018/2001); the derogation conditions of paragraph 3 are unchanged |
| `reg_2017_2195_art4` (approval of terms and methodologies) | Rewritten by Implementing Regulation 2021/280: procedures by Art. 5(2) to (4), deadline prolongation, majority voting instead of "consensus", ACER role, "ENTSO-E" → "the ENTSO for Electricity" |

Identical in both texts (82): `reg_2019_943` Arts. 3, 5, 6, 10 to 17, 34, 35, 42, 49; `dir_2019_944` Arts. 15, 16, 17, 32, 36, 40, 54; `reg_2017_2195` Arts. 2, 3, 16 to 25, 29, 30, 32 to 35, 38 to 42, 52 to 55 (all kept articles except 4); `reg_2015_1222` all 15 kept articles; `reg_2017_1485` all 14; `reg_2016_1719` all 4.

## 3. Question review

Scope: every question of `qa_eu.jsonl` (42), `qa_eu_hard.jsonl` (39), `qa_eu_nokw.jsonl` (49), `qa_eu_fresh.jsonl` (29) whose gold article is one of the 12 changed articles (17 questions), plus the 45 questions of `unanswerable_eu.jsonl`. Beyond the gold check, a keyword scan of all other answerable questions against the changed text found no question whose answer a changed article now also supplies; the 11 questions that list a changed article only as a decoy (`qa_eu_hard` 13, 20, 22, 26, 28, 37, 38; `qa_eu_nokw` 21, 22, 40, 44) were re-read against the new text and the decoys still do not answer them. I (an agent) read the new text and decided; no human has reviewed these decisions. Index `#n` is the 0-based line of the file.

### 3.1 Answerable questions with a changed gold article (17)

| Question | Gold | Decision | Reason |
|---|---|---|---|
| `qa_eu#17` money from scarce capacity | art19 | **Reference answer changed** | Art. 19(2) now has three priority objectives; the answer said two. Gold article stays |
| `qa_eu#28` locational signals in fees | art18 | **Reference answer changed** (quotation) | Art. 18(3) now reads "locational investment signals … such as incentives via tariff structure to reduce re-dispatching and power grid reinforcement costs"; the old verbatim quote no longer matches. Substance ("only where appropriate", losses, congestion, investment cost) holds |
| `qa_eu#35` RCC procurement of balancing capacity, last-resort action | art37, art16 | Holds | Art. 37(1)(k) and Art. 16(3) unchanged; only 37(1)(a) changed |
| `qa_eu#41` RCC sizing of reserve capacity, FRR bound | art37, 2017/1485 art157 | Holds | Art. 37(1)(j) unchanged; Art. 157 identical |
| `qa_eu_hard#14` date for the revenue-use methodology | art19 | Holds | Art. 19(4): 5 July 2020, six months for ACER, unchanged |
| `qa_eu_hard#23` neutral market facilitator for losses | dir art31 | Holds | Art. 31(5) unchanged and still number 5 (new paragraphs are 3a, 3b) |
| `qa_eu_hard#24` DSO non-frequency ancillary services | dir art31 | Holds | Art. 31(7) unchanged |
| `qa_eu_hard#32` derogations from the 15-minute settlement period | art8 | Holds | Art. 8(4) unchanged; new text is in paragraphs 1, 1a to 1c and 3 |
| `qa_eu_nokw#20` DSO energy for losses | dir art31 | Holds | Art. 31(5) unchanged |
| `qa_eu_nokw#23` priority dispatch of renewables/CHP | dir art31 | Holds | Art. 31(4) unchanged |
| `qa_eu_nokw#24` DSO non-frequency services skip | dir art31 | Holds | Art. 31(7) unchanged |
| `qa_eu_nokw#41` "Which **two** aims take precedence" for the takings | art19 | **Question and answer changed** | Art. 19(2) lists three objectives, so "two" no longer matches; reworded to "Which aims take precedence …", answer lists points (a) to (c). The set is keyword-free: the reworded question still shares no content term with the article beyond what it had |
| `qa_eu_nokw#42` date for the use-of-income report | art19 | Holds | Art. 19(5): 1 March each year, unchanged |
| `qa_eu_nokw#43` derogation end for the 15-minute period | art8 | Holds | Art. 8(4) unchanged |
| `qa_eu_fresh#8` DSO charging points | dir art33 | Holds | Art. 33(3)(a) to (c) unchanged (the change is in paragraph 1) |
| `qa_eu_fresh#10` 1 March report | art19 | Holds | Art. 19(5) unchanged |
| `qa_eu_fresh#27` CO2 limit of capacity mechanisms | art22 | Holds | Art. 22(4)(a) and (b) unchanged (the deleted point is 1(a), "be temporary") |

### 3.2 Unanswerable questions (45)

All 45 were checked against the new text of the 12 changed articles (read in full where the question touches the topic). Decisions that are not "holds, unrelated":

| Question (`unanswerable_eu#n`) | Decision | Reason |
|---|---|---|
| `#3` two-sided contracts for difference under the 2024 reform | **Label changed: now answerable** | Consolidated Art. 2(76) defines "two-way contract for difference" as one that "provides both minimum remuneration protection and a limit to excess remuneration", which answers the question; the old reason ("the 2024 reform is not included") is no longer true. Moved to the answerable file as a `paraphrase` question with gold `reg_2019_943_art2` |
| `#8` FCR minimum bid size | Label holds, **reason changed** | Reason cited Art. 8(3) "500 kW or less"; consolidated text says 100 kW. FCR bid sizes are still not stated |
| `#33` minimum balancing capacity for prequalification | Label holds, **reason changed** | Same 500 kW → 100 kW figure in the reason. No minimum prequalification volume is stated |
| `#22` DSO exempt from legal unbundling below N customers | Holds (reason not changed) | New Art. 31(3b) has a 100 000-customer threshold, but for the connection-information duty, not unbundling; reason (Art. 32(5), unbundling not in corpus) is still true |
| `#27` maximum duration of long-term transmission rights | Holds (reason not changed) | New Art. 9(4)(b) says the Commission shall assess maturities "extended up to at least three years", a possible change and not the current maximum; no duration is stated |
| `#39` waiting time for large consumers' connection (NL, ACM) | Holds | New Art. 31(3) requires a status update within three months of a connection request, which is not a national waiting time |
| `#13` REMIT record keeping | Holds | New Art. 7(1) cites Regulation (EU) No 1227/2011 for supervision, still no record-keeping period |
| `#23` ERAA horizon | Holds | Art. 21(8) "10 years" is still the approval limit of a capacity mechanism |
| `#4` hours of interruption on non-firm connection | Holds | New Art. 18(2)(c) and Art. 31(3) mention flexible connections but give no hour limit |
| `#41` Poland capacity mechanism | Holds | Arts. 21 and 22 still set EU design principles only; no national mechanism or auction result |

The other 36 are unaffected: no changed article mentions their subject, a national authority or figure, or a number they ask for.

### 3.3 Files for the consolidated preset

`--corpus eu-consolidated` uses these files; every changed record carries a `consolidation_note` field that says why:

- `qa_eu_consolidated.jsonl`: `qa_eu.jsonl` with the answers of #17 and #28 updated and the moved question appended (43 questions: 17 direct, 17 + 1 paraphrase, 8 multi-hop).
- `unanswerable_eu_consolidated.jsonl`: `unanswerable_eu.jsonl` without `#3`, with the reasons of `#8` and `#33` updated (44 questions).
- `qa_eu_nokw_consolidated.jsonl`: `qa_eu_nokw.jsonl` with `#41` reworded. It is not part of the preset; use it as `--qa-file` with `--corpus eu-consolidated`.
- `qa_eu_hard.jsonl` and `qa_eu_fresh.jsonl` need no change and work on both corpora.

## 4. Scores on both texts

Commands (offline, deterministic, no model download):

```bash
uv run ragkit eval --corpus eu
uv run ragkit eval --corpus eu-consolidated --qa-file src/ragkit/data/qa_eu.jsonl     # consolidated text, original questions
uv run ragkit eval --corpus eu-consolidated                                              # consolidated text, re-reviewed questions
uv run ragkit abstention --corpus eu                                                     # same three variants for abstention (min_support 0.50)
uv run ragkit abstention --corpus eu-consolidated --qa-file src/ragkit/data/qa_eu.jsonl --unanswerable-file src/ragkit/data/unanswerable_eu.jsonl
uv run ragkit abstention --corpus eu-consolidated
```

Retrieval, no rewrite, recursive chunks of 500 characters, Recall@5 / MRR (point estimates; 95% intervals in the command output are about ±0.10 wide and overlap between all three columns):

| Strategy | Original text, 42 questions | Consolidated text, same 42 questions | Consolidated text, re-reviewed 43 questions |
|---|---|---|---|
| BM25 | 0.857 / 0.676 | 0.845 / 0.701 | 0.826 / 0.684 |
| Dense (hashing) | 0.619 / 0.528 | 0.619 / 0.528 | 0.605 / 0.519 |
| Hybrid (RRF) | 0.786 / 0.636 | 0.833 / 0.643 | 0.814 / 0.629 |
| Hybrid + Rerank | 0.821 / 0.591 | 0.786 / 0.605 | 0.767 / 0.591 |

Abstention at `min_support` 0.50:

| Set | Original text | Consolidated, original questions | Consolidated, re-reviewed |
|---|---|---|---|
| answerable answered | 20/42 | 20/42 | 20/43 |
| unanswerable abstained | 30/45 | 30/45 | 29/44 |
| 5th percentile best-chunk support (answerable) | 0.250 | 0.216 | 0.218 |

Reading: the differences are one to a few questions (one answerable question moves Recall@5 by 0.024), the intervals overlap, and no paired test was run, so none of the movements is established as a real effect of the amendments. What the amendments do change is the text a user sees: the consolidated text differs in 12 of 94 articles and in six question records (five answers, questions or reasons changed, one label moved). Not re-run: the neural, ctxprefix, generation and `--signals all` numbers in the README stay on the original text (the generation numbers come from an LLM run on the original articles).
