# P-1 question set: how it was written (`src/ragkit/data/qa_papers_forecasting.jsonl`)

Written after the protocol (`docs/papers_protocol.md`) and before any retrieval run on the papers.

1. Six Claude subagents drafted the questions from the page-marked text dump
   (`ragkit papers dump`): five wrote one `named` and one `described` question per paper for four
   papers each, one wrote the 10 `cross` questions with grep over all 20 texts. None saw a
   retrieval result or ragkit's retrieval code.
2. A seventh subagent reviewed all 50 against the protocol and made 13 changes, then
   `ragkit papers validate` passed (every quote verbatim on one page, 10 to 60 words).
   - Evidence that described another model, or did not fully answer, was replaced or the question
     narrowed: `2012.07436-named`, `2201.12740-named`, `2201.12740-described` (quote extended),
     `2310.10688-described`, `2405.00946-named`, `2405.00946-described` (quote covers both
     scenarios).
   - Title words removed from described questions: `2012.07436-described`, `2106.13008-described`.
   - Method named explicitly: `2302.11939-named`.
   - Cross: `cross-01` and `cross-06` PatchTST quotes now show PatchTST itself; `cross-04` TimesFM
     quote is the body result, not the introduction; `cross-02` drops TimesFM (it normalises but
     does not reverse), leaving 3 gold papers.
3. Known limits: cross gold sets are incomplete by design (for example `cross-08` omits FEDformer,
   `cross-04` omits Timer and MOMENT), so a system can pack a correct paper that is not counted.
   Quotes keep the extraction's hyphenation and ligatures (`ﬁ`). No human review.

Counts: 20 named, 20 described, 10 cross (4 with 2 papers, 2 with 3, 4 with 4).
