# Confirmation protocol (K-R6): a larger reranker, with and without LLM expansion, on unseen data (frozen before any question, code or run)

Problem. K-R5 (`docs/vocab_protocol.md`) adopted nothing. Its non-gating table showed one cell with
a pooled MRR CI above 0 on the 130 EU held-out questions: bge-reranker-base as the cross-encoder
plus Qwen query2doc expansion, original question to the cross-encoder (+0.051 [+0.002, +0.101],
pack recall 117 / 130 against 111). That cell was found by looking at held-out data, so it is a
hypothesis. This protocol tests it, and the reranker alone, **once, on data no candidate has been
scored on**, and decides whether either ships.

ragkit is meant as general retrieval for the group, EU law being one corpus among others (the
user's statement, 2026-11-17), so a change must also hold on research-paper retrieval. This file is
committed before the new questions are written, before any code of the study and before any
expansion of the new data is generated. `git log` shows the order: this protocol, the questions
and their review, the code, the expansion cache, the single scoring run, then the documentation.

## Disclosed prior knowledge and one rule change

- The two candidates were chosen from K-R5 results, including the held-out 130 for C. None of the
  data below was used for that.
- **K-R5's synthetic guard is not a condition here**, and this change is made after seeing it
  fail: on the synthetic 40 (25 toy energy-flexibility documents) both candidates are already
  known to lose (MRR 0.958 against 0.988, -0.030; the guard allowed -0.020). The synthetic set is
  saturated (B0 at 0.988, pack recall 1.000) and is not a served corpus. Its role, "does the change
  break another kind of collection?", moves to two public research-paper benchmarks that are 15
  times larger (below). The synthetic result is still printed.
- SciFact and NFCorpus were scored before with bge-small and the MiniLM cross-encoder over a Lucene
  BM25 (`docs/neural_protocol.md`). Neither candidate has been run on them, and B0 here uses
  ragkit's own BM25 (the shipped one), so no number below exists yet.

## Data (scored once)

| set | queries | corpus | metric |
|---|---|---|---|
| EU-new (`qa_eu_confirm.jsonl`, written for this study) | 60 | `corpus_eu_consolidated`, the 94 articles `ragkit.search` serves | MRR, wrong-act, pack recall, as K-R5 |
| SciFact test (BEIR) | 300 | 5 183 scientific abstracts | nDCG@10 (trec_eval), Recall@100 |
| NFCorpus test (BEIR) | 323 | 3 633 medical / nutrition articles | nDCG@10, Recall@100 |
| synthetic (`qa.jsonl`) | 40 | 25 documents | MRR (printed, not gating) |

**EU-new questions.** One AI agent writes 60 questions from the consolidated articles: 30
*natural* (how a grid or market practitioner would ask; the article's own terms are allowed, but
not the act's name, its number or the article number) and 30 *keyword-free* (the mechanical rule of
`ragkit.nokw`: at most one of the gold article's top-20 TF-IDF stems, computed on the consolidated
corpus). Coverage: every one of the 6 acts has at least 5 questions; no article is gold for more
than 2; no question restates one of the 199 existing EU or synthetic questions (the writer reads
them to avoid that). Each item has the existing fields (`q`, `relevant_doc_ids`, `answer`, `type`,
`provenance`, `status`). A second, separate agent reviews every item against the whole corpus
(does the gold article support the answer; does another article answer it fully, then it is added
to `relevant_doc_ids`; is it ambiguous, then it is rewritten or dropped and replaced by the
writer). Neither agent runs any ragkit retriever or sees any system output. Both are AI agents
of one model family; no human reviews the set (the user's choice). The reviewed file is committed
before any code of the study.

## Systems

All rank units with the shipped `hybrid+ce@100` pipeline (ragkit BM25 top 100 and bge-small top
100, fused by RRF k = 60, cut to 100, a cross-encoder reranks all 100), as `ragkit.vocabbench`
does. EU units are the K-R3 chunks (recursive 500 / 50) of the consolidated corpus; a BEIR unit is a
whole document (`title + " " + text`, as `docs/neural_protocol.md`).

| id | cross-encoder | expansion |
|---|---|---|
| B0 (shipped) | `cross-encoder/ms-marco-MiniLM-L-6-v2@233902d` | off |
| R | `BAAI/bge-reranker-base@2cfc18c` | off |
| C | `BAAI/bge-reranker-base@2cfc18c` | on, original question to the cross-encoder |

**Expansion** is K-R5's, with the domain phrase as the only change per corpus. Prompt template
(frozen): "You write one short passage that answers the question as it would be worded in
{domain}. Use the formal terms such a text would use. Write only the passage, 80 to 120 words: no
preamble, no title, no citations, no mention that the passage is hypothetical." Domains:

- EU-new and synthetic: "EU electricity-market legislation or in a technical document on energy
  flexibility" (this reproduces the K-R5 prompt exactly, so the 40 synthetic passages of
  `vocab_expansions.jsonl` are reused, not regenerated);
- SciFact: "the abstract of a scientific research paper";
- NFCorpus: "a medical or nutrition research article".

Same model and decoding as K-R5 (Qwen3.6 27B, `qwen3.6:27b-ctx4k`, temperature 0, seed 0,
256 tokens, thinking off); the query text goes in verbatim. Calls are cached in
`docs/sota-runs/confirm_expansions.jsonl`. BM25 gets the query five times plus the passage, the
dense side `query + "\n" + passage`. An empty passage falls back to the query (count reported).
There is no tuning: nothing is chosen in this study except the verdict.

## Decision rule (per candidate; R and C are judged separately)

X is **adopted** iff all hold:

1. **Gain on research papers:** pooled SciFact + NFCorpus (623 queries) paired nDCG@10 difference
   X minus B0, 95 % percentile bootstrap (2000 resamples, seed 0), lower bound above 0.
2. **No collection hurt:** the paired difference point estimate is at least 0 on SciFact, on
   NFCorpus (nDCG@10) and on EU-new (MRR).
3. **EU serving context:** on EU-new, wrong-act count not above B0's and pack-recall difference
   point estimate at least 0.

EU-new alone (60 questions) resolves differences of about ±0.08 MRR, too coarse to demand a CI
above 0 there; its paired CI is printed. **Rejected** when the pooled BEIR CI lies entirely below 0;
otherwise **not adopted (no evidence)**. Two candidates, two tests, no multiplicity correction.

## What an adoption changes

- **R adopted:** bge-reranker-base becomes the cross-encoder of `retriever="hybrid+ce"` in
  `ragkit.search`, `ragkit answer` and `NeuralChunkRetriever`; the `hybrid+ce` answerable
  threshold is re-derived by its existing rule (`abstention.best_threshold` on the 87 EU
  questions, `_search.py`). First use downloads 1.1 GB instead of 88 MB. The agent evaluations are
  not re-run; their numbers stay those of MiniLM, and the README says so.
- **C adopted:** expansion needs an LLM at query time, so it ships **opt-in** only: an `expand`
  argument of `ragkit.search` (a callable from query to passage) that applies C's recipe, with
  bge-reranker-base as its cross-encoder. The default changes only through R.
- **Neither:** nothing changes, and the K-R5 lead is recorded as not confirmed.

## Also printed (not gating)

Per set and pooled: means with CIs, paired differences with better / worse / tied counts, Recall@100
(BEIR) and Recall@5, G@5 (prod) (EU-new); the synthetic MRR; expansion fallback counts; the mean
cross-encoder wall time per query of each reranker on this machine (Apple MPS), since R makes every
`hybrid+ce` search slower.

## Predictions (written before any run)

- P-R: R gains on SciFact and NFCorpus (bge-reranker-base is a stronger general reranker than
  MiniLM-L6 [published BEIR rerank tables report this; not reproduced here]); pooled BEIR gain
  +0.01 to +0.03 nDCG@10 with a CI above 0; EU-new MRR gain positive but its CI includes 0. R is
  adopted.
- P-C: expansion helps NFCorpus (short, lay queries) more than SciFact (claims already in paper
  wording); C beats R on EU-new; C is adopted only if its BEIR gain holds on both sets, which I
  rate as less likely than for R.
- P-time: bge-reranker-base costs 5 to 15 times MiniLM's cross-encoder time per query.

## Not claimed

Answer quality through the agent; other LLMs, prompts or decodings for expansion; collections
unlike EU law and biomedical abstracts. The EU-new questions are agent-written and agent-reviewed.
