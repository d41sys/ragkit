# Context-prefix protocol (frozen before scoring)

Question: does prepending `<act title> — Article N: <article heading>` to **every** chunk, in both
the BM25 text and the hashing-dense text, fix the wrong-document failure across near-identical
articles? This is "contextual retrieval, lite": Anthropic's contextual retrieval and arXiv
2510.06999 prepend a context string to each chunk before indexing, but here the context is the
article's own heading, so it needs **no model**. Everything is numpy and the standard library; the
eval path builds `BM25Index`, `HashingEmbedder`, `DenseIndex` and `HybridRetriever` only.

This file and the fresh question set are committed before the first prefix variant is scored on
any question. `git log` shows the order: fresh questions, review, this protocol, then code, then
the scored run. Numbers are reported whatever they turn out to be.

## What was already known (disclosed taint)

- The 42 questions of `qa_eu.jsonl` were used in earlier diagnosis (`docs/lexical_protocol.md`,
  "Why it did not help"). They are **tainted** for this decision and are reported as secondary,
  never gating. The baseline numbers on them are already published (BM25 0.857 / 0.676 / 0.702).
- The author of this protocol had seen the first 90 characters of every old question and its
  relevant ids (while choosing which near-duplicate clusters to cover), and avoided writing
  fresh questions on the same paragraphs. No retriever was run on the fresh questions.
- Earlier result A2 (`+structure`, heading prepended to structured chunks, plain BM25) showed no
  separated gain on the old 42 questions. That was a different chunker, so it does not settle
  this question, but it is a prior against a large effect.
- The heading of an article already sits at the start of its **first** chunk in the baseline;
  the prefix adds it to the other chunks of the article.

## Fresh question set (the primary evidence)

`src/ragkit/data/qa_eu_fresh.jsonl`: 29 answerable questions on the same 94-article EU corpus,
each with `provenance` (`doc_id#paragraph`) and the answer text.

- Drafted by an AI agent after reading the cited articles, **before any prefix variant was run**,
  and committed with `status: draft-agent` (commit `2662cea`). A separate agent then reviewed
  every item adversarially (reading cited articles, grepping the corpus for other articles that
  independently answer the question, rewriting ambiguous items) without running any retriever;
  report in `docs/qa_eu_fresh_review.md`. **No human reviewed them**, and drafter and reviewer
  are probably the same model family, so shared errors can survive.
- Aimed at the failure the prefix targets: 14 `near-duplicate` (sibling articles with
  near-identical wording inside one act, e.g. the four European platform articles 19/20/21/22 of
  2017/2195, DSO vs TSO storage ownership), 8 `cross-act` (same topic in another act: congestion
  income, firmness, bidding zone review), 7 `paraphrase`. Several include a paraphrase of the
  article wording.
- **Known bias in favour of the prefix.** The review rewrote two questions (items 11 and 24 of
  the file) to name their act (`2015/1222`, `Regulation 2017/2195`) so that sibling acts could not
  also answer them. Act numbers occur in the first chunk of an article (its heading) but not in
  the other chunks of the baseline, so these two questions can only help the prefix. A
  non-gating sensitivity row drops every question whose text contains an act number
  (`\d{4}/\d{1,4}`; exactly these two) and is reported next to the full-set result.
- Frozen at the commit that contains the reviewed file and this protocol. If a later reviewer or
  human finds an error, the scored numbers stay as committed and the correction is reported next
  to them.

## Variants (all frozen)

Documents are the corpus files with the metadata comment stripped (`_load_documents`). Chunks are
the existing `recursive` chunks (500 characters, 50 overlap), exactly as in `ragkit eval` and
lexical A0. Ranking unit is the document (article): top 10 **chunks**, deduplicated to
documents by first occurrence; metrics use the resulting list.

| id | chunk text |
|---|---|
| `base` | the recursive chunk, unchanged |
| `prefix` | `<prefix>` + newline + the recursive chunk, with a leading `# heading` line removed first so the first chunk does not carry the heading twice |

`<prefix>` is built from the article's leading `# ` heading line. If it matches
`<act title>, Article <N>: <article heading>` it becomes `<act title> — Article <N>: <article
heading>`; any other heading is used as is; a document without a heading gets no prefix. Same
chunk boundaries, same `start`/`end`; only `Chunk.text` changes. Queries are never modified.

| retriever | definition |
|---|---|
| `bm25` | `BM25Index()` (k1 1.5, b 0.75), top 10 chunks |
| `hybrid` | `HybridRetriever(bm25=BM25Index(), dense=DenseIndex(HashingEmbedder(dim=512)))`, embedder fitted on the (variant's) chunk texts, RRF (k = 60), top 10 chunks. Same construction as the `hybrid` row of `ragkit eval` |

No reranker, no STEmbedder, no cross-encoder; `ragkit ctxprefix` is pinned to these constructors
and a subprocess test checks that `sentence_transformers`, `torch` and `transformers` are never
imported.

## Metrics

Per question: Recall@5, MRR, nDCG@10 of the deduplicated document list (`ragkit.eval`,
`relevant_doc_ids`). Mean with a 95% percentile-bootstrap CI over questions (2000 resamples,
seed 0); paired differences (`prefix` minus `base`, same questions) with the paired bootstrap CI
of the per-question difference, and per-question better / worse / tied counts.

**Wrong-act count.** The top-1 document's act is its id without the `_art<N>` suffix
(`reg_2019_943_art16` -> `reg_2019_943`; for the synthetic corpus every document is its own
"act", so there it is a wrong-document count). A question is a wrong-act error when the top-1
document's act is not the act of any relevant document, or when nothing is retrieved. Reported
as counts with how many questions flipped wrong -> right and right -> wrong. Diagnostic; it does
not gate.

## Datasets

1. **fresh** (primary): `qa_eu_fresh.jsonl`, 94-article EU corpus.
2. **old42** (secondary, tainted): `qa_eu.jsonl`, same corpus. Reported; never gates.
3. **synthetic** (guard): `qa.jsonl`, 25-document synthetic corpus, 40 questions.
4. **fresh, act-named questions removed** (sensitivity): the fresh set without the questions
   whose text contains an act number. Reported; never gates.

## Decision rule (per retriever, `bm25` and `hybrid` each judged on its own)

The prefix is **adopted** for a retriever iff both hold:

1. **Gain on fresh:** the paired MRR difference (`prefix` - `base`) on the fresh set has a 95% CI
   whose lower bound is above 0.
2. **No regression on synthetic:** the paired MRR difference on synthetic is at least -0.02
   **and** its 95% CI upper bound is not below 0.

Otherwise: **rejected** when the fresh MRR CI lies entirely below 0 or the synthetic check
fails; **not adopted (no evidence)** when the fresh CI includes 0. The old-42 numbers can add a
caveat to the verdict but cannot change it. The two retrievers are two tests with no
multiplicity correction; a lone pass for one retriever is reported as such. With 29 questions one
question moves MRR by at most 1/29 = 0.034, so only a gain that shows up on several questions can
clear rule 1. **Wording:** "not adopted (no evidence)" means no detectable effect on that set. It
never means "the prefix does not help": a set at its ceiling cannot show a gain.

## Not tried (would need a fresh protocol)

A shuffled-prefix control (a prefix from a different article, to separate "right metadata" from
"extra tokens and length"), prefixes on structured chunks, a prefix only in one of BM25 or dense,
a prefix weighted differently from the body, model-generated chunk context (the real contextual
retrieval), and any tuning of the prefix format.

## Outcome and follow-up

The first scored run (`runs/ctxprefix_protocol.txt`) found no detectable effect on the fresh set
(BM25 dMRR +0.032 [-0.017, +0.095], hybrid +0.000 [-0.072, +0.072]); that set was close to its
ceiling (BM25 MRR 0.891, Recall@5 0.948, 3 of 29 wrong-act top-1), so it could not show a gain.
The set was not edited afterwards. A second, harder set with a saturation guard and the same
decision rule was pre-registered in `docs/ctxprefix_hard_protocol.md` (set review in
`docs/qa_eu_hard_review.md`). Its result, `runs/ctxprefix_hard.txt`: the hard set was also
saturated for BM25 (Recall@5 0.897 > 0.85), and neither retriever was adopted (BM25 dMRR -0.009
[-0.039, +0.012], hybrid +0.026 [-0.030, +0.085]). No default changed.

A third and last set, written without the gold article's discriminating vocabulary by a mechanical
overlap rule, was pre-registered in `docs/ctxprefix_nokw_protocol.md`. It was a valid test (BM25
Recall@5 0.347, between the saturation guard 0.85 and the floor guard 0.15): BM25 dMRR -0.022
[-0.056, +0.005], hybrid +0.010 [-0.056, +0.069]. Neither retriever was adopted; no default changed
(`runs/ctxprefix_nokw.txt`).
