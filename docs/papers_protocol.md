# Research papers as a corpus (P-1), frozen before any code or run

Problem. ragkit's main users are researchers who ask about the group's papers and builders whose
ezcross agent reads papers to review literature. Every study so far used short documents (EU
articles, at most 13 253 characters). Papers are PDFs of 60 000 to 160 000 characters, and the
shipped packing returns whole documents: a paper over the 12 000-token budget is the only document
returned (cut to a window around its best chunk since `e3b710a`), so one search can show one paper.
This study adds PDF loading and paper citations, and tests two changes on a stand-in corpus of
published forecasting papers until the group's own papers exist.

## Corpus (fixed before any run)

The 20 papers below: time-series forecasting, top AI venues 2020 to 2024, open access on arXiv.
PDFs are fetched by `ragkit papers fetch` into `data/raw/papers/forecasting/` (gitignored: the arXiv
licence does not allow redistribution); only the list and the questions are committed. Text is
extracted with `pypdf` (pages joined by a form feed, so citations can name pages); the whole PDF is
indexed, references and appendix included, as a user upload would be. Measured before this
protocol: 1 857 936 characters, 15 to 38 pages per paper; no question existed then.

| arXiv | short name | venue |
|---|---|---|
| 1905.10437 | N-BEATS | ICLR 2020 |
| 2012.07436 | Informer | AAAI 2021 |
| 2106.13008 | Autoformer | NeurIPS 2021 |
| 2201.12740 | FEDformer | ICML 2022 |
| 2205.14415 | Non-stationary Transformers | NeurIPS 2022 |
| 2205.13504 | DLinear (LTSF-Linear) | AAAI 2023 |
| 2211.14730 | PatchTST | ICLR 2023 |
| 2210.02186 | TimesNet | ICLR 2023 |
| 2201.12886 | N-HiTS | AAAI 2023 |
| 2302.11939 | One Fits All (GPT4TS) | NeurIPS 2023 |
| 2310.06625 | iTransformer | ICLR 2024 |
| 2405.14616 | TimeMixer | ICLR 2024 |
| 2310.01728 | Time-LLM | ICLR 2024 |
| 2307.03756 | FITS | ICLR 2024 |
| 2310.10688 | TimesFM | ICML 2024 |
| 2402.02592 | Moirai | ICML 2024 |
| 2402.02368 | Timer | ICML 2024 |
| 2402.03885 | MOMENT | ICML 2024 |
| 2405.00946 | SparseTSF | ICML 2024 |
| 2402.19072 | TimeXer | NeurIPS 2024 |

## Questions (written after this commit, before any retrieval run)

50 questions in `src/ragkit/data/qa_papers_forecasting.jsonl`, written by Claude subagents that see
only the extracted paper text (never a retrieval result), checked by a second subagent pass, no
human review (as the EU sets). Each gold item is a paper id plus a verbatim evidence quote of 10 to
60 words that answers the question and is found in the extracted text after whitespace
normalisation (`ragkit papers validate` checks it before the run).

- **single** (40): two per paper, one fact from the body (not the abstract, not the references).
  One of the two names the method (`named`), the other describes it without the method or paper
  name (`described`), as a researcher who does not know the name asks.
- **cross** (10): literature-review questions whose answer spans 2 to 4 papers (for example which
  models use channel independence), one evidence quote per gold paper. Gold sets can be incomplete;
  recall is measured on the listed papers only.

## Systems

All are the shipped `hybrid+ce@100` over recursive 500 / 50 chunks of the 20 papers (BM25 and
bge-small top 100, RRF k = 60, MiniLM reranks the top 100), pool of the top 10 chunks, budget
12 000 estimated tokens, overflow `stop` (the shipped `search()` settings).

| id | packed unit |
|---|---|
| B0 (shipped) | whole document; a first document over the budget is cut to the budget-sized window around its best chunk and packing ends |
| W | window of at most 4 000 tokens: a document up to 4 000 tokens whole, a longer one as the 4 000-token window around its best chunk; packing continues, so up to 3 papers fit |
| WC | W, with a Qwen context prepended to every chunk for BM25, the embedder and the cross-encoder (the returned text is the paper text, not the context) |

4 000 tokens is fixed, not tuned: it is above the longest EU article (13 253 characters, 3 314
estimated tokens), so W returns exactly what B0 returns on the EU corpus. Contexts as K-R7
(Anthropic's prompt, `qwen3.6:27b-ctx4k`, temperature 0, at most 200 output tokens, thinking off),
with the document cut to its first 96 000 characters (24 000 estimated tokens) and `num_ctx`
32 768; 6 of 20 papers are longer (up to 161 863 characters) and are cut. Cached in
`docs/sota-runs/papers_contexts.jsonl` (committed).

## Metrics

- **Evidence recall** (primary): per question, the share of gold quotes contained in the packed
  context (lowercase, whitespace collapsed); pooled mean over the 50 questions.
- Paper recall (share of gold papers with any packed text), MRR of the first gold paper in the
  chunk ranking, packed tokens, distinct papers packed. Per type (named, described, cross).

## Decision rule

Paired bootstrap over questions (10 000 resamples, seed 0, 95 % CI) of evidence recall.

1. **W** becomes the `search()` default iff W - B0 has CI lower bound > 0 and the EU guard holds:
   on the 190 pooled EU questions, W's packed article ids equal B0's for every question.
2. **WC** (contexts for paper corpora, generated at upload) is adopted iff WC - W has CI lower bound
   > 0. It is judged only if W is adopted; otherwise WC - B0 is reported, not judged.

## Predictions (written before any run)

- P1: W gains mostly on cross questions (more papers fit); single questions change little.
- P2: WC - W is +0.05 to +0.15 (K-R7 on long documents: +0.112 over larger chunks alone).
- P3: described questions are harder than named ones for every system.
- P-cost: about 4 100 chunks, so about 4 100 context calls, 4 to 6 hours on Korea.

## Not claimed

The group's own papers (not available yet); answer quality through the agent; other prompts or
LLMs; PDFs with scanned pages (no OCR).

## Amendment 1 (before any question or context exists)

The raw call cache stores the whole request, so for papers it would be about 370 MB (4 100 calls
times up to 96 000 characters), above GitHub's file limit. It is kept in
`data/raw/papers/contexts_cache.jsonl` (gitignored); `ragkit papers contextualize` writes the
generated contexts with the model, `num_ctx`, document cut and prompt to
`docs/sota-runs/papers_contexts.json` (committed), and `ragkit papers score` reads that file. The
paper list lives in `src/ragkit/data/papers_forecasting.json`; `fetch` copies it as `SOURCES.json`
next to the PDFs, which gives search results their citation (`Article.citation`, `Article.pages`).

## Amendment 2 (before any context exists; the questions exist, no retrieval run has happened)

Contexts are generated with **`qwen3:8b`** (already on the Korea server, about 5 GB), not
`qwen3.6:27b-ctx4k`. Reason: the 27B model at `num_ctx` 32 768 needs about 27 GB of GPU memory, the
shared GPUs had 17 to 22 GB free across four cards, and the first load returned HTTP 500 with no
context generated. A small model matches Anthropic's published setup (Claude Haiku writes the
contexts). Everything else is unchanged: prompt, temperature 0, at most 200 output tokens, thinking
off, document cut at 96 000 characters, `num_ctx` 32 768. P-cost becomes about 2 hours.

## Amendment 3 (after 14 context calls, before any retrieval run or score)

The 96 000-character cut assumed 4 characters per token; the paper text has about 2.7 (math,
tables, numbers). The first 14 calls sent prompts of 35 000 tokens, which Ollama silently cut from
the start to `num_ctx` 32 768 (log: "truncating input prompt"), so the model lost the beginning of
the paper and no prompt prefix could be reused. The document cut becomes **80 000 characters**
(about 29 000 tokens), so document, chunk, instructions and the 200-token reply fit in 32 768. The
14 calls are discarded (they stay in the gitignored raw cache under other keys); generation restarts.
Since amendment 2 the contexts run on a separate Ollama server limited to one GPU
(`CUDA_VISIBLE_DEVICES=1`, own port), because the shared server fails on the full GPU 3; the shared
server was not restarted.

## Amendment 4 (after 69 context calls at 80 000, before any retrieval run or score)

Amendment 3 measured characters per token on the first paper only. A probe (each paper's text
cut, plus its longest chunk, sent once with one output token; Ollama's `prompt_eval_count`) found
that at 80 000 characters four papers (2205.14415, 2211.14730, 2302.11939, 2405.14616) exceed 32 768
tokens (up to 37 903, about 2.1 characters per token) and are cut from the start again. The same
probe at **60 000 characters** gives at most 24 491 tokens (2307.03756) over all 20 papers, which
leaves room for the 200-token reply. The cut becomes 60 000 characters for every paper; the 69 calls
at 80 000 are discarded (gitignored raw cache, other keys) and generation restarts.
