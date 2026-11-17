# Contextual retrieval protocol (K-R7): the published industry recipe on ragkit (frozen before any code or run)

Problem. K-R5 and K-R6 swapped one small open model at a time and changed little. ragkit already
has the core of the published industry pipeline: BM25 and dense search fused by RRF, then a
cross-encoder. Two parts of that recipe are missing:

- **Chunk size.** ragkit's chunks are 500 characters (mean 385, about 95 tokens). Published
  defaults: OpenAI file search 800 tokens with 400 overlap
  (<https://developers.openai.com/api/docs/guides/retrieval>, "By default, `max_chunk_size_tokens`
  is set to `800` and `chunk_overlap_tokens` is set to `400`"); Azure AI Search's benchmark 512
  tokens with 25 % overlap
  (<https://techcommunity.microsoft.com/discussions/azure-ai-foundry-discussions/azure-cognitive-search-outperforming-vector-search-with-hybrid-retrieval-and-ran/3931019>);
  Vertex AI RAG Engine quickstart 512 tokens with 100 overlap
  (<https://docs.cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-quickstart>).
- **Contextual retrieval** (Anthropic, <https://www.anthropic.com/engineering/contextual-retrieval>):
  an LLM writes a short context that situates each chunk in its document; the context is prepended
  to the chunk before BM25 indexing and embedding. Reported: top-20 retrieval failures -35 %
  (contextual embeddings), -49 % (plus contextual BM25), -67 % (plus a reranker), averaged over
  codebases, fiction, ArXiv and science papers.

This protocol tests both, once, and decides whether either ships. It is committed before any code
of the study and before any context is generated. `git log` shows: this protocol, the code, the
context cache, the single scoring run, the documentation.

## Disclosed prior knowledge

- The candidates come from published industry practice, not from ragkit results.
- All 190 EU questions below were scored before (K-R3, K-R5, packing, K-R6) with the shipped
  chunks. None was used to choose or tune anything here; there is nothing to tune.
- LegalBench-RAG-mini was scored before with naive 250 / 500 / 1000-character chunks
  (`docs/legalsize_protocol.md`); the two sources used here were part of it. Bigger chunks raised
  recall there, mostly through more characters per chunk; the budget metric below removes that.
- K-R3 found article units (whole articles, far above 512 tokens) worse than chunks on EU MRR
  (-0.034 [-0.089, +0.016]).

## Why 1600 characters and not 800 tokens

ragkit's models read at most 512 model tokens (bge-small: `MAX_SEQ_LENGTH = 512`; MiniLM reads
query plus passage within 512). An 800-token chunk would be cut silently, and the context needs
room. The study therefore follows the 512-token, 25 %-overlap setting (Azure, close to Vertex):
**1600-character chunks (about 400 tokens) with 400 characters of overlap**, leaving about 100
tokens for the context. Same recursive splitter as now (`chunking.recursive`), only the size and
overlap change.

## Systems

All use the shipped `hybrid+ce@100` pipeline (ragkit BM25 top 100 and bge-small top 100, RRF
k = 60, MiniLM reranks the top 100), built as `ragkit.vocabbench.Engine` does.

| id | chunks | context |
|---|---|---|
| B0 (shipped) | recursive 500 / 50 | none |
| S | recursive 1600 / 400 | none |
| SC | recursive 1600 / 400 | Qwen-written context + `"\n\n"` + chunk text, used by BM25, the embedder and the cross-encoder |

**Context generation (SC).** Anthropic's published prompt, verbatim apart from trailing spaces,
as the single user message (no system text):

```
<document>
{WHOLE_DOCUMENT}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{CHUNK_CONTENT}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
```

The document is the corpus file the chunk comes from: one EU article (its heading names the act
and article), one LegalBench contract or privacy policy. Model and decoding: Qwen3.6 27B
(`qwen3.6:27b-ctx4k`, Q4_K_M) through Ollama on the Korea server, `num_ctx` 24576 (the longest
document, 54 158 characters, must fit), temperature 0, seed 0, at most 200 output tokens, thinking
off. Every call is cached in `docs/sota-runs/contextual_contexts.jsonl` (committed). An empty reply
leaves the chunk without context (count reported).

## Data (scored once)

| set | queries | corpus | metric |
|---|---|---|---|
| EU (pooled) | 190: `qa_eu` 42, `qa_eu_hard` 39, `qa_eu_nokw` 49, `qa_eu_confirm` 60 | `corpus_eu_consolidated`, 94 articles (served corpus) | MRR over distinct articles, wrong-act, pack recall (K-R5 / K-R6 definitions) |
| LegalBench-RAG-mini, `privacy_qa` + `contractnli` | 388 (194 + 194) | their 25 documents (7 privacy policies, 18 NDAs; 361 131 characters) | gold-character recall at a budget of 8000 retrieved characters |

**Budget recall** (because larger chunks return more text at the same k): walk the ranked chunks,
keep each chunk's character span, and stop when 8000 distinct characters are covered (the last
chunk is cut at the budget). Score = gold characters covered by the union of kept spans / all gold
characters of the query. Context text is never counted: it is not document text. Also printed at
4000 and 16000. MAUD and CUAD are left out: their 8.3 M characters would take about 7 000
context calls.

## Decision rule (S and SC judged separately)

X is **adopted** iff all hold:

1. **EU gain:** pooled 190 paired MRR difference X - B0, 95 % percentile bootstrap (2000
   resamples, seed 0), lower bound above 0.
2. **EU serving context:** pooled wrong-act count not above B0's, pack-recall difference point
   estimate at least 0.
3. **Other collection not hurt:** LegalBench pooled 388 paired budget-recall difference (8000) point
   estimate at least 0.

**Rejected** when the EU CI lies entirely below 0; otherwise **not adopted (no evidence)**.
If both are adopted, SC ships only if SC - S paired EU MRR has a CI lower bound above 0 (SC needs
an LLM at index time); otherwise S ships.

## What an adoption changes

- **S:** the neural retrievers of `ragkit.search` (`hybrid+ce`, `hybrid+bge-ce`) index 1600 / 400
  chunks; the prebuilt index is rebuilt; the `hybrid+ce` answerable threshold is re-derived by its
  existing rule (`abstention.best_threshold` on the 87 EU questions), since the reader reads the top
  5 chunks and they grow. The agent evaluations are not re-run (README says so).
- **SC:** as S, plus the EU contexts ship as a data file used at index time (no LLM at search
  time). Another corpus gets contexts only through an explicit index-build step with an Ollama
  server; without it, it is indexed as S. The README states that the SC evidence is EU plus two
  LegalBench sources.
- **Neither:** nothing changes.

## Also printed (not gating)

Per EU set: MRR, Recall@5, G@5 (prod), pack recall, wrong-act with paired differences; LegalBench
per source and at 4000 / 16000; chunk counts; context length (mean words) and empty contexts;
share of SC chunk texts over 512 bge-small tokens (cut by the embedder).

## Predictions (written before any run)

- P-S: S changes EU MRR by -0.02 to +0.03 (larger chunks blur the article signal, K-R3's article
  units lost); LegalBench budget recall +0 to +3 points. Not adopted on EU.
- P-SC: SC gains on EU, mostly by fewer wrong-act errors (the context names the act): pooled MRR
  +0.03 to +0.08 with a CI above 0; LegalBench +1 to +5 points. SC adopted.
- P-cost: about 700 context calls, 1 to 2 hours on one GPU.

## Not claimed

Answer quality through the agent; hosted embedders or rerankers (Voyage, Gemini, Cohere), which the
published results used; other prompts or LLMs for the context; MAUD and CUAD; the group's own
research documents (no labelled questions exist yet).
