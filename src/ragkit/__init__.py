"""RagKit: modular RAG from scratch with hybrid search, reranking, and evaluation."""

from __future__ import annotations

from ragkit.bm25 import BM25Index, tokenize
from ragkit.chunking import Chunk, fixed, recursive, semantic, sentence
from ragkit.dense import DenseIndex
from ragkit.embed import Embedder, HashingEmbedder, STEmbedder
from ragkit.eval import (
    answer_relevance,
    dcg_at_k,
    faithfulness,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ragkit.hybrid import HybridRetriever, rrf, weighted_fusion
from ragkit.pipeline import Answer, ExtractiveGenerator, Generator, RAGPipeline
from ragkit.rerank import CrossEncoderReranker, LexicalReranker, Reranker
from ragkit.rewrite import (
    LLM,
    AnthropicLLM,
    DecompositionRewriter,
    HyDERewriter,
    MultiQueryRewriter,
    TemplateLLM,
)

# Last import: importing the submodule ``ragkit.search`` binds the name ``search`` to the module,
# and this import then rebinds it to the function.
from ragkit.search import Article, SearchResult, search

__version__ = "0.2.0"

__all__ = [
    "LLM",
    "Answer",
    "AnthropicLLM",
    "Article",
    "BM25Index",
    "Chunk",
    "CrossEncoderReranker",
    "DecompositionRewriter",
    "DenseIndex",
    "Embedder",
    "ExtractiveGenerator",
    "Generator",
    "HashingEmbedder",
    "HyDERewriter",
    "HybridRetriever",
    "LexicalReranker",
    "MultiQueryRewriter",
    "RAGPipeline",
    "Reranker",
    "STEmbedder",
    "SearchResult",
    "TemplateLLM",
    "answer_relevance",
    "dcg_at_k",
    "faithfulness",
    "fixed",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "recursive",
    "rrf",
    "search",
    "semantic",
    "sentence",
    "tokenize",
    "weighted_fusion",
]
