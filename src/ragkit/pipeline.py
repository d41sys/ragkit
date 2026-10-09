"""End-to-end RAG pipeline composing chunking, retrieval, reranking, and generation."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ragkit.bm25 import tokenize
from ragkit.chunking import Chunk, recursive
from ragkit.hybrid import HybridRetriever
from ragkit.rerank import LexicalReranker, Reranker

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Answer:
    """Generated answer accompanied by cited chunk references."""

    text: str
    citations: list[str] = field(default_factory=list)


class Generator(Protocol):
    """Protocol for answer generation from retrieved context."""

    def generate(self, query: str, chunks: Sequence[Chunk]) -> Answer:
        """Generate answer with chunk citations."""
        ...


class ExtractiveGenerator:
    """Offline extractive generator selecting top sentences with [chunk_id] citations."""

    def __init__(self, max_sentences: int = 3) -> None:
        self.max_sentences = max_sentences

    def generate(self, query: str, chunks: Sequence[Chunk]) -> Answer:
        """Extract key sentences supported by chunks and attach citations."""
        if not chunks:
            return Answer(text="No relevant information found in the corpus.", citations=[])

        q_terms = set(tokenize(query))

        candidates: list[tuple[str, Chunk, float]] = []
        for chunk in chunks:
            raw_sents = _SENTENCE_SPLIT.split(chunk.text.strip())
            for s in raw_sents:
                clean_s = s.strip()
                if len(clean_s) < 20:
                    continue
                s_tokens = set(tokenize(clean_s))
                overlap = len(q_terms & s_tokens)
                score = overlap + (len(clean_s) / 500.0)  # slight tie-break for detail
                candidates.append((clean_s, chunk, score))

        if not candidates:
            # Fallback to first chunk snippet
            c = chunks[0]
            first_sent = c.text.strip().split(".")[0] + "."
            return Answer(text=f"{first_sent} [{c.id}]", citations=[c.id])

        # Sort sentences by relevance
        candidates.sort(key=lambda item: item[2], reverse=True)
        chosen = candidates[: self.max_sentences]

        lines: list[str] = []
        cited_ids: list[str] = []

        for sent, chunk, _ in chosen:
            # Ensure sentence ends with punctuation
            if not sent.endswith((".", "!", "?")):
                sent = sent + "."
            lines.append(f"{sent} [{chunk.id}]")
            if chunk.id not in cited_ids:
                cited_ids.append(chunk.id)

        answer_text = " ".join(lines)
        return Answer(text=answer_text, citations=cited_ids)


class RAGPipeline:
    """Configurable RAG pipeline coordinating ingestion, retrieval, reranking, and generation."""

    def __init__(
        self,
        chunker: Callable[[str, str], list[Chunk]] | None = None,
        retriever: HybridRetriever | None = None,
        reranker: Reranker | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.chunker: Callable[[str, str], list[Chunk]] = (
            chunker if chunker is not None else (lambda text, doc_id: recursive(text, doc_id))
        )
        self.retriever: HybridRetriever = retriever if retriever is not None else HybridRetriever()
        self.reranker: Reranker | None = reranker if reranker is not None else LexicalReranker()
        self.generator: Generator = generator if generator is not None else ExtractiveGenerator()
        self.indexed_chunks: list[Chunk] = []

    def index_documents(self, documents: Sequence[tuple[str, str]]) -> None:
        """Chunk and index a collection of (doc_id, text) pairs."""
        all_chunks: list[Chunk] = []
        for doc_id, text in documents:
            chunks = self.chunker(text, doc_id)
            all_chunks.extend(chunks)

        self.index_chunks(all_chunks)

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index pre-chunked passages directly into the retriever."""
        self.indexed_chunks.extend(chunks)
        self.retriever.add(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        """Retrieve and rerank candidate chunks for a query."""
        initial_candidates = self.retriever.search(query, k=max(top_k * 2, 10))
        if not initial_candidates:
            return []

        if self.reranker is not None:
            reranked = self.reranker.rerank(query, initial_candidates, top_k=top_k)
            return [chunk for chunk, _ in reranked]

        return [chunk for chunk, _ in initial_candidates[:top_k]]

    def answer(self, query: str, top_k: int = 5) -> Answer:
        """Answer a query end-to-end with chunk retrieval and generation."""
        top_chunks = self.retrieve(query, top_k=top_k)
        return self.generator.generate(query, top_chunks)
