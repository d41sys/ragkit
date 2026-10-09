"""Dense retrieval index using numpy cosine similarity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ragkit.chunking import Chunk
from ragkit.embed import Embedder, HashingEmbedder


class DenseIndex:
    """In-memory dense vector index using cosine similarity."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder: Embedder = embedder if embedder is not None else HashingEmbedder(dim=512)
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray = np.empty((0, 512), dtype=np.float32)

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks to the index, embedding their text."""
        if not chunks:
            return

        texts = [c.text for c in chunks]

        # If using HashingEmbedder and it's not yet fitted, fit it on corpus
        if isinstance(self.embedder, HashingEmbedder) and self.embedder.idf_ is None:
            self.embedder.fit(texts)

        new_vecs = self.embedder.embed(texts)
        if len(self.chunks) == 0:
            self.vectors = new_vecs
        else:
            self.vectors = np.vstack([self.vectors, new_vecs])
        self.chunks.extend(chunks)

    def search_vec(self, query_vec: np.ndarray, k: int = 10) -> list[tuple[Chunk, float]]:
        """Search top-k chunks given a precomputed query vector."""
        if len(self.chunks) == 0 or len(self.vectors) == 0:
            return []

        # Ensure query_vec is unit normalized
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(q))
        if norm > 1e-12:
            q = q / norm

        scores = self.vectors @ q
        top_k = min(k, len(self.chunks))
        if top_k <= 0:
            return []

        # Get top-k indices sorted descending
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.chunks[idx], float(scores[idx])) for idx in top_indices]

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        """Embed text query and retrieve top-k chunks by cosine similarity."""
        if len(self.chunks) == 0:
            return []
        q_vec = self.embedder.embed_query(query)
        return self.search_vec(q_vec, k=k)
