"""Embedding protocols and implementations.

Provides HashingEmbedder (offline default with char 3-5 gram hashing + TF-IDF + L2)
and STEmbedder (sentence-transformers all-MiniLM-L6-v2 under optional extra [st]).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding textual passages into fixed-dimensional vectors."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of texts into an (N, D) float32 matrix."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string into a (D,) float32 vector."""
        ...


def _stable_hash(gram: str) -> int:
    """Deterministic hash for a string."""
    h = 2166136261
    for ch in gram:
        h = (h ^ ord(ch)) * 16777619
        h &= 0xFFFFFFFF
    return h


class HashingEmbedder:
    """Offline embedding using character 3-5 gram hashing, TF-IDF weighting, and L2 normalization.

    Extracts character 3-grams, 4-grams, and 5-grams, maps them to buckets via stable hashing,
    weights terms using TF-IDF, and projects onto a unit sphere.
    """

    def __init__(self, dim: int = 512, min_n: int = 3, max_n: int = 5) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.min_n = min_n
        self.max_n = max_n
        self.idf_: np.ndarray | None = None
        self._doc_count: int = 0

    def _extract_ngrams(self, text: str) -> list[str]:
        cleaned = f" {text.lower().strip()} "
        ngrams: list[str] = []
        text_len = len(cleaned)
        for n in range(self.min_n, self.max_n + 1):
            if text_len >= n:
                for i in range(text_len - n + 1):
                    ngrams.append(cleaned[i : i + n])
        return ngrams

    def fit(self, texts: Sequence[str]) -> HashingEmbedder:
        """Compute document frequencies and IDFs across a corpus."""
        n_docs = len(texts)
        if n_docs == 0:
            return self

        df = np.zeros(self.dim, dtype=np.float32)
        for t in texts:
            ngrams = self._extract_ngrams(t)
            buckets = set(_stable_hash(ng) % self.dim for ng in ngrams)
            for b in buckets:
                df[b] += 1.0

        # Smooth Robertson/Spärck Jones or smoothed logarithmic IDF
        self.idf_ = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        self._doc_count = n_docs
        return self

    def _vectorize(self, text: str, idf: np.ndarray) -> np.ndarray:
        ngrams = self._extract_ngrams(text)
        if not ngrams:
            return np.zeros(self.dim, dtype=np.float32)

        counts = Counter(_stable_hash(ng) % self.dim for ng in ngrams)
        vec = np.zeros(self.dim, dtype=np.float32)

        for bucket, cnt in counts.items():
            # Sublinear term frequency
            tf = 1.0 + math.log(cnt)
            vec[bucket] = float(tf * idf[bucket])

        norm = float(np.linalg.norm(vec))
        if norm > 1e-12:
            vec /= norm
        return vec

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed multiple texts into an (N, D) normalized numpy array."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        if self.idf_ is None:
            # If not pre-fitted, calculate IDF over this batch
            df = np.zeros(self.dim, dtype=np.float32)
            for t in texts:
                ngrams = self._extract_ngrams(t)
                buckets = set(_stable_hash(ng) % self.dim for ng in ngrams)
                for b in buckets:
                    df[b] += 1.0
            idf = np.log((len(texts) + 1.0) / (df + 1.0)) + 1.0
        else:
            idf = self.idf_

        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            matrix[i] = self._vectorize(t, idf)
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query into a (D,) normalized vector."""
        idf = self.idf_ if self.idf_ is not None else np.ones(self.dim, dtype=np.float32)
        return self._vectorize(text, idf)


class STEmbedder:
    """SentenceTransformer neural embedder using all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is not installed. Install with: pip install 'ragkit[st]'"
            ) from err

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed multiple texts with all-MiniLM-L6-v2."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vectors = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with all-MiniLM-L6-v2."""
        vec: np.ndarray = self.embed([text])[0]
        return vec
