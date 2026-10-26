"""Okapi BM25 ranking algorithm from scratch."""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from ragkit.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 inverted index with configurable k1 and b parameters.

    ``analyzer`` turns text into terms (default: :func:`tokenize`); documents and queries use
    the same one. ``length_fn`` maps an exact document length to the length used in the length
    normalisation (identity by default; ``ragkit.analysis.lucene_length`` reproduces Lucene's
    one-byte norm). ``avgdl`` always uses the exact lengths.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        analyzer: Callable[[str], list[str]] = tokenize,
        length_fn: Callable[[int], int] | None = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.analyzer = analyzer
        self.length_fn = length_fn
        self.chunks: list[Chunk] = []
        self.doc_lens: list[int] = []
        self.df: Counter[str] = Counter()
        self.avgdl: float = 0.0
        # term -> (chunk indices, term frequencies), in chunk order
        self._postings: dict[str, tuple[list[int], list[int]]] = {}
        self._norm: tuple[float, np.ndarray] | None = None
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks to the index and update corpus statistics."""
        for chunk in chunks:
            idx = len(self.chunks)
            term_counts = Counter(self.analyzer(chunk.text))
            self.chunks.append(chunk)
            self.doc_lens.append(sum(term_counts.values()))
            for term, freq in term_counts.items():
                self.df[term] += 1
                entry = self._postings.setdefault(term, ([], []))
                entry[0].append(idx)
                entry[1].append(freq)

        total_tokens = sum(self.doc_lens)
        total_docs = len(self.doc_lens)
        self.avgdl = total_tokens / max(1, total_docs)
        self._norm = None
        self._arrays = {}

    def idf(self, term: str) -> float:
        """Compute Robertson/Spärck Jones IDF with +1 smoothing."""
        n = len(self.chunks)
        df_val = self.df.get(term, 0)
        if df_val == 0 or n == 0:
            return 0.0
        return math.log((n - df_val + 0.5) / (df_val + 0.5) + 1.0)

    def _length_norms(self) -> np.ndarray:
        if self._norm is None or self._norm[0] != self.b:  # ``b`` may be reassigned
            lens = self.doc_lens
            if self.length_fn is not None:
                lens = [self.length_fn(n) for n in lens]
            dl = np.asarray(lens, dtype=np.float64)
            self._norm = (self.b, 1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
        return self._norm[1]

    def _posting_arrays(self, term: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._arrays.get(term)
        if cached is None:
            idx, freq = self._postings[term]
            cached = (np.asarray(idx, dtype=np.int64), np.asarray(freq, dtype=np.float64))
            self._arrays[term] = cached
        return cached

    def score_all(self, query: str) -> np.ndarray:
        """BM25 score of every indexed chunk (zeros for chunks sharing no query term)."""
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        if not self.chunks:
            return scores
        n = len(self.chunks)
        norm = self._length_norms()
        for term in self.analyzer(query):
            df_val = self.df.get(term, 0)
            if df_val == 0:
                continue
            term_idf = math.log((n - df_val + 0.5) / (df_val + 0.5) + 1.0)
            idx, f = self._posting_arrays(term)
            denom = f + self.k1 * norm[idx]
            scores[idx] += term_idf * (f * (self.k1 + 1.0) / denom)
        return scores

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        """Rank indexed chunks against the query using Okapi BM25 scoring."""
        if not self.chunks:
            return []
        return [(self.chunks[i], s) for i, s in top_k_indices(self.score_all(query), k)]


def top_k_indices(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices and scores of the ``k`` largest positive scores, ties broken by lower index."""
    if k <= 0:
        return []
    pos = np.flatnonzero(scores > 0.0)
    if pos.size == 0:
        return []
    if pos.size > k:
        vals = scores[pos]
        # Everything at or above the k-th largest value; ties at the boundary are then resolved
        # by index, like a stable sort of the full list.
        threshold = np.partition(vals, pos.size - k)[pos.size - k]
        pos = pos[vals >= threshold]
    picked = heapq.nsmallest(k, ((-float(scores[i]), int(i)) for i in pos))
    return [(i, -neg) for neg, i in picked]
