"""BM25F: BM25 over weighted fields (chunk title and body), with weighted query terms."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

from ragkit.bm25 import tokenize, top_k_indices
from ragkit.chunking import Chunk


class BM25FIndex:
    """BM25F (Robertson, Zaragoza and Taylor) over a ``title`` and a ``body`` field.

    Per chunk, the term frequency is combined across fields after per-field length
    normalisation::

        tf~ = title_weight * tf_title / (1 - b + b * len_title / avglen_title)
            + body_weight  * tf_body  / (1 - b + b * len_body  / avglen_body)
        score(term) = idf * (k1 + 1) * tf~ / (k1 + tf~)

    The title is ``Chunk.title`` and the body is ``Chunk.text``. ``idf`` is the Robertson/Spärck
    Jones form of ``BM25Index``; a chunk counts towards the document frequency of a term when
    the term is in the body, or in the title and ``title_weight > 0``. With
    ``title_weight == 0`` the scores equal those of ``BM25Index``.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        title_weight: float = 2.0,
        body_weight: float = 1.0,
        analyzer: Callable[[str], list[str]] = tokenize,
    ) -> None:
        if title_weight < 0.0 or body_weight < 0.0:
            raise ValueError("field weights must be non-negative")
        self.k1 = k1
        self.b = b
        self.title_weight = title_weight
        self.body_weight = body_weight
        self.analyzer = analyzer
        self.chunks: list[Chunk] = []
        self.title_lens: list[int] = []
        self.body_lens: list[int] = []
        self.df: Counter[str] = Counter()
        # term -> [(chunk index, title tf, body tf)]
        self.postings: dict[str, list[tuple[int, int, int]]] = {}
        self.avg_title_len = 0.0
        self.avg_body_len = 0.0
        self._arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._norms: tuple[float, np.ndarray, np.ndarray] | None = None

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks to the index and update corpus statistics."""
        for chunk in chunks:
            idx = len(self.chunks)
            title_tf = Counter(self.analyzer(chunk.title))
            body_tf = Counter(self.analyzer(chunk.text))
            self.chunks.append(chunk)
            self.title_lens.append(sum(title_tf.values()))
            self.body_lens.append(sum(body_tf.values()))
            for term in title_tf.keys() | body_tf.keys():
                self.postings.setdefault(term, []).append((idx, title_tf[term], body_tf[term]))
                if body_tf[term] > 0 or (title_tf[term] > 0 and self.title_weight > 0.0):
                    self.df[term] += 1
        n = max(1, len(self.chunks))
        self.avg_title_len = sum(self.title_lens) / n
        self.avg_body_len = sum(self.body_lens) / n
        self._arrays = {}
        self._norms = None

    def idf(self, term: str) -> float:
        """Robertson/Spärck Jones IDF with +1 smoothing."""
        n = len(self.chunks)
        df_val = self.df.get(term, 0)
        if df_val == 0 or n == 0:
            return 0.0
        return math.log((n - df_val + 0.5) / (df_val + 0.5) + 1.0)

    def search(
        self, query: str | Sequence[tuple[str, float]], k: int = 10
    ) -> list[tuple[Chunk, float]]:
        """Rank chunks for a query string or for weighted ``(term, weight)`` pairs.

        A string is tokenised and every token gets weight 1.0. Repeated terms add up, as in
        ``BM25Index``. Only chunks with a positive score are returned.
        """
        if not self.chunks:
            return []
        return [(self.chunks[i], s) for i, s in top_k_indices(self.score_all(query), k)]

    def score_all(self, query: str | Sequence[tuple[str, float]]) -> np.ndarray:
        """BM25F score of every indexed chunk (zeros for chunks sharing no query term)."""
        terms = [(t, 1.0) for t in self.analyzer(query)] if isinstance(query, str) else list(query)
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        if not self.chunks:
            return scores
        if self._norms is None or self._norms[0] != self.b:  # ``b`` may be reassigned
            title_len = np.asarray(self.title_lens, dtype=np.float64)
            body_len = np.asarray(self.body_lens, dtype=np.float64)
            # A zero norm needs b == 1 and an empty field; its term frequency is 0 anyway.
            title_norm = 1.0 - self.b + self.b * title_len / (self.avg_title_len or 1.0)
            body_norm = 1.0 - self.b + self.b * body_len / (self.avg_body_len or 1.0)
            self._norms = (
                self.b,
                np.where(title_norm > 0.0, title_norm, 1.0),
                np.where(body_norm > 0.0, body_norm, 1.0),
            )
        _, title_norm, body_norm = self._norms
        for term, q_weight in terms:
            term_idf = self.idf(term)
            if term_idf == 0.0 or q_weight == 0.0:
                continue
            arrays = self._arrays.get(term)
            if arrays is None:
                post = self.postings[term]
                arrays = (
                    np.asarray([p[0] for p in post], dtype=np.int64),
                    np.asarray([p[1] for p in post], dtype=np.float64),
                    np.asarray([p[2] for p in post], dtype=np.float64),
                )
                self._arrays[term] = arrays
            idx, title_tf, body_tf = arrays
            tf = self.title_weight * title_tf / title_norm[idx]
            tf = tf + self.body_weight * body_tf / body_norm[idx]
            positive = tf > 0.0
            if not positive.all():
                idx, tf = idx[positive], tf[positive]
            scores[idx] += q_weight * term_idf * (self.k1 + 1.0) * tf / (self.k1 + tf)
        return scores
