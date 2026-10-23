"""BM25F: BM25 over weighted fields (chunk title and body), with weighted query terms."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from ragkit.bm25 import tokenize
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
    ) -> None:
        if title_weight < 0.0 or body_weight < 0.0:
            raise ValueError("field weights must be non-negative")
        self.k1 = k1
        self.b = b
        self.title_weight = title_weight
        self.body_weight = body_weight
        self.chunks: list[Chunk] = []
        self.title_lens: list[int] = []
        self.body_lens: list[int] = []
        self.df: Counter[str] = Counter()
        # term -> [(chunk index, title tf, body tf)]
        self.postings: dict[str, list[tuple[int, int, int]]] = {}
        self.avg_title_len = 0.0
        self.avg_body_len = 0.0

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks to the index and update corpus statistics."""
        for chunk in chunks:
            idx = len(self.chunks)
            title_tf = Counter(tokenize(chunk.title))
            body_tf = Counter(tokenize(chunk.text))
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
        terms = [(t, 1.0) for t in tokenize(query)] if isinstance(query, str) else list(query)

        scores = [0.0] * len(self.chunks)
        for term, q_weight in terms:
            term_idf = self.idf(term)
            if term_idf == 0.0 or q_weight == 0.0:
                continue
            for idx, title_tf, body_tf in self.postings.get(term, ()):
                tf = self._combined_tf(idx, title_tf, body_tf)
                if tf > 0.0:
                    scores[idx] += q_weight * term_idf * (self.k1 + 1.0) * tf / (self.k1 + tf)

        ranked = [(self.chunks[i], s) for i, s in enumerate(scores) if s > 0.0]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]

    def _combined_tf(self, idx: int, title_tf: int, body_tf: int) -> float:
        total = 0.0
        if title_tf:
            norm = 1.0 - self.b + self.b * self.title_lens[idx] / (self.avg_title_len or 1.0)
            total += self.title_weight * title_tf / norm
        if body_tf:
            norm = 1.0 - self.b + self.b * self.body_lens[idx] / (self.avg_body_len or 1.0)
            total += self.body_weight * body_tf / norm
        return total
