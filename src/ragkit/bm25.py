"""Okapi BM25 ranking algorithm from scratch."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from ragkit.chunking import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 inverted index with configurable k1 and b parameters."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.doc_lens: list[int] = []
        self.df: Counter[str] = Counter()
        self.tf: list[Counter[str]] = []
        self.avgdl: float = 0.0

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Add chunks to the index and update corpus statistics."""
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            term_counts = Counter(tokens)
            self.chunks.append(chunk)
            self.doc_lens.append(len(tokens))
            self.tf.append(term_counts)
            for term in term_counts:
                self.df[term] += 1

        total_tokens = sum(self.doc_lens)
        total_docs = len(self.doc_lens)
        self.avgdl = total_tokens / max(1, total_docs)

    def idf(self, term: str) -> float:
        """Compute Robertson/Spärck Jones IDF with +1 smoothing."""
        n = len(self.chunks)
        df_val = self.df.get(term, 0)
        if df_val == 0 or n == 0:
            return 0.0
        return math.log((n - df_val + 0.5) / (df_val + 0.5) + 1.0)

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        """Rank indexed chunks against the query using Okapi BM25 scoring."""
        if not self.chunks:
            return []

        q_terms = tokenize(query)
        if not q_terms:
            return []

        n = len(self.chunks)
        scores = [0.0] * n

        for term in q_terms:
            df_val = self.df.get(term, 0)
            if df_val == 0:
                continue

            term_idf = math.log((n - df_val + 0.5) / (df_val + 0.5) + 1.0)

            for i, counts in enumerate(self.tf):
                f = counts.get(term, 0)
                if f == 0:
                    continue

                dl = self.doc_lens[i]
                norm_len = 1.0 - self.b + self.b * (dl / (self.avgdl or 1.0))
                denom = f + self.k1 * norm_len
                tf_weight = f * (self.k1 + 1.0) / denom
                scores[i] += term_idf * tf_weight

        # Filter positive scores and sort descending
        ranked = [(self.chunks[i], score) for i, score in enumerate(scores) if score > 0.0]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:k]
