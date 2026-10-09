"""Reranking algorithms for retrieved candidates."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ragkit.bm25 import tokenize
from ragkit.chunking import Chunk


@runtime_checkable
class Reranker(Protocol):
    """Protocol for candidate rerankers."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Chunk, float]],
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """Rerank candidate chunks according to refined relevance criteria."""
        ...


class LexicalReranker:
    """Reranker combining query-term coverage, minimum span proximity, and BM25 score linearly.

    - Coverage: fraction of unique query terms present in candidate.
    - Proximity: compactness of the shortest window containing the matched query terms.
    - BM25: Okapi BM25 score normalized across candidate set.
    """

    def __init__(
        self,
        weight_coverage: float = 0.4,
        weight_proximity: float = 0.3,
        weight_bm25: float = 0.3,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.w_cov = weight_coverage
        self.w_prox = weight_proximity
        self.w_bm25 = weight_bm25
        self.k1 = k1
        self.b = b

    def _calc_coverage(self, q_terms: set[str], doc_terms: set[str]) -> float:
        if not q_terms:
            return 0.0
        return len(q_terms & doc_terms) / len(q_terms)

    def _calc_min_span_proximity(self, q_terms: set[str], doc_tokens: list[str]) -> float:
        if not q_terms or not doc_tokens:
            return 0.0

        # Map each position to matched query term
        term_positions: list[tuple[int, str]] = [
            (i, tok) for i, tok in enumerate(doc_tokens) if tok in q_terms
        ]
        if not term_positions:
            return 0.0

        matched_terms = set(tok for _, tok in term_positions)
        target_distinct = len(matched_terms)
        if target_distinct <= 1:
            return 1.0

        # Two-pointer shortest window containing all matched_terms
        min_span = len(doc_tokens)
        counts: Counter[str] = Counter()
        left = 0

        for right in range(len(term_positions)):
            pos_r, term_r = term_positions[right]
            counts[term_r] += 1

            while len(counts) == target_distinct:
                pos_l, term_l = term_positions[left]
                span = pos_r - pos_l + 1
                if span < min_span:
                    min_span = span
                counts[term_l] -= 1
                if counts[term_l] == 0:
                    del counts[term_l]
                left += 1

        # Proximity is inverted span length normalized by distinct term count
        return min(1.0, target_distinct / max(1, min_span))

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Chunk, float]],
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """Rerank candidates using linear blend of coverage, proximity, and BM25."""
        if not candidates:
            return []

        q_token_list = tokenize(query)
        q_terms = set(q_token_list)
        if not q_terms:
            return list(candidates[:top_k])

        # Compute corpus statistics across the candidate pool for local BM25
        n = len(candidates)
        tokenized_pool = [tokenize(c.text) for c, _ in candidates]
        lens = [len(toks) for toks in tokenized_pool]
        avgdl = sum(lens) / max(1, n)

        df: Counter[str] = Counter()
        for toks in tokenized_pool:
            for term in set(toks):
                if term in q_terms:
                    df[term] += 1

        bm25_raw: list[float] = []
        for i, toks in enumerate(tokenized_pool):
            tf = Counter(toks)
            dl = lens[i]
            score = 0.0
            for term in q_terms:
                df_val = df.get(term, 0)
                if df_val == 0:
                    continue
                idf = math.log((n - df_val + 0.5) / (df_val + 0.5) + 1.0)
                f = tf.get(term, 0)
                if f > 0:
                    norm_len = 1.0 - self.b + self.b * (dl / (avgdl or 1.0))
                    score += idf * f * (self.k1 + 1.0) / (f + self.k1 * norm_len)
            bm25_raw.append(score)

        max_bm25 = max(bm25_raw) if bm25_raw else 1.0
        min_bm25 = min(bm25_raw) if bm25_raw else 0.0
        range_bm25 = max_bm25 - min_bm25

        scored_candidates: list[tuple[Chunk, float]] = []
        for i, (chunk, _) in enumerate(candidates):
            toks = tokenized_pool[i]
            doc_terms = set(toks)
            cov = self._calc_coverage(q_terms, doc_terms)
            prox = self._calc_min_span_proximity(q_terms, toks)
            norm_bm25 = 1.0 if range_bm25 == 0.0 else (bm25_raw[i] - min_bm25) / range_bm25

            final_score = self.w_cov * cov + self.w_prox * prox + self.w_bm25 * norm_bm25
            scored_candidates.append((chunk, float(final_score)))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)
        return scored_candidates[:top_k]


class CrossEncoderReranker:
    """Neural cross-encoder reranker using cross-encoder/ms-marco-MiniLM-L-6-v2."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as err:
            raise ImportError(
                "sentence-transformers is not installed. Install with: pip install 'ragkit[st]'"
            ) from err

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[Chunk, float]],
        top_k: int = 10,
    ) -> list[tuple[Chunk, float]]:
        """Score candidate pairs [query, chunk.text] using the cross-encoder."""
        if not candidates:
            return []

        pairs = [[query, chunk.text] for chunk, _ in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        scored = [
            (chunk, float(score)) for (chunk, _), score in zip(candidates, scores, strict=True)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
