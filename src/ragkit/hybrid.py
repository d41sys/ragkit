"""Hybrid retrieval combining lexical (BM25) and dense vector search."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from ragkit.bm25 import BM25Index
from ragkit.chunking import Chunk
from ragkit.dense import DenseIndex


def rrf(
    rankings: Sequence[Sequence[tuple[Chunk, float]]],
    k: int = 60,
) -> list[tuple[Chunk, float]]:
    """Reciprocal Rank Fusion across multiple rankings.

    Score for chunk d = sum_over_rankings( 1.0 / (k + rank + 1) ).
    """
    scores: dict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, Chunk] = {}

    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking):
            scores[chunk.id] += 1.0 / (k + rank + 1)
            chunks_by_id[chunk.id] = chunk

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(chunks_by_id[cid], score) for cid, score in fused]


def _min_max_normalize(results: Sequence[tuple[Chunk, float]]) -> dict[str, float]:
    """Min-max normalize scores of a ranked list into [0.0, 1.0]."""
    if not results:
        return {}
    scores = [s for _, s in results]
    min_s = min(scores)
    max_s = max(scores)
    diff = max_s - min_s

    norm_scores: dict[str, float] = {}
    for chunk, s in results:
        norm_scores[chunk.id] = 1.0 if diff == 0.0 else (s - min_s) / diff
    return norm_scores


def weighted_fusion(
    bm25_results: Sequence[tuple[Chunk, float]],
    dense_results: Sequence[tuple[Chunk, float]],
    alpha: float = 0.5,
) -> list[tuple[Chunk, float]]:
    """Weighted score fusion with min-max normalization.

    score = alpha * norm(bm25) + (1 - alpha) * norm(dense)
    """
    norm_bm25 = _min_max_normalize(bm25_results)
    norm_dense = _min_max_normalize(dense_results)

    chunks_by_id: dict[str, Chunk] = {}
    for c, _ in bm25_results:
        chunks_by_id[c.id] = c
    for c, _ in dense_results:
        chunks_by_id[c.id] = c

    all_ids = set(norm_bm25.keys()) | set(norm_dense.keys())
    fused_scores: dict[str, float] = {}

    for cid in all_ids:
        s_b = norm_bm25.get(cid, 0.0)
        s_d = norm_dense.get(cid, 0.0)
        fused_scores[cid] = alpha * s_b + (1.0 - alpha) * s_d

    fused = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [(chunks_by_id[cid], score) for cid, score in fused]


class HybridRetriever:
    """Hybrid retriever orchestrating BM25 and Dense search."""

    def __init__(self, bm25: BM25Index | None = None, dense: DenseIndex | None = None) -> None:
        self.bm25: BM25Index = bm25 if bm25 is not None else BM25Index()
        self.dense: DenseIndex = dense if dense is not None else DenseIndex()

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks in both BM25 and dense indices."""
        self.bm25.add(chunks)
        self.dense.add(chunks)

    def search(
        self,
        query: str,
        k: int = 10,
        method: str = "rrf",
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve top chunks using lexical and dense signals fused together."""
        k_retrieve = max(k * 2, 20)
        bm25_res = self.bm25.search(query, k=k_retrieve)
        dense_res = self.dense.search(query, k=k_retrieve)

        if method == "rrf":
            fused = rrf([bm25_res, dense_res], k=rrf_k)
        elif method == "weighted":
            fused = weighted_fusion(bm25_res, dense_res, alpha=alpha)
        else:
            raise ValueError(f"Unknown fusion method '{method}'. Choose 'rrf' or 'weighted'.")

        return fused[:k]
