"""Tests for hybrid retrieval, RRF ordering, dense indexing, and lexical reranking."""

import pytest
from ragkit.chunking import Chunk
from ragkit.dense import DenseIndex
from ragkit.embed import HashingEmbedder
from ragkit.hybrid import rrf, weighted_fusion
from ragkit.rerank import LexicalReranker


def test_rrf_ordering_and_exact_scores() -> None:
    """Verify Reciprocal Rank Fusion ordering and score calculation with k=60.

    Doc A: rank 0 in List 1, rank 1 in List 2 -> 1/61 + 1/62 ≈ 0.0163934 + 0.0161290 = 0.0325225
    Doc B: rank 1 in List 1, rank 0 in List 2 -> 1/62 + 1/61 = 0.0325225 (tied)
    Doc C: rank 2 in List 1, rank 2 in List 2 -> 1/63 + 1/63 ≈ 0.0317460
    Doc D: rank 0 in List 1 only            -> 1/61 + 0     ≈ 0.0163934
    """
    ca = Chunk(id="A", doc_id="A", text="text A", start=0, end=6)
    cb = Chunk(id="B", doc_id="B", text="text B", start=0, end=6)
    cc = Chunk(id="C", doc_id="C", text="text C", start=0, end=6)

    # List 1: A (rank 0), C (rank 1)
    list1 = [(ca, 10.0), (cc, 5.0)]
    # List 2: C (rank 0), B (rank 1)
    list2 = [(cc, 8.0), (cb, 4.0)]

    # C has rank 1 in list 1 (1/62) and rank 0 in list 2 (1/61) -> 1/61 + 1/62
    # A has rank 0 in list 1 (1/61) -> 1/61
    # B has rank 1 in list 2 (1/62) -> 1/62
    fused = rrf([list1, list2], k=60)
    assert len(fused) == 3
    assert fused[0][0].id == "C"
    assert fused[1][0].id == "A"
    assert fused[2][0].id == "B"

    expected_score_c = (1.0 / 61.0) + (1.0 / 62.0)
    assert fused[0][1] == pytest.approx(expected_score_c, rel=1e-5)


def test_weighted_score_fusion() -> None:
    c1 = Chunk(id="c1", doc_id="c1", text="chunk one", start=0, end=9)
    c2 = Chunk(id="c2", doc_id="c2", text="chunk two", start=0, end=9)

    bm25_res = [(c1, 10.0), (c2, 0.0)]  # normalized: c1=1.0, c2=0.0
    dense_res = [(c2, 1.0), (c1, 0.0)]  # normalized: c2=1.0, c1=0.0

    # With alpha = 0.5: c1 = 0.5*1.0 + 0.5*0.0 = 0.5; c2 = 0.5*0.0 + 0.5*1.0 = 0.5
    fused_half = weighted_fusion(bm25_res, dense_res, alpha=0.5)
    assert len(fused_half) == 2
    assert fused_half[0][1] == pytest.approx(0.5, rel=1e-5)

    # With alpha = 0.8: c1 = 0.8*1.0 = 0.8; c2 = 0.2*1.0 = 0.2 -> c1 ranks first
    fused_lexical = weighted_fusion(bm25_res, dense_res, alpha=0.8)
    assert fused_lexical[0][0].id == "c1"
    assert fused_lexical[0][1] == pytest.approx(0.8, rel=1e-5)
    assert fused_lexical[1][0].id == "c2"
    assert fused_lexical[1][1] == pytest.approx(0.2, rel=1e-5)


def test_dense_index_cosine_search() -> None:
    embedder = HashingEmbedder(dim=256)
    dense = DenseIndex(embedder=embedder)

    c1 = Chunk(id="c1", doc_id="d1", text="battery energy storage dispatch", start=0, end=32)
    c2 = Chunk(id="c2", doc_id="d2", text="deep sea oceanographic biology", start=0, end=30)
    dense.add([c1, c2])

    results = dense.search("battery storage", k=2)
    assert len(results) == 2
    assert results[0][0].id == "c1"
    assert results[0][1] > results[1][1]


def test_lexical_reranker_coverage_and_proximity() -> None:
    reranker = LexicalReranker()

    # Chunk with terms spread apart
    c_far = Chunk(
        id="far",
        doc_id="d1",
        text="demand for electricity is high and response time is critical",
        start=0,
        end=60,
    )
    # Chunk with query terms adjacent
    c_close = Chunk(
        id="close",
        doc_id="d2",
        text="industrial demand response mechanisms balance the transmission grid",
        start=0,
        end=68,
    )

    candidates = [(c_far, 0.5), (c_close, 0.5)]
    reranked = reranker.rerank("demand response", candidates, top_k=2)

    assert len(reranked) == 2
    # c_close should rank first due to higher span proximity
    assert reranked[0][0].id == "close"
