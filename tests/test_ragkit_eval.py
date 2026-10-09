"""Tests for evaluation metrics: precision, recall, MRR, nDCG, faithfulness, answer relevance."""

import math

import pytest
from ragkit.embed import HashingEmbedder
from ragkit.eval import (
    answer_relevance,
    faithfulness,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_retrieval_metrics_on_fixed_rankings() -> None:
    retrieved = ["d1", "d2", "d3", "d4", "d5"]
    relevant = {"d1", "d3"}  # 2 relevant documents total

    # Precision@k
    assert precision_at_k(retrieved, relevant, k=1) == pytest.approx(1.0)
    assert precision_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2.0 / 3.0)
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(2.0 / 5.0)
    assert precision_at_k([], relevant, k=5) == 0.0
    assert precision_at_k(retrieved, set(), k=5) == 0.0

    # Recall@k
    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(0.5)
    assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(0.5)
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)
    assert recall_at_k(retrieved, set(), k=5) == 0.0

    # MRR
    assert mrr(retrieved, relevant) == pytest.approx(1.0)  # first match at index 0 (rank 1)
    assert mrr(["d2", "d1"], relevant) == pytest.approx(0.5)  # first match at index 1 (rank 2)
    assert mrr(["d2", "d4"], relevant) == 0.0  # no match


def test_ndcg_at_k_exact_values() -> None:
    # Perfect ranking
    graded = {"doc_rel": 2, "doc_med": 1, "doc_irr": 0}
    perfect = ["doc_rel", "doc_med", "doc_irr"]
    assert ndcg_at_k(perfect, graded, k=3) == pytest.approx(1.0)

    # Inverted ranking
    # Actual DCG: doc_med at rank 0, doc_rel at rank 1
    # doc_med (rel 1): (2^1 - 1)/log2(2) = 1.0
    # doc_rel (rel 2): (2^2 - 1)/log2(3) = 3.0 / log2(3) ≈ 1.892789
    # Actual DCG = 1.0 + 1.892789 = 2.892789
    # Ideal DCG: doc_rel at rank 0, doc_med at rank 1
    # doc_rel (rel 2): 3.0 / log2(2) = 3.0
    # doc_med (rel 1): 1.0 / log2(3) ≈ 0.630930
    # Ideal DCG = 3.0 + 0.630930 = 3.630930
    # nDCG = 2.892789 / 3.630930 ≈ 0.7967
    inverted = ["doc_med", "doc_rel", "doc_irr"]
    ideal_dcg = 3.0 + (1.0 / math.log2(3))
    actual_dcg = 1.0 + (3.0 / math.log2(3))
    expected_ndcg = actual_dcg / ideal_dcg

    assert ndcg_at_k(inverted, graded, k=2) == pytest.approx(expected_ndcg, rel=1e-4)
    assert ndcg_at_k(["doc_irr"], graded, k=1) == 0.0


def test_faithfulness_token_overlap() -> None:
    context = [
        "Imbalance settlement financially penalizes deviations between scheduled "
        "positions and metered flows.",
        "Dual pricing applies separate penalty rates depending on system direction.",
    ]

    # Sentence heavily overlapping context
    supported_ans = (
        "Imbalance settlement financially penalizes deviations between scheduled and "
        "metered flows. "
        "Dual pricing applies penalty rates depending on system direction."
    )
    score_supp = faithfulness(supported_ans, context, threshold=0.5)
    assert score_supp == pytest.approx(1.0)

    # Sentence with completely hallucinated / disjoint content
    hallucinated_ans = (
        "Astronauts discovered ancient lunar basalt structures on the far side of the moon."
    )
    score_unsupp = faithfulness(hallucinated_ans, context, threshold=0.5)
    assert score_unsupp == pytest.approx(0.0)


def test_answer_relevance_cosine() -> None:
    embedder = HashingEmbedder(dim=256)
    q = "What is battery energy storage dispatch?"
    a_good = "Battery energy storage dispatch optimizes charge and discharge cycles."
    a_unrelated = "Ancient medieval manuscripts were penned using iron gall ink."

    score_good = answer_relevance(q, a_good, embedder=embedder)
    score_bad = answer_relevance(q, a_unrelated, embedder=embedder)

    assert score_good > score_bad
    assert score_good > 0.3
