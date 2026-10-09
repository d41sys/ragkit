"""Tests for RagKit chunking strategies: boundary invariants and text reconstruction."""

import pytest
from ragkit.chunking import Chunk, fixed, recursive, semantic, sentence
from ragkit.embed import HashingEmbedder


def test_chunk_dataclass_validation() -> None:
    c = Chunk(id="d1#0", doc_id="d1", text="Sample", start=0, end=6)
    assert c.id == "d1#0"
    assert c.start == 0
    assert c.end == 6

    with pytest.raises(ValueError, match="start cannot be negative"):
        Chunk(id="d1#0", doc_id="d1", text="Sample", start=-1, end=5)

    with pytest.raises(ValueError, match=r"end.*must be >= start"):
        Chunk(id="d1#0", doc_id="d1", text="Sample", start=10, end=5)


def test_fixed_window_invariants() -> None:
    text = (
        "Demand response allows industrial consumers to modulate electricity consumption. "
        "Batteries provide fast frequency response. "
        "Solar PV output exhibits cloud ramp volatility. "
        "Day ahead electricity auctions clear supply and demand using uniform marginal pricing."
    )
    chunks = fixed(text, doc_id="doc_test", size=80, overlap=20)
    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.doc_id == "doc_test"
        assert 0 <= chunk.start < chunk.end <= len(text)
        assert text[chunk.start : chunk.end] == chunk.text
        assert len(chunk.text) <= 80

    # Test empty text
    assert fixed("", doc_id="doc_empty") == []

    # Test invalid parameters
    with pytest.raises(ValueError, match="size must be positive"):
        fixed(text, doc_id="d", size=0)
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        fixed(text, doc_id="d", size=50, overlap=-1)
    with pytest.raises(ValueError, match=r"overlap.*smaller than size"):
        fixed(text, doc_id="d", size=50, overlap=50)


def test_sentence_chunking_invariants() -> None:
    text = (
        "First sentence defines energy markets. Second sentence explains reserve capacity. "
        "Third sentence details imbalance settlement. Fourth sentence covers heat pumps."
    )
    chunks = sentence(text, doc_id="doc_sent", max_tokens=10, overlap=0)
    assert len(chunks) >= 2

    for chunk in chunks:
        assert chunk.doc_id == "doc_sent"
        assert 0 <= chunk.start < chunk.end <= len(text)
        assert text[chunk.start : chunk.end] == chunk.text

    assert sentence("", doc_id="empty") == []


def test_recursive_chunking_invariants() -> None:
    text = (
        "# Heading One\n\n"
        "Paragraph one discusses virtual power plants. "
        "VPPs aggregate diverse distributed energy resources.\n\n"
        "# Heading Two\n\n"
        "Paragraph two discusses electrolytic hydrogen production. "
        "PEM electrolyzers feature fast ramping.\n\n"
        "Conclusion paragraph with summary remarks."
    )
    chunks = recursive(text, doc_id="doc_rec", chunk_size=120, overlap=30)
    assert len(chunks) >= 2

    for chunk in chunks:
        assert chunk.doc_id == "doc_rec"
        assert 0 <= chunk.start < chunk.end <= len(text)
        assert text[chunk.start : chunk.end] == chunk.text

    assert recursive("", doc_id="empty") == []


def test_semantic_chunking_invariants() -> None:
    text = (
        "Industrial demand response adjusts large factory power consumption. "
        "Large manufacturing plants shift heavy electricity usage during peak hours. "
        "Completely unrelated topic regarding biological taxonomy of deep ocean marine organisms. "
        "Marine biologists categorize benthic fauna according to genetic ancestry."
    )
    embedder = HashingEmbedder(dim=128)
    chunks = semantic(text, doc_id="doc_sem", threshold=0.1, embedder=embedder)
    assert len(chunks) >= 1

    for chunk in chunks:
        assert chunk.doc_id == "doc_sem"
        assert 0 <= chunk.start < chunk.end <= len(text)
        assert text[chunk.start : chunk.end] == chunk.text

    assert semantic("", doc_id="empty") == []
