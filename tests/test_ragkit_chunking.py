"""Tests for RagKit chunking strategies: boundary invariants and text reconstruction."""

import itertools
from pathlib import Path

import pytest

from ragkit.chunking import Chunk, fixed, recursive, semantic, sentence, structured
from ragkit.embed import HashingEmbedder
from ragkit.eurlex import strip_metadata_comment

EU_CORPUS = Path(__file__).parent.parent / "src" / "ragkit" / "data" / "corpus_eu"


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


ARTICLE = (
    "# Regulation (EU) 9999/1 (test act), Article 7: Active customers\n"
    "\n"
    "1. Member States shall ensure that customers may act as active customers without "
    "disproportionate requirements and without discriminatory charges of any kind.\n"
    "\n"
    "2. Member States shall ensure that active customers are:\n"
    "\n"
    "(a) entitled to operate directly or through aggregation;\n"
    "\n"
    "(b) entitled to sell self-generated electricity;\n"
    "\n"
    "3. Short.\n"
    "\n"
    "4. Also short."
)


def test_structured_keeps_heading_paragraphs_and_points() -> None:
    chunks = structured(ARTICLE, "art7", min_chars=100, max_chars=400)
    title = "Regulation (EU) 9999/1 (test act), Article 7: Active customers"
    assert [c.id for c in chunks] == ["art7#struct_0", "art7#struct_1", "art7#struct_2"]
    assert all(c.title == title and c.doc_id == "art7" for c in chunks)
    # Paragraph 1 stands alone, paragraph 2 keeps its (a) and (b) points, 3 and 4 merge.
    bodies = [ARTICLE[c.start : c.end] for c in chunks]
    assert bodies[0].startswith("1. Member States") and "2." not in bodies[0]
    assert bodies[1].startswith("2. Member States") and "(a)" in bodies[1] and "(b)" in bodies[1]
    assert "3. Short." not in bodies[1]
    assert bodies[2] == "3. Short.\n\n4. Also short."
    assert all(c.text == f"{title}\n\n{b}" for c, b in zip(chunks, bodies, strict=True))


def test_structured_without_prepended_heading_is_the_exact_body_span() -> None:
    chunks = structured(ARTICLE, "art7", min_chars=100, max_chars=400, prepend_heading=False)
    assert all(c.text == ARTICLE[c.start : c.end] for c in chunks)
    assert all(c.title.endswith("Active customers") for c in chunks)
    assert all("Article 7" not in c.text for c in chunks)


def test_structured_splits_long_unit_at_points_and_long_block_recursively() -> None:
    points = "\n\n".join(f"({chr(97 + i)}) {'word ' * 20}".strip() for i in range(6))
    text = f"# T\n\n1. Intro line:\n\n{points}\n\n2. {'long ' * 100}".strip()
    chunks = structured(text, "d", min_chars=50, max_chars=250, prepend_heading=False)
    assert len(chunks) > 3
    for c in chunks:
        assert text[c.start : c.end] == c.text
        assert len(c.text) <= 250
    # Point-level splits never cut a point in half: every chunk but the long block's pieces
    # starts at a point or paragraph marker.
    first = chunks[0].text
    assert first.startswith("1. Intro line:") and "(a)" in first
    assert all(chunks[i].start < chunks[i + 1].start for i in range(len(chunks) - 1))


def test_structured_without_heading_and_edge_cases() -> None:
    chunks = structured("1. Only body here.\n\n2. And more body.", "d", min_chars=5, max_chars=40)
    assert [c.title for c in chunks] == ["", ""]
    assert chunks[0].text == "1. Only body here."
    assert structured("", "d") == []
    assert structured("# Just a heading\n", "d") == []
    with pytest.raises(ValueError, match="min_chars"):
        structured("x", "d", min_chars=0)
    with pytest.raises(ValueError, match="min_chars"):
        structured("x", "d", min_chars=100, max_chars=10)


def test_structured_covers_every_eu_article() -> None:
    paths = sorted(EU_CORPUS.glob("*.md"))
    assert len(paths) == 94
    for path in paths:
        text = strip_metadata_comment(path.read_text(encoding="utf-8"))
        chunks = structured(text, path.stem, prepend_heading=False)
        assert chunks, path.stem
        assert all(text[c.start : c.end] == c.text for c in chunks)
        assert all(c.title.startswith(("Regulation", "Directive", "Commission")) for c in chunks)
        # Non-overlapping, in order.
        assert all(a.end <= b.start for a, b in itertools.pairwise(chunks))
