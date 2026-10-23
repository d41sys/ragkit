"""Tests for the frozen synonym/acronym expansion (offline, no models)."""

from pathlib import Path

import pytest

from ragkit.bm25 import tokenize
from ragkit.expand import DEFAULT_EXPANSION_WEIGHT, SYNONYM_GROUPS, SynonymExpander, _contains

EU_CORPUS = Path(__file__).parent.parent / "src" / "ragkit" / "data" / "corpus_eu"


def test_acronym_expands_to_full_phrase_with_reduced_weight() -> None:
    terms = SynonymExpander().expand("Which TSO pays?")
    assert terms[:3] == [("which", 1.0), ("tso", 1.0), ("pays", 1.0)]
    expanded = dict(terms[3:])
    for tok in ("transmission", "system", "operator", "operators", "tsos"):
        assert expanded[tok] == DEFAULT_EXPANSION_WEIGHT


def test_phrase_expands_to_acronym_and_is_case_and_hyphen_insensitive() -> None:
    expander = SynonymExpander()
    assert ("tso", DEFAULT_EXPANSION_WEIGHT) in expander.expand("transmission system operator")
    assert ("zonal", DEFAULT_EXPANSION_WEIGHT) in expander.expand("CROSS-BORDER capacity")


def test_no_match_returns_original_tokens_unchanged() -> None:
    assert SynonymExpander().expand("solar panels on roofs") == [
        ("solar", 1.0),
        ("panels", 1.0),
        ("on", 1.0),
        ("roofs", 1.0),
    ]


def test_original_terms_keep_multiplicity_and_are_never_re_added() -> None:
    terms = SynonymExpander().expand("tso tso transmission system operator")
    assert [t for t, w in terms if w == 1.0] == ["tso", "tso", "transmission", "system", "operator"]
    expanded = [t for t, w in terms if w < 1.0]
    assert "tso" not in expanded and "transmission" not in expanded
    assert len(expanded) == len(set(expanded))


def test_custom_groups_and_weight() -> None:
    expander = SynonymExpander(groups=[("fast reserve", "FCR")], weight=0.25)
    assert expander.expand("fast reserve") == [("fast", 1.0), ("reserve", 1.0), ("fcr", 0.25)]
    assert expander.expand("fcr")[1:] == [("fast", 0.25), ("reserve", 0.25)]


def test_weight_must_be_non_negative() -> None:
    with pytest.raises(ValueError):
        SynonymExpander(weight=-0.1)


def test_contains_requires_a_contiguous_run() -> None:
    assert _contains(["a", "b", "c"], ("b", "c"))
    assert not _contains(["a", "b", "c"], ("a", "c"))
    assert not _contains(["a"], ())


def test_every_group_has_a_member_attested_in_the_eu_corpus() -> None:
    """The map is grounded in corpus vocabulary: each expansion has a target in the corpus."""
    corpus = [tokenize(p.read_text(encoding="utf-8")) for p in sorted(EU_CORPUS.glob("*.md"))]
    assert corpus, "bundled EU corpus missing"
    for group in SYNONYM_GROUPS:
        assert len(group) >= 2
        attested = [m for m in group if any(_contains(doc, tokenize(m)) for doc in corpus)]
        assert attested, f"no member of {group} occurs in the EU corpus"
