"""Tests for Okapi BM25 implementation: hand-computed 3-doc example and invariants."""

import math

import pytest
from ragkit.bm25 import BM25Index, tokenize
from ragkit.chunking import Chunk


def test_bm25_hand_computed_three_doc_example() -> None:
    """Verify BM25 calculations against analytical hand-computed values.

    Corpus (N = 3):
      Doc 0: "cat dog"        (len = 2, f(cat)=1, f(dog)=1)
      Doc 1: "cat mouse"      (len = 2, f(cat)=1, f(mouse)=1)
      Doc 2: "dog dog dog"    (len = 3, f(dog)=3)

    Stats:
      avgdl = (2 + 2 + 3) / 3 = 7 / 3 ≈ 2.3333333
      df("cat") = 2, df("dog") = 2, df("mouse") = 1

    IDF formula: ln( (N - df + 0.5) / (df + 0.5) + 1.0 )
      IDF("cat") = ln( (3 - 2 + 0.5) / (2 + 0.5) + 1.0 ) = ln( 1.5/2.5 + 1.0 ) = ln(1.6) ≈ 0.4700036
      IDF("dog") = ln(1.6) ≈ 0.4700036

    For Query "cat dog" with k1 = 1.5, b = 0.75:
      Doc 0:
        dl = 2
        norm_len = 1 - 0.75 + 0.75 * (2 / (7/3)) = 0.25 + 0.75 * (6/7) = 25/28 ≈ 0.892857
        tf_weight(cat) = 1 * 2.5 / (1 + 1.5 * (25/28)) = 2.5 / 2.3392857 ≈ 1.068702
        tf_weight(dog) = 1.068702
        score(Doc 0) = ln(1.6) * 1.068702 + ln(1.6) * 1.068702 ≈ 1.0045876

      Doc 1:
        score(Doc 1) = ln(1.6) * 1.068702 ≈ 0.5022938

      Doc 2:
        dl = 3, f(dog) = 3
        norm_len = 1 - 0.75 + 0.75 * (3 / (7/3)) = 0.25 + 0.75 * (9/7) = 8.5/7 ≈ 1.2142857
        tf_weight(dog) = 3 * 2.5 / (3 + 1.5 * (8.5/7)) = 7.5 / 4.82142857 = 14/9 ≈ 1.5555555
        score(Doc 2) = ln(1.6) * (14/9) ≈ 0.7311168

    Expected ranking: Doc 0 > Doc 2 > Doc 1.
    """
    bm = BM25Index(k1=1.5, b=0.75)

    c0 = Chunk(id="d0", doc_id="d0", text="cat dog", start=0, end=7)
    c1 = Chunk(id="d1", doc_id="d1", text="cat mouse", start=0, end=9)
    c2 = Chunk(id="d2", doc_id="d2", text="dog dog dog", start=0, end=11)

    bm.add([c0, c1, c2])

    assert bm.avgdl == pytest.approx(7.0 / 3.0, rel=1e-5)
    assert bm.idf("cat") == pytest.approx(math.log(1.6), rel=1e-5)
    assert bm.idf("dog") == pytest.approx(math.log(1.6), rel=1e-5)
    assert bm.idf("mouse") == pytest.approx(math.log(2.666666666666667), rel=1e-5)

    results = bm.search("cat dog", k=10)
    assert len(results) == 3

    r_chunks = [c.id for c, _ in results]
    assert r_chunks == ["d0", "d2", "d1"], "Expected ordering d0 > d2 > d1"

    # Verify scores against exact hand-calculated floats
    expected_s0 = 2.0 * math.log(1.6) * (2.5 / (1.0 + 1.5 * (25.0 / 28.0)))
    expected_s1 = math.log(1.6) * (2.5 / (1.0 + 1.5 * (25.0 / 28.0)))
    expected_s2 = math.log(1.6) * (14.0 / 9.0)

    assert results[0][1] == pytest.approx(expected_s0, rel=1e-5)
    assert results[1][1] == pytest.approx(expected_s2, rel=1e-5)
    assert results[2][1] == pytest.approx(expected_s1, rel=1e-5)


def test_bm25_edge_cases() -> None:
    bm = BM25Index()
    # Empty index search
    assert bm.search("test") == []

    # Unseen terms in query
    c = Chunk(id="d1", doc_id="d1", text="frequency containment reserve", start=0, end=30)
    bm.add([c])
    assert bm.search("solar panel") == []

    # Tokenizer behavior
    assert tokenize("Smart-Grid & Battery_Storage 2026!") == [
        "smart",
        "grid",
        "battery",
        "storage",
        "2026",
    ]
