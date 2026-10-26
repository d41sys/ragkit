"""Tests for the Lucene-like analyzer, the Porter stemmer and the posting-based BM25 scorer."""

import math

import pytest

from ragkit.analysis import lucene_english, lucene_length, lucene_stop, lucene_tokens, porter_stem
from ragkit.bm25 import BM25Index, top_k_indices
from ragkit.chunking import Chunk

PORTER_PAIRS = {
    "caresses": "caress",
    "ponies": "poni",
    "ties": "ti",
    "cats": "cat",
    "feed": "feed",
    "agreed": "agre",
    "plastered": "plaster",
    "motoring": "motor",
    "conflated": "conflat",
    "troubled": "troubl",
    "sized": "size",
    "hopping": "hop",
    "falling": "fall",
    "hissing": "hiss",
    "filing": "file",
    "happy": "happi",
    "sky": "sky",
    "relational": "relat",
    "conditional": "condit",
    "rational": "ration",
    "digitizer": "digit",
    "generalization": "gener",
    "oscillators": "oscil",
    "probate": "probat",
    "cease": "ceas",
    "controll": "control",
    "roll": "roll",
}


@pytest.mark.parametrize(("word", "stem"), sorted(PORTER_PAIRS.items()))
def test_porter_stem_known_pairs(word: str, stem: str) -> None:
    assert porter_stem(word) == stem


def test_porter_leaves_short_and_non_ascii_words() -> None:
    assert porter_stem("is") == "is"
    assert porter_stem("é") == "é"
    assert porter_stem("covid19") == "covid19"


def test_lucene_tokens_follow_word_boundaries() -> None:
    assert lucene_tokens("Don't stop; the U.S. economy's 3.5% rise, 1,000 jobs") == [
        "don't",
        "stop",
        "the",
        "u.s",
        "economy",
        "3.5",
        "rise",
        "1,000",
        "jobs",
    ]
    assert lucene_tokens("covid-19 test_case") == ["covid", "19", "test_case"]


def test_lucene_stop_and_english_chain() -> None:
    assert lucene_stop("The cats are in the garden") == ["cats", "garden"]
    assert lucene_english("The cats are running in the gardens") == ["cat", "run", "garden"]


def test_lucene_length_is_exact_below_24_and_lossy_above() -> None:
    assert [lucene_length(n) for n in (0, 1, 23)] == [0, 1, 23]
    assert lucene_length(100) == 96
    assert lucene_length(24) == 24
    for n in range(24, 5000):
        assert lucene_length(n) <= n
        assert n - lucene_length(n) <= n // 8


def _chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(id=f"d{i}", doc_id=f"d{i}", text=t, start=0, end=len(t)) for i, t in enumerate(texts)
    ]


def test_bm25_matches_formula_with_analyzer_and_repeated_query_terms() -> None:
    index = BM25Index(k1=0.9, b=0.4, analyzer=lucene_english)
    index.add(_chunks(["running dogs", "a dog and a cat", "cats cats cats"]))
    n, avgdl = 3, (2 + 2 + 3) / 3
    df_dog = 2
    idf = math.log((n - df_dog + 0.5) / (df_dog + 0.5) + 1.0)
    f, dl = 1, 2
    expected = idf * f * 1.9 / (f + 0.9 * (1 - 0.4 + 0.4 * dl / avgdl))
    result = dict((c.id, s) for c, s in index.search("dog dog", k=10))
    assert result["d0"] == pytest.approx(2 * expected)
    assert result["d1"] == pytest.approx(2 * expected)
    assert "d2" not in result


def test_bm25_length_fn_changes_only_the_length_normalisation() -> None:
    texts = [" ".join(["word"] * 100) + " target", "target"]
    exact = BM25Index()
    lossy = BM25Index(length_fn=lucene_length)
    for index in (exact, lossy):
        index.add(_chunks(texts))
    assert exact.avgdl == lossy.avgdl
    assert exact.score_all("target")[1] == lossy.score_all("target")[1]
    assert exact.score_all("target")[0] != lossy.score_all("target")[0]


def test_top_k_indices_breaks_ties_by_index_and_skips_zero_scores() -> None:
    import numpy as np

    scores = np.array([0.0, 2.0, 2.0, 3.0, 2.0, 0.0, 1.0])
    assert top_k_indices(scores, 3) == [(3, 3.0), (1, 2.0), (2, 2.0)]
    assert top_k_indices(scores, 10) == [(3, 3.0), (1, 2.0), (2, 2.0), (4, 2.0), (6, 1.0)]
    assert top_k_indices(np.zeros(4), 3) == []
    assert top_k_indices(scores, 0) == []
