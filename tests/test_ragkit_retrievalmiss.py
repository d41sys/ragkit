"""Tests for retrievalmiss (offline, fake models, no model library)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from ragkit.embed import HashingEmbedder
from ragkit.lexical import Question
from ragkit.retrievalmiss import (
    ALPHAS,
    BASELINE,
    CANDIDATE_IDS,
    DEPTHS,
    article_lists,
    article_windows,
    build_indexes,
    diagnose,
    grid,
    length_bounds,
    render_diagnosis,
    render_heldout,
    render_tuning,
    run_heldout,
    tune,
)

DOCS = [
    (
        "reg_2019_943_art16",
        "Grid capacity allocation and congestion management rules across European bidding zones.",
    ),
    (
        "reg_2019_943_art17",
        "Allocation of congestion income from cross-zonal transmission capacities.",
    ),
    (
        "reg_2017_2195_art19",
        "European platform for the exchange of balancing energy from replacement reserves.",
    ),
    (
        "reg_2017_2195_art20",
        "European platform for exchanging balancing energy from frequency restoration reserves.",
    ),
]


class FakeEmbedder:
    """Hashing embedder behind SentenceEmbedder surface."""

    model_id = "fake@0"

    def __init__(self, texts: Sequence[str], prefix: str = "") -> None:
        self.query_prefix = prefix
        self.doc_prefix = ""
        self.inner = HashingEmbedder(dim=128).fit(list(texts))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.inner.embed(list(texts))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.inner.embed_query(self.query_prefix + t) for t in texts])


class FakeCrossEncoder:
    """Term overlap as a relevance logit."""

    scorer_id = "fake-ce"

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        terms = set(question.lower().split())
        return [float(len(terms & set(p.lower().split()))) for p in passages]


@pytest.fixture
def fake_study() -> tuple[dict[str, Any], dict[str, Any]]:
    texts = [t for _, t in DOCS]
    on = FakeEmbedder(texts, prefix="Represent: ")
    off = FakeEmbedder(texts, prefix="")
    ce = FakeCrossEncoder()
    eu = build_indexes(DOCS, on, off, ce)
    synthetic_docs = [
        ("doc_01_demand_response", "Demand response mechanisms and tariffs."),
        ("doc_02_battery_dispatch", "Battery dispatch strategies for frequency regulation."),
    ]
    syn_texts = [t for _, t in synthetic_docs]
    syn_on = FakeEmbedder(syn_texts, prefix="Represent: ")
    syn_off = FakeEmbedder(syn_texts, prefix="")
    syn = build_indexes(synthetic_docs, syn_on, syn_off, ce)
    return eu, syn


def test_config_invariants() -> None:
    assert BASELINE.unit == "chunk"
    assert BASELINE.instruction is True
    assert BASELINE.alpha is None
    assert BASELINE.depth == 100
    assert BASELINE.stage == "hybrid+ce"
    assert len(grid()) == 2 * 2 * len(DEPTHS) * (1 + len(ALPHAS))


def test_article_windows_short_and_long() -> None:
    short = article_windows("hello world", "d1")
    assert short == ["hello world"]
    long_text = "word " * 600
    windows = article_windows(long_text, "d2")
    assert len(windows) > 1
    assert all(len(w) <= 1800 for w in windows)


def test_baseline_matches_neural_engine(fake_study: tuple[dict[str, Any], dict[str, Any]]) -> None:
    from ragkit.neural import NeuralEngine

    eu, _ = fake_study
    chunk_index = eu["chunk"]
    engine = NeuralEngine(
        chunk_index.units,
        chunk_index.bm25,
        {"bge": chunk_index.sides[True]},
        chunk_index.ce,
        depth=BASELINE.depth,
    )
    q = "congestion income rules"
    cands = engine.candidates(q)
    engine_hits = engine.system("hybrid+ce@100", q, cands)
    unit_hits = chunk_index.rank(BASELINE, q)
    assert [i for i, _ in unit_hits] == [i for i, _ in engine_hits]


def test_ranking_and_dedup(fake_study: tuple[dict[str, Any], dict[str, Any]]) -> None:
    eu, _ = fake_study
    q = "congestion income rules"
    chunk_hits = eu["chunk"].rank(BASELINE, q)
    art_hits = eu["article"].rank(BASELINE.at_stage("hybrid"), q)
    assert len(chunk_hits) <= BASELINE.depth
    assert len(art_hits) <= len(DOCS)
    listed, prod = article_lists(eu["chunk"], chunk_hits)
    assert len(listed) <= 10
    assert len(prod) <= 5
    assert len(prod) <= len(listed)


def test_tuning_and_heldout_offline(fake_study: tuple[dict[str, Any], dict[str, Any]]) -> None:
    eu, syn = fake_study
    fresh = [
        Question("congestion income allocation", ("reg_2019_943_art17",), "q"),
        Question("balancing energy platform replacement reserves", ("reg_2017_2195_art19",), "q"),
    ]
    syn_qs = [Question("tariffs for demand response", ("doc_01_demand_response",), "q")]
    tuning = tune(eu, syn, fresh, syn_qs)
    assert set(tuning.chosen) == set(CANDIDATE_IDS)
    rendered_tuning = render_tuning(tuning)
    assert "Chosen configuration" in rendered_tuning or "chosen configuration" in rendered_tuning

    heldout_sets = {
        "original 42": [
            Question("cross-zonal congestion management", ("reg_2019_943_art16",), "q")
        ],
        "hard 39": [Question("balancing frequency restoration", ("reg_2017_2195_art20",), "q")],
    }
    heldout = run_heldout(eu, syn, heldout_sets, syn_qs, tuning)
    assert "B0" in heldout.candidates
    rendered_heldout = render_heldout(heldout, tuning)
    assert "Decision rule" in rendered_heldout

    all_qs = [*fresh, *heldout_sets["original 42"], *heldout_sets["hard 39"]]
    bounds = length_bounds(all_qs)
    assert bounds[0] <= bounds[1]

    misses, sizes = diagnose(heldout_sets, eu["chunk"], eu["article"], DOCS, all_qs)
    rendered_diag = render_diagnosis(misses, sizes, bounds, focus="original 42")
    assert "### ragkit retrieval-miss: diagnosis" in rendered_diag
