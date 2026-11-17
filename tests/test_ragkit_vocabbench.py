"""Tests for vocabbench (offline, fake models and a fake LLM transport)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragkit.ctxprefix import build_chunks
from ragkit.embed import HashingEmbedder
from ragkit.lexical import Question
from ragkit.llm import CachedMessages, CacheMiss
from ragkit.neural import DenseSide
from ragkit.retrievalmiss import BASELINE as KR3_BASELINE
from ragkit.retrievalmiss import build_indexes
from ragkit.vocabbench import (
    BASELINE,
    BM25_QUERY_REPEATS,
    Cell,
    Engine,
    Tuning,
    expand,
    grid,
    queries,
    render_heldout,
    render_tuning,
    run_heldout,
    tune,
)

DOCS = [
    ("reg_2019_943_art16", "Capacity allocation and congestion management across bidding zones."),
    ("reg_2019_943_art17", "Allocation of congestion income from cross-zonal capacities."),
    ("reg_2017_2195_art19", "European platform for balancing energy from replacement reserves."),
    ("reg_2017_2195_art20", "Platform for balancing energy from frequency restoration reserves."),
]
SYN_DOCS = [
    ("doc_01_demand_response", "Demand response mechanisms and tariffs."),
    ("doc_02_battery_dispatch", "Battery dispatch strategies for frequency regulation."),
]


class FakeEmbedder:
    model_id = "fake@0"

    def __init__(self, texts: Sequence[str]) -> None:
        self.query_prefix, self.doc_prefix = "Represent: ", ""
        self.inner = HashingEmbedder(dim=128).fit(list(texts))

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.inner.embed(list(texts))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self.inner.embed_query(self.query_prefix + t) for t in texts])


class OverlapCE:
    scorer_id = "overlap"

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        terms = set(question.lower().split())
        return [float(len(terms & set(p.lower().split()))) for p in passages]


class LengthCE:
    """A different reranker: prefers short chunks."""

    scorer_id = "length"

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        return [-float(len(p)) for p in passages]


def engine(docs: list[tuple[str, str]]) -> Engine:
    chunks = build_chunks(docs, prefix=False)
    texts = [c.text for c in chunks]
    sides = {
        name: DenseSide(emb, emb.embed(texts))
        for name, emb in (("bge-small", FakeEmbedder(texts)), ("bge-base", FakeEmbedder(texts)))
    }
    return Engine(chunks, dict(docs), sides, {"minilm": OverlapCE(), "bge-reranker": LengthCE()})


def test_grid_and_cell_validation() -> None:
    cells = grid()
    assert len(cells) == len(set(cells)) == 12
    assert BASELINE in cells
    with pytest.raises(ValueError):
        Cell(ce_expanded=True)


def test_queries_follow_query2doc_and_fall_back_on_empty_passage() -> None:
    q, p = "congestion income", "Income from congestion shall be allocated."
    assert queries(BASELINE, q, p) == (q, q, q)
    bm25, dense, ce = queries(Cell(expansion=True), q, p)
    assert bm25.count(q) == BM25_QUERY_REPEATS and bm25.endswith(p)
    assert dense == f"{q}\n{p}"
    assert ce == q
    assert queries(Cell(expansion=True, ce_expanded=True), q, p)[2] == dense
    assert queries(Cell(expansion=True, ce_expanded=True), q, "") == (q, q, q)


def test_baseline_reproduces_retrieval_miss_b0() -> None:
    """B0 here must be K-R3's B0 (the shipped hybrid+ce@100) on the same models."""
    texts = [c.text for c in build_chunks(DOCS, prefix=False)]
    emb = FakeEmbedder(texts)
    kr3 = build_indexes(DOCS, emb, emb, OverlapCE())["chunk"]
    ours = engine(DOCS)
    ours.sides["bge-small"] = DenseSide(emb, emb.embed(texts))
    for q in ("congestion income allocation", "balancing energy platform reserves"):
        assert [i for i, _ in ours.rank(BASELINE, q, "")] == [
            i for i, _ in kr3.rank(KR3_BASELINE, q)
        ]


def test_expand_caches_and_replays_offline(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def transport(payload: dict[str, Any], key: str) -> dict[str, Any]:
        calls.append(payload)
        return {"content": [{"type": "text", "text": "  A passage.  "}], "usage": {}}

    path = tmp_path / "exp.jsonl"
    live = CachedMessages(path, api_key="k", model="m", temperature=0.0, transport=transport)
    out = expand(live, ["q1", "q2", "q1"])
    assert out == {"q1": "A passage.", "q2": "A passage."}
    assert len(calls) == 2
    offline = CachedMessages(path, model="m", temperature=0.0, transport=transport)
    assert expand(offline, ["q2"]) == {"q2": "A passage."}
    assert len(calls) == 2
    with pytest.raises(CacheMiss):
        expand(offline, ["unseen"])


def test_tune_and_heldout_end_to_end(tmp_path: Path) -> None:
    eu, syn = engine(DOCS), engine(SYN_DOCS)
    fresh = [Question("congestion income allocation", ("reg_2019_943_art17",), "")]
    syn_qs = [Question("battery frequency regulation", ("doc_02_battery_dispatch",), "")]
    held = {
        "original 42": [Question("capacity congestion zones", ("reg_2019_943_art16",), "")],
        "hard 39": [Question("replacement reserves platform", ("reg_2017_2195_art19",), "")],
    }
    passages = {q.text: "balancing energy reserves" for qs in held.values() for q in qs}
    tuning = tune(eu, syn, fresh, syn_qs, passages)
    assert tuning.chosen["M1"] == Cell(embedder="bge-base")
    assert tuning.chosen["M2"] == Cell(reranker="bge-reranker")
    q1 = tuning.chosen["Q1"]
    assert q1.expansion and q1.embedder == "bge-small" and q1.reranker == "minilm"
    again = Tuning.from_json(tuning.to_json())
    assert again.chosen == tuning.chosen
    assert "Chosen:" in render_tuning(tuning)

    run = run_heldout(eu, syn, held, syn_qs, passages, tuning)
    assert run.rows["B0"].cell == BASELINE
    assert len(run.grid_rows) == 12
    assert run.fallbacks == 0
    pooled = run.rows["B0"].pooled()
    assert len(pooled.scores["pack"]) == 2
    report = render_heldout(run)
    assert "Decision rule" in report and "Verdict" in report
