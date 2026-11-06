"""LegalBench-RAG-mini size-matched chunk control (docs/legalsize_protocol.md)."""

from __future__ import annotations

import itertools
import zlib

import numpy as np
import pytest

from ragkit.analysis import lucene_english
from ragkit.bm25 import BM25Index
from ragkit.legalbench import KS, SOURCES, LegalResult, MiniBenchmark, MiniTest, chunk_corpus
from ragkit.legalsize import (
    BUDGET_EFFECT,
    SIZES,
    Interval,
    budget_curve,
    budget_k,
    paired_boot,
    render_report,
    run_cells,
    size_contrast,
    size_rule,
)
from ragkit.legalsplit import Cell
from ragkit.neural import DenseSide


def test_naive_chunkers_of_any_size_cover_the_text_without_overlap() -> None:
    corpus = {"d.txt": "abcdefghij" * 30}  # 300 characters
    for size, count in ((250, 2), (100, 3), (7, 43)):
        chunks = chunk_corpus(corpus, f"naive{size}")
        assert len(chunks) == count
        assert chunks[0].start == 0 and chunks[-1].end == 300
        assert all(a.end == b.start for a, b in itertools.pairwise(chunks))
        assert "".join(c.text for c in chunks) == corpus["d.txt"]
    for bad in ("naive", "naive0", "naive05x", "semantic"):
        with pytest.raises(ValueError):
            chunk_corpus(corpus, bad)


def test_budget_k_scales_k_with_size_and_rejects_off_grid_budgets() -> None:
    assert [budget_k(4000, s) for s in SIZES] == [16, 8, 4]
    assert [budget_k(1000, s) for s in SIZES[1:]] == [2, 1]
    assert budget_k(16000, 250) == 64
    with pytest.raises(ValueError):
        budget_k(1500, 500)  # k = 3 is not on the grid
    with pytest.raises(ValueError):
        budget_k(500, 1000)  # k = 0
    with pytest.raises(ValueError):
        budget_k(32000, 250)  # k = 128, deeper than the grid


def _result(fn, n: int = 30) -> LegalResult:
    """Per-query recall ``fn(query_index, k)`` clipped to 1, identical in every source."""
    res = LegalResult("synthetic")
    for source in SOURCES:
        rows = [[min(1.0, fn(q, k)) for k in KS] for q in range(n)]
        res.recall[source] = rows
        res.precision[source] = [[0.0] * len(KS) for _ in range(n)]
    return res


def test_a_pure_budget_effect_is_detected_and_fully_removed_at_equal_budget() -> None:
    # recall depends on the retrieved characters only: k * size
    def make(size: int) -> LegalResult:
        return _result(lambda q, k: (0.2 + 0.01 * q) * k * size / 8000)

    results = {("bm25", s): make(s) for s in SIZES}
    con = size_contrast(results, "bm25", 1000, 500, 4000)
    assert con.d_k.lo > 0.0  # a gain at equal k exists
    assert con.d_b.mean == pytest.approx(0.0, abs=1e-9)
    assert con.d_b.lo <= 0.0 <= con.d_b.hi
    assert con.rule == BUDGET_EFFECT
    assert con.share is not None and con.share.mean == pytest.approx(1.0)
    for _, diff in budget_curve(results, "bm25", 1000, 500):
        assert diff.mean == pytest.approx(0.0, abs=1e-9)


def test_a_real_size_effect_survives_equal_budget_and_is_not_called_a_budget_effect() -> None:
    def make(size: int) -> LegalResult:
        return _result(lambda q, k: (0.1 + 0.005 * q) * (k * size / 4000) + 0.1 * (size == 1000))

    results = {("dense-e5", s): make(s) for s in SIZES}
    con = size_contrast(results, "dense-e5", 1000, 500, 4000)
    assert con.d_b.lo > 0.0
    assert con.rule.startswith("size effect at equal budget: larger chunks help")
    assert con.share is not None and con.share.mean < 1.0
    # smaller chunks win per character: sign flips
    reverse = {
        ("dense-e5", s): _result(lambda q, k, s=s: (0.1 + 0.005 * q) * k * (s**-0.5)) for s in SIZES
    }
    rev = size_contrast(reverse, "dense-e5", 1000, 500, 4000)
    assert rev.d_b.hi < 0.0 and rev.d_k.mean < 0.0
    assert rev.rule == "no detectable equal-k gain to explain"
    assert rev.share is None


def test_size_rule_branches() -> None:
    gain = Interval(5.0, 3.0, 7.0)
    assert size_rule(gain, Interval(0.2, -0.6, 1.0)) == BUDGET_EFFECT
    assert "larger chunks help" in size_rule(gain, Interval(2.0, 1.0, 3.0))
    assert "smaller chunks win" in size_rule(gain, Interval(-2.0, -3.0, -1.0))
    assert size_rule(Interval(0.5, -0.5, 1.5), Interval(2.0, 1.0, 3.0)).startswith("no detectable")
    # an interval touching 0 includes 0
    assert size_rule(gain, Interval(1.0, 0.0, 2.0)) == BUDGET_EFFECT


def test_paired_boot_uses_one_resample_for_all_pairs_and_is_seed_fixed() -> None:
    a = _result(lambda q, k: 0.3 + 0.01 * q)
    b = _result(lambda q, k: 0.2 + 0.005 * q)
    pairs = [(a, 8, b, 8), (b, 8, a, 8)]
    point, boot = paired_boot(pairs, n_resamples=200)
    assert point[0] == pytest.approx(-point[1])
    assert np.allclose(boot[:, 0], -boot[:, 1])  # same resample, opposite sign
    again, boot2 = paired_boot(pairs, n_resamples=200)
    assert np.array_equal(boot, boot2) and np.array_equal(point, again)
    assert point[0] == pytest.approx(100 * (0.1 + 0.005 * 14.5))


class _FakeEmbedder:
    """Deterministic unit query vectors; the model is never loaded."""

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        rows = [np.random.default_rng(zlib.crc32(t.encode())).normal(size=8) for t in texts]
        arr = np.asarray(rows, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def test_end_to_end_report_on_a_toy_corpus() -> None:
    corpus: dict[str, str] = {}
    tests: list[MiniTest] = []
    for n, source in enumerate(SOURCES):
        path = f"{source}/doc{n}.txt"
        words = [f"w{n}x{i}" for i in range(1200)]  # distinct tokens, about 7 characters each
        corpus[path] = " ".join(words)
        for j in range(3):
            start = 100 + 2000 * j
            tests.append(
                MiniTest(
                    source, f"Consider doc{n}; {words[start // 7]}", ((path, start, start + 60),)
                )
            )
    bench = MiniBenchmark(tests=tests, corpus=corpus)
    cells: dict[tuple[str, int], Cell] = {}
    rng = np.random.default_rng(0)
    for size in SIZES:
        chunks = chunk_corpus(corpus, f"naive{size}")
        index = BM25Index(k1=0.9, b=0.4, analyzer=lucene_english)
        index.add(chunks)
        matrix = rng.normal(size=(len(chunks), 8)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        cells[("bm25", size)] = Cell("bm25", f"naive{size}", chunks, bm25=index)
        cells[("dense-e5", size)] = Cell(
            "dense-e5",
            f"naive{size}",
            chunks,
            dense=DenseSide(_FakeEmbedder(), matrix),  # type: ignore[arg-type]
        )
    indices = list(range(len(tests)))
    results = run_cells(cells, tests, indices)
    full = run_cells({k: c for k, c in cells.items() if k[1] == 500}, tests, indices)
    text = render_report("### toy", bench, cells, results, indices, full, verdict=True)
    for needle in (
        "equal character budget",
        "bm25: equal k",
        "dense-e5: equal k",
        "Pre-registered verdict",
        "Harness check",
        "| dense-e5 | 1000 vs 250 | 8000 |",
    ):
        assert needle in text
    assert f"| bm25-naive250 | {len(cells[('bm25', 250)].chunks):,} |" in text
    pilot = render_report("### toy", bench, cells, results, indices, full, verdict=False)
    assert "Pre-registered verdict" not in pilot and "Dev pilot" in pilot
