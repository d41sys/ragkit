"""LegalBench-RAG-mini: size-matched chunk control (equal k vs equal character budget).

Protocol, split and the pre-registered rule: ``docs/legalsize_protocol.md`` (committed before any
number). The official metric counts k in chunks and sums overlapping ground-truth characters, so a
larger chunk size retrieves more characters at the same k. This module scores ``naive250``,
``naive500`` and ``naive1000`` chunks with BM25 and ``e5-small-v2``, at equal k and at equal
nominal character budget (``k = budget / size``), and measures how much of an equal-k difference
disappears at equal budget. Document matrices of ``e5-small-v2`` come from the cache or are
embedded once, offline, with the cached model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ragkit.analysis import lucene_english
from ragkit.beir import LUCENE_B, LUCENE_K1
from ragkit.bm25 import BM25Index
from ragkit.legalbench import (
    KS,
    SOURCES,
    LegalResult,
    MiniBenchmark,
    MiniTest,
    chunk_corpus,
)
from ragkit.legalsplit import (
    BASELINE,
    EXPECTED_BASELINE_R8,
    REPRODUCTION_TOLERANCE,
    Cell,
    evaluate,
)
from ragkit.neural import DenseSide, e5_small, embed_corpus

SIZES = (250, 500, 1000)
RETRIEVERS = ("bm25", "dense-e5")
BUDGETS = (1000, 2000, 4000, 8000, 16000)
K8 = KS.index(8)
# (large size, small size, equal budget B in characters); the first is the primary contrast.
CONTRASTS = ((1000, 500, 4000), (500, 250, 4000), (1000, 250, 8000))
PRIMARY_RETRIEVER = "bm25"
PRIMARY_CONTRAST = CONTRASTS[0]
BUDGET_EFFECT = "chunk-size gains on this metric are mostly a budget effect"
Key = tuple[str, int]  # (retriever, chunk size)


def chunker_name(size: int) -> str:
    """``naive<size>``."""
    return f"naive{size}"


def budget_k(budget: int, size: int) -> int:
    """Number of ``size``-character chunks that fill ``budget`` characters (must be in ``KS``)."""
    k, rest = divmod(budget, size)
    if rest or k not in KS:
        raise ValueError(f"budget {budget} is not k * {size} with k in {KS}")
    return k


# ---------------------------------------------------------------------------------------------
# Cells and scoring
# ---------------------------------------------------------------------------------------------


def build_cells(
    bench: MiniBenchmark, cache_dir: Path, device: str | None = None
) -> dict[Key, Cell]:
    """Six cells; missing e5 document matrices are embedded once (offline) and cached."""
    embedder = e5_small(device)
    cells: dict[Key, Cell] = {}
    for size in SIZES:
        chunks = chunk_corpus(bench.corpus, chunker_name(size))
        bm25 = BM25Index(k1=LUCENE_K1, b=LUCENE_B, analyzer=lucene_english)
        bm25.add(chunks)
        matrix = embed_corpus(embedder, [c.text for c in chunks], cache_dir)
        if len(matrix) != len(chunks):
            raise ValueError(f"{len(matrix)} embedded rows for {len(chunks)} chunks")
        cells[("bm25", size)] = Cell("bm25", chunker_name(size), chunks, bm25=bm25)
        cells[("dense-e5", size)] = Cell(
            "dense-e5", chunker_name(size), chunks, dense=DenseSide(embedder, matrix)
        )
    return cells


def run_cells(
    cells: dict[Key, Cell], tests: Sequence[MiniTest], indices: Sequence[int]
) -> dict[Key, LegalResult]:
    """Full-query, unrouted scores of every cell on ``tests[i]``, ``i`` in ``indices``."""
    return {key: evaluate(cell, tests, indices, BASELINE).result for key, cell in cells.items()}


# ---------------------------------------------------------------------------------------------
# Statistics and the rule
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """A point estimate with a 95 % percentile-bootstrap interval."""

    mean: float
    lo: float
    hi: float

    def fmt(self, signed: bool = True) -> str:
        """``mean [lo, hi]`` with two decimals."""
        sign = "+" if signed else ""
        return f"{self.mean:{sign}.2f} [{self.lo:{sign}.2f}, {self.hi:{sign}.2f}]"


Pair = tuple[LegalResult, int, LegalResult, int]  # (a, k_a, b, k_b): Recall@k_a(a) - Recall@k_b(b)


def paired_boot(
    pairs: Sequence[Pair], *, n_resamples: int = 2000, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Point differences (points) and their bootstrap draws ``(n_resamples, len(pairs))``.

    Queries are resampled within each source, one resample shared by every pair (so a derived
    quantity such as a ratio of two differences is bootstrapped consistently), and the four
    source means are averaged with equal weight (as ``LegalResult.all_ci``).
    """
    rng = np.random.default_rng(seed)
    boot = np.zeros((n_resamples, len(pairs)))
    point = np.zeros(len(pairs))
    for source in SOURCES:
        n = len(pairs[0][0].recall[source])
        idx = rng.integers(0, n, size=(n_resamples, n))
        for j, (a, ka, b, kb) in enumerate(pairs):
            diff = (
                np.asarray(a.recall[source])[:, KS.index(ka)]
                - np.asarray(b.recall[source])[:, KS.index(kb)]
            )
            boot[:, j] += diff[idx].mean(axis=1) / len(SOURCES)
            point[j] += float(diff.mean()) / len(SOURCES)
    return 100.0 * point, 100.0 * boot


def interval(point: float, draws: np.ndarray) -> Interval:
    """``point`` with the 2.5 / 97.5 percentiles of ``draws``."""
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return Interval(point, float(lo), float(hi))


def size_rule(d_k: Interval, d_b: Interval) -> str:
    """The pre-registered rule on the equal-k (``D_8``) and equal-budget (``D_B``) differences."""
    if d_k.lo <= 0.0:
        return "no detectable equal-k gain to explain"
    if d_b.lo <= 0.0 <= d_b.hi:
        return BUDGET_EFFECT
    if d_b.lo > 0.0:
        return f"size effect at equal budget: larger chunks help beyond their budget ({d_b.fmt()})"
    return f"size effect at equal budget: smaller chunks win per retrieved character ({d_b.fmt()})"


@dataclass(frozen=True)
class SizeContrast:
    """Equal-k and equal-budget difference of two chunk sizes in one retriever."""

    retriever: str
    large: int
    small: int
    budget: int
    d_k: Interval  # Recall@8(large) - Recall@8(small)
    d_b: Interval  # Recall@(budget/large)(large) - Recall@(budget/small)(small)
    share: Interval | None  # 1 - d_b / d_k; None when some bootstrap d_k is not above 0
    rule: str


def size_contrast(
    results: dict[Key, LegalResult], retriever: str, large: int, small: int, budget: int
) -> SizeContrast:
    """Both differences, the share of the equal-k one removed at equal budget, and the rule."""
    big, little = results[(retriever, large)], results[(retriever, small)]
    pairs: list[Pair] = [
        (big, 8, little, 8),
        (big, budget_k(budget, large), little, budget_k(budget, small)),
    ]
    point, boot = paired_boot(pairs)
    d_k = interval(float(point[0]), boot[:, 0])
    d_b = interval(float(point[1]), boot[:, 1])
    share: Interval | None = None
    if point[0] > 0.0 and bool((boot[:, 0] > 0.0).all()):
        share = interval(1.0 - float(point[1] / point[0]), 1.0 - boot[:, 1] / boot[:, 0])
    return SizeContrast(retriever, large, small, budget, d_k, d_b, share, size_rule(d_k, d_b))


def budget_curve(
    results: dict[Key, LegalResult], retriever: str, large: int, small: int
) -> list[tuple[int, Interval]]:
    """``Recall(large at B) - Recall(small at B)`` for every budget valid for both sizes."""
    big, little = results[(retriever, large)], results[(retriever, small)]
    budgets = [b for b in BUDGETS if b % large == 0 and b % small == 0]
    pairs: list[Pair] = [(big, budget_k(b, large), little, budget_k(b, small)) for b in budgets]
    point, boot = paired_boot(pairs)
    return [(b, interval(float(point[j]), boot[:, j])) for j, b in enumerate(budgets)]


# ---------------------------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------------------------


def _counts(bench: MiniBenchmark, indices: Sequence[int]) -> str:
    return ", ".join(
        f"{s} {sum(1 for i in indices if bench.tests[i].source == s)}" for s in SOURCES
    )


def _pct(result: LegalResult, metric: str, k: int) -> float:
    return 100.0 * result.all_mean(metric)[KS.index(k)]


def render_report(
    title: str,
    bench: MiniBenchmark,
    cells: dict[Key, Cell],
    results: dict[Key, LegalResult],
    indices: Sequence[int],
    full: dict[Key, LegalResult],
    *,
    verdict: bool,
) -> str:
    """Markdown report; ``verdict`` adds the pre-registered rule line (test split only)."""
    lines = [
        title,
        "",
        f"{len(indices)} queries ({_counts(bench, indices)}); 72 documents pooled; full query, "
        "no routing; ALL = equal-weight mean of the four source means; values in %. k is a "
        "number of chunks; the nominal character budget is k x size. Differences are 95% paired "
        "percentile-bootstrap intervals (2000 resamples, seed 0, queries resampled within each "
        "source, one resample shared by all quantities).",
        "",
        "| cell | chunks | shorter than size |",
        "|---|---|---|",
    ]
    for (retriever, size), cell in cells.items():
        short = sum(1 for c in cell.chunks if len(c.text) < size)
        lines.append(f"| {retriever}-{chunker_name(size)} | {len(cell.chunks):,} | {short} |")
    for retriever in RETRIEVERS:
        lines += [
            "",
            f"#### {retriever}: equal k (rows: chunk size; Recall@k ALL with 95% CI, then "
            "Precision@k)",
            "",
            "| size | " + " | ".join(f"R@{k}" for k in KS) + " |",
            "|---|" + "---|" * len(KS),
        ]
        for size in SIZES:
            res = results[(retriever, size)]
            cis = res.all_ci("recall")
            row = " | ".join(
                f"{_pct(res, 'recall', k):.2f} [{100 * lo:.1f}, {100 * hi:.1f}]"
                for k, (lo, hi) in zip(KS, cis, strict=True)
            )
            lines.append(f"| {size} | {row} |")
        lines += [
            "",
            "| size | " + " | ".join(f"P@{k}" for k in KS) + " |",
            "|---|" + "---|" * len(KS),
        ]
        for size in SIZES:
            row = " | ".join(f"{_pct(results[(retriever, size)], 'precision', k):.2f}" for k in KS)
            lines.append(f"| {size} | {row} |")
        lines += [
            "",
            f"#### {retriever}: equal character budget (cell = Recall ALL at k = B / size, "
            "(k) shown)",
            "",
            "| size | " + " | ".join(f"B = {b}" for b in BUDGETS) + " |",
            "|---|" + "---|" * len(BUDGETS),
        ]
        for size in SIZES:
            res = results[(retriever, size)]
            row = " | ".join(
                f"{_pct(res, 'recall', budget_k(b, size)):.2f} ({budget_k(b, size)})"
                for b in BUDGETS
            )
            lines.append(f"| {size} | {row} |")
        lines += [
            "",
            "| size | " + " | ".join(f"P at B = {b}" for b in BUDGETS) + " |",
            "|---|" + "---|" * len(BUDGETS),
        ]
        for size in SIZES:
            res = results[(retriever, size)]
            row = " | ".join(f"{_pct(res, 'precision', budget_k(b, size)):.2f}" for b in BUDGETS)
            lines.append(f"| {size} | {row} |")
        per_source = "; ".join(
            f"{size}: "
            + ", ".join(
                f"{s} {100 * results[(retriever, size)].source_mean('recall', s)[col]:.2f}"
                for s in SOURCES
            )
            for size in SIZES
            for col in [KS.index(budget_k(4000, size))]
        )
        lines += ["", f"Per-source Recall at B = 4000 ({retriever}): {per_source}."]
    lines += [
        "",
        "#### Size contrasts (points of Recall ALL): D_8 = equal k = 8; D_B = equal budget B; "
        "S = 1 - D_B / D_8, the share of the equal-k difference that disappears at equal budget",
        "",
        "| retriever | large vs small | B | D_8 [CI] | D_B [CI] | S [CI] | rule |",
        "|---|---|---|---|---|---|---|",
    ]
    contrasts: dict[tuple[str, int, int], SizeContrast] = {}
    for retriever in RETRIEVERS:
        for large, small, budget in CONTRASTS:
            con = size_contrast(results, retriever, large, small, budget)
            contrasts[(retriever, large, small)] = con
            share = con.share.fmt(signed=False) if con.share is not None else "undefined"
            lines.append(
                f"| {retriever} | {large} vs {small} | {budget} | {con.d_k.fmt()} | "
                f"{con.d_b.fmt()} | {share} | {con.rule} |"
            )
    lines += [
        "",
        "#### Larger minus smaller size at every equal budget (points of Recall ALL, [CI])",
        "",
        "| retriever | large vs small | " + " | ".join(f"B = {b}" for b in BUDGETS) + " |",
        "|---|---|" + "---|" * len(BUDGETS),
    ]
    for retriever in RETRIEVERS:
        for large, small, _ in CONTRASTS:
            curve = dict(budget_curve(results, retriever, large, small))
            row = " | ".join(curve[b].fmt() if b in curve else "n/a" for b in BUDGETS)
            lines.append(f"| {retriever} | {large} vs {small} | {row} |")
    primary = contrasts[(PRIMARY_RETRIEVER, *PRIMARY_CONTRAST[:2])]
    if verdict:
        lines += [
            "",
            f"**Pre-registered verdict** (`{PRIMARY_RETRIEVER}`, naive{PRIMARY_CONTRAST[0]} vs "
            f"naive{PRIMARY_CONTRAST[1]}, B = {PRIMARY_CONTRAST[2]}, test): D_8 = "
            f"{primary.d_k.fmt()}, D_B = {primary.d_b.fmt()} -> **{primary.rule}**.",
        ]
    else:
        lines += [
            "",
            "Dev pilot: the rule line is not a verdict (nothing is chosen from dev); the same "
            f"computation on the primary contrast gives: {primary.rule}.",
        ]
    lines += [
        "",
        "**Harness check, post-hoc, all 776 queries (includes dev): Recall@8 ALL, %**",
        "",
        "| cell | recomputed | committed row | gap | status |",
        "|---|---|---|---|---|",
    ]
    for retriever in RETRIEVERS:
        expected = EXPECTED_BASELINE_R8[(retriever, "naive500")]
        got = _pct(full[(retriever, 500)], "recall", 8)
        status = "reproduced" if abs(got - expected) <= REPRODUCTION_TOLERANCE else "NOT reproduced"
        lines.append(
            f"| {retriever}-naive500 | {got:.2f} | {expected:.2f} | {got - expected:+.2f} | "
            f"{status} |"
        )
    return "\n".join(lines)
