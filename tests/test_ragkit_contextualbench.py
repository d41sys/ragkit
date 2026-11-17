"""K-R7 budget recall and decision rule (docs/contextual_protocol.md)."""

from __future__ import annotations

from ragkit.contextualbench import BUDGETS, LEGAL_SOURCES, Row, budget_recall, choose, decide
from ragkit.vocabbench import METRICS, Result


def test_budget_recall_counts_gold_inside_the_budget_only() -> None:
    assert budget_recall([("a", 0, 10)], [("a", 5, 15)], 100) == 0.5
    spans = [("a", 0, 10), ("a", 10, 20)]
    assert budget_recall(spans, [("a", 10, 20)], 10) == 0.0
    assert budget_recall(spans, [("a", 10, 20)], 15) == 0.5


def test_budget_recall_does_not_spend_budget_on_overlap_or_credit_other_files() -> None:
    spans = [("a", 0, 10), ("a", 5, 15)]
    assert budget_recall(spans, [("a", 10, 15)], 15) == 1.0
    assert budget_recall(spans, [("a", 10, 15)], 12) == 0.4
    assert budget_recall([("b", 0, 10)], [("a", 0, 10)], 100) == 0.0


def _row(mrr: list[float], legal: float, wrong: int = 0) -> Row:
    n = len(mrr)
    eu = Result({m: [1.0] * n for m in METRICS}, [i < wrong for i in range(n)])
    eu.scores["mrr"] = mrr
    return Row({"eu": eu}, {s: {b: [legal] * 4 for b in BUDGETS} for s in LEGAL_SOURCES})


BASE = _row([0.5, 0.6] * 20, 0.30, wrong=3)


def test_legal_loss_blocks_an_eu_gain() -> None:
    gain = _row([0.7, 0.8] * 20, 0.29)
    assert decide(BASE, gain).status == "not adopted (no evidence)"
    assert decide(BASE, _row([0.7, 0.8] * 20, 0.30)).status == "adopted"


def test_sc_ships_only_if_it_beats_s() -> None:
    rows = {"B0": BASE, "S": _row([0.7, 0.8] * 20, 0.3), "SC": _row([0.7, 0.81] * 20, 0.3)}
    verdicts = {k: decide(BASE, rows[k]) for k in ("S", "SC")}
    assert choose(rows, verdicts) == "SC"
    rows["SC"] = _row([0.7, 0.8] * 20, 0.3)
    verdicts = {k: decide(BASE, rows[k]) for k in ("S", "SC")}
    assert choose(rows, verdicts) == "S"
