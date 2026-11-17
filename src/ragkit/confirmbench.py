"""Confirmation of a larger reranker, with and without LLM expansion, on unseen data (K-R6).

Protocol, systems and decision rule: ``docs/confirm_protocol.md`` (committed before this code).
Every system is ``vocabbench``'s ``hybrid+ce@100`` engine; BEIR units are whole documents scored
by nDCG@10, EU units are the K-R3 chunks scored as in K-R5. Expansions come from a call cache.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from ragkit.beir import BeirData, ndcg_cut, recall_at
from ragkit.eval import paired_bootstrap_ci
from ragkit.lexical import Question
from ragkit.modelsignals import PassageScorer
from ragkit.vocabbench import (
    BASELINE,
    EU_DOMAIN,
    EXPANSION_TEMPLATE,
    METRIC_LABELS,
    METRICS,
    TIE,
    Cell,
    Engine,
    Result,
    _ci,
    _delta,
    score_set,
)

BEIR_SETS = ("scifact", "nfcorpus")
DOMAINS = {
    "eu": EU_DOMAIN,
    "synthetic": EU_DOMAIN,
    "scifact": "the abstract of a scientific research paper",
    "nfcorpus": "a medical or nutrition research article",
}
SYSTEMS = {
    "B0": BASELINE,
    "R": Cell(reranker="bge-reranker"),
    "C": Cell(reranker="bge-reranker", expansion=True),
}
CANDIDATES = ("R", "C")


def expansion_system(corpus: str) -> str:
    """The frozen expansion prompt with the domain phrase of ``corpus``."""
    return EXPANSION_TEMPLATE.format(domain=DOMAINS[corpus])


@dataclass
class BeirResult:
    """Per-query nDCG@10 and Recall@100 of one system on one BEIR test set."""

    ndcg: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)


def score_beir(
    engine: Engine, cell: Cell, data: BeirData, passages: Mapping[str, str]
) -> BeirResult:
    """Scores of ``cell`` on every test query of ``data`` (qrels order)."""
    items = list(data.queries.items())
    engine.prepare([cell], [text for _, text in items], passages)
    out = BeirResult()
    for qid, text in items:
        hits = engine.rank(cell, text, passages.get(text, ""))
        ranked = [engine.chunks[i].id for i, _ in hits]
        out.ndcg.append(ndcg_cut(ranked, data.qrels[qid]))
        out.recall.append(recall_at(ranked, data.qrels[qid]))
    return out


@dataclass
class Row:
    """One system on SciFact, NFCorpus, EU-new and synthetic, plus its cross-encoder time."""

    beir: dict[str, BeirResult]
    eu: Result
    synthetic: Result
    ce_ms_per_query: float = 0.0

    def pooled_ndcg(self) -> list[float]:
        return [v for name in BEIR_SETS for v in self.beir[name].ndcg]


def _mean_diff(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x - y for x, y in zip(a, b, strict=True)) / len(a)


@dataclass(frozen=True)
class Verdict:
    """Outcome of the decision rule for one candidate."""

    status: str
    d_pooled: tuple[float, float, float]
    points: dict[str, float]
    wrong: int
    base_wrong: int
    d_pack: float
    failed: list[str]


def decide(base: Row, cand: Row) -> Verdict:
    """The protocol's three conditions (pooled BEIR gain, no set below 0, EU serving context)."""
    d_pooled = paired_bootstrap_ci(cand.pooled_ndcg(), base.pooled_ndcg())
    points = {n: _mean_diff(cand.beir[n].ndcg, base.beir[n].ndcg) for n in BEIR_SETS}
    points["EU-new"] = _mean_diff(cand.eu.scores["mrr"], base.eu.scores["mrr"])
    wrong, base_wrong = sum(cand.eu.wrong_act), sum(base.eu.wrong_act)
    d_pack = _mean_diff(cand.eu.scores["pack"], base.eu.scores["pack"])
    failed = []
    if d_pooled[1] <= 0:
        failed.append(f"1: pooled BEIR nDCG@10 CI lower bound {d_pooled[1]:+.3f} <= 0")
    failed += [f"2: {n} difference {d:+.3f} < 0" for n, d in points.items() if d < -TIE]
    if wrong > base_wrong:
        failed.append(f"3: EU-new wrong-act {wrong} > {base_wrong}")
    if d_pack < -TIE:
        failed.append(f"3: EU-new pack recall difference {d_pack:+.3f} < 0")
    if not failed:
        status = "adopted"
    elif d_pooled[2] < 0:
        status = "rejected"
    else:
        status = "not adopted (no evidence)"
    return Verdict(status, d_pooled, points, wrong, base_wrong, d_pack, failed)


def run(
    engines: Mapping[str, Engine],
    beir: Mapping[str, BeirData],
    eu_qs: Sequence[Question],
    syn_qs: Sequence[Question],
    passages: Mapping[str, Mapping[str, str]],
) -> dict[str, Row]:
    """Score every system once on every set (B0 first, so its timing has a cold cache)."""
    rows: dict[str, Row] = {}
    for name, cell in SYSTEMS.items():
        before = sum(engines[n].ce_seconds[cell.reranker] for n in BEIR_SETS)
        results = {n: score_beir(engines[n], cell, beir[n], passages[n]) for n in BEIR_SETS}
        spent = sum(engines[n].ce_seconds[cell.reranker] for n in BEIR_SETS) - before
        n_queries = sum(len(r.ndcg) for r in results.values())
        rows[name] = Row(
            results,
            score_set(engines["eu"], cell, eu_qs, passages["eu"]),
            score_set(engines["synthetic"], cell, syn_qs, passages["synthetic"]),
            1000.0 * spent / n_queries,
        )
    return rows


def cpu_ms(
    scorers: Mapping[str, PassageScorer], questions: Sequence[str], texts: Sequence[str]
) -> dict[str, float]:
    """Mean milliseconds per question for one call scoring all ``texts`` (protocol amendment)."""
    out: dict[str, float] = {}
    for name, scorer in scorers.items():
        scorer.score(questions[0], texts[:1])  # load the model outside the timing
        started = time.perf_counter()
        for q in questions:
            scorer.score(q, texts)
        out[name] = 1000.0 * (time.perf_counter() - started) / len(questions)
    return out


def render(
    rows: Mapping[str, Row],
    passages: Mapping[str, Mapping[str, str]],
    cpu: Mapping[str, float],
) -> str:
    """Per-set tables with paired differences, the verdicts and the non-gating rows."""
    base = rows["B0"]
    lines = ["### ragkit confirm score: K-R6 (docs/confirm_protocol.md)", ""]
    for name, cell in SYSTEMS.items():
        lines.append(f"- {name}: {cell.label()}")
    for n in BEIR_SETS:
        size = len(base.beir[n].ndcg)
        lines += ["", f"#### {n} test ({size} queries)", ""]
        lines += ["| system | nDCG@10 | difference vs B0 (better/worse/tied) | Recall@100 |"]
        lines += ["|---|---|---|---|"]
        for name, row in rows.items():
            r = row.beir[n]
            delta = "" if name == "B0" else _delta(r.ndcg, base.beir[n].ndcg)
            lines.append(f"| {name} | {_ci(r.ndcg)} | {delta} | {_ci(r.recall)} |")
    pooled = len(base.pooled_ndcg())
    lines += ["", f"#### pooled SciFact + NFCorpus ({pooled} queries)", ""]
    lines += ["| system | nDCG@10 | difference vs B0 (better/worse/tied) |", "|---|---|---|"]
    for name, row in rows.items():
        delta = "" if name == "B0" else _delta(row.pooled_ndcg(), base.pooled_ndcg())
        lines.append(f"| {name} | {_ci(row.pooled_ndcg())} | {delta} |")
    size = len(base.eu.wrong_act)
    lines += ["", f"#### EU-new ({size} questions, consolidated corpus)", ""]
    header = " | ".join(METRIC_LABELS[m] for m in METRICS)
    lines += [f"| system | {header} | wrong-act | MRR difference vs B0 |"]
    lines += ["|---|" + "---|" * (len(METRICS) + 2)]
    for name, row in rows.items():
        cells = " | ".join(_ci(row.eu.scores[m]) for m in METRICS)
        delta = "" if name == "B0" else _delta(row.eu.scores["mrr"], base.eu.scores["mrr"])
        lines.append(f"| {name} | {cells} | {sum(row.eu.wrong_act)} | {delta} |")
    lines += ["", "#### Verdicts", ""]
    for name in CANDIDATES:
        v = decide(base, rows[name])
        m, lo, hi = v.d_pooled
        points = ", ".join(f"{n} {d:+.3f}" for n, d in v.points.items())
        lines.append(
            f"- {name}: **{v.status}**. Pooled BEIR {m:+.3f} [{lo:+.3f}, {hi:+.3f}]; {points}; "
            f"EU-new wrong-act {v.wrong} vs {v.base_wrong}, pack recall {v.d_pack:+.3f}."
        )
        lines += [f"  - failed {f}" for f in v.failed]
    lines += ["", "#### Not gating", ""]
    syn = len(base.synthetic.wrong_act)
    for name, row in rows.items():
        mrr = row.synthetic.scores["mrr"]
        delta = "" if name == "B0" else f", {_delta(mrr, base.synthetic.scores['mrr'])}"
        lines.append(f"- synthetic {syn} MRR, {name}: {_ci(mrr)}{delta}")
    for corpus, by_text in passages.items():
        empty = sum(1 for p in by_text.values() if not p)
        lines.append(f"- expansion fallbacks (empty passage), {corpus}: {empty} / {len(by_text)}")
    for name, row in rows.items():
        note = " (shares R's cross-encoder cache, so understated)" if name == "C" else ""
        ms = f"{row.ce_ms_per_query:.0f} ms / query"
        lines.append(f"- cross-encoder time on BEIR, {name}: {ms}{note}")
    for reranker, ms in cpu.items():
        lines.append(f"- cross-encoder time on CPU, {reranker}: {ms:.0f} ms / question")
    return "\n".join(lines)
