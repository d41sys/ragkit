"""Chunk pool and per-paper cap for paper corpora (P-4, docs/paper_pool_protocol.md).

A configuration is (pool, cap, slices) at the 36 000 budget: the papers of the top ``pool`` chunks
are candidates, and a paper longer than ``cap`` is packed as windows of that size. One
configuration is chosen on data already studied and scored once on fresh data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ragkit import openpapers as op
from ragkit import papersbench as pb
from ragkit._search import load_documents
from ragkit.eval import bootstrap_ci, paired_bootstrap_ci
from ragkit.parent import PACK_OVERFLOW, PAPER_BUDGET_TOKENS, Passage, pack_articles

POOLS = (10, 20, 30, 50)
CAPS: tuple[tuple[int | None, int], ...] = (
    (None, 1),
    (18_000, 1),
    (18_000, 3),
    (12_000, 1),
    (12_000, 3),
)
Config = tuple[int, int | None, int]
B36: Config = (10, None, 1)  # the shipped packing
CONFIGS: tuple[Config, ...] = tuple((k, w, s) for k in POOLS for w, s in CAPS)
BENCHMARKS = ("qasper", "mdaqa", "long")
TARGETS = ("mdaqa", "long")  # rule 1: a gain on either
MARGINS = {"qasper": -0.02, "mdaqa": -0.02, "long": -0.05}  # rule 2: no benchmark loses
TIE = 0.005
N_RESAMPLES = 10_000
DEV_PATH = Path("docs/sota-runs/paper_pool_dev.json")
FRESH_QA_PATH = pb.DATA_DIR / "qa_papers_forecasting_fresh.jsonl"
FRESH2_QA_PATH = pb.DATA_DIR / "qa_papers_forecasting_fresh2.jsonl"
CAP_SYSTEMS: dict[str, Config] = {"S30": (30, None, 1), "W18": (30, 18_000, 1)}  # P-5
METRICS = {
    "primary": "primary",
    "all": "all gold papers packed",
    "packed": "papers packed",
    "tokens": "tokens",
}
Rows = Mapping[str, Mapping[Any, Mapping[str, Sequence[float]]]]  # bench -> system -> metric


def label(config: Config) -> str:
    pool, cap, slices = config
    return f"pool {pool}, " + (f"cap {cap} x {slices}" if cap else "no cap")


def datasets(stage: str) -> dict[str, list[tuple[op.Docs, list[Any]]]]:
    """Corpora and questions per benchmark; MDA-QA test is two pools, each with its own corpus."""
    long_docs = load_documents(pb.CORPUS_DIR)
    if stage == "dev":
        return {
            "qasper": [op.qasper("test")],
            "mdaqa": [op.mdaqa("test")],
            "long": [(long_docs, pb.load_questions())],
        }
    fresh = FRESH2_QA_PATH if stage == "cap" else FRESH_QA_PATH
    return {
        "qasper": [op.qasper_train(op.ANSWER_PAPERS, 2 * op.ANSWER_PAPERS)],
        "mdaqa": [op.mdaqa("pool_a"), op.mdaqa("pool_b")],
        "long": [(long_docs, pb.load_questions(fresh))],
    }


def pack(pool: Sequence[Passage], documents: Mapping[str, str], config: Config) -> list[Passage]:
    k, cap, slices = config
    return pack_articles(
        pool[:k],
        documents,
        budget_tokens=PAPER_BUDGET_TOKENS,
        overflow=PACK_OVERFLOW,
        window_tokens=cap,
        slices=slices,
    )


def measure(q: Any, ranked_papers: Sequence[str], packed: Sequence[Passage]) -> dict[str, float]:
    """One question: evidence recall (QASPER, long PDFs) or share of needed papers (MDA-QA)."""
    if isinstance(q, op.OpenQuestion):
        got = op.measure(q, ranked_papers, packed)
    else:
        long = pb.measure(q, ranked_papers, packed)
        got = {**long, "primary": long["evidence"], "all": float(long["papers"] == 1.0)}
    return {m: got[m] for m in METRICS}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def dev_scores(rows: Rows) -> dict[Config, float]:
    """Mean of the three dev primary metrics per configuration."""
    return {c: _mean([_mean(rows[b][c]["primary"]) for b in BENCHMARKS]) for c in CONFIGS}


def choose(scores: Mapping[Config, float]) -> Config:
    """The simplest configuration within ``TIE`` of the best dev score.

    Simpler: fewer settings that differ from B36 (pool, cap), then the smaller pool, then the
    order of ``CAPS``.
    """
    best = max(scores.values())
    by_simplicity = sorted(
        CONFIGS,
        key=lambda c: ((c[0] != B36[0]) + (c[1] is not None), c[0], CAPS.index((c[1], c[2]))),
    )
    return next(c for c in by_simplicity if scores[c] >= best - TIE)


def systems(chosen: Config) -> dict[str, Config]:
    """B36, C and the halves of C that differ from both (not gating)."""
    out = {"B36": B36, "C": chosen}
    halves = {"C pool only": (chosen[0], None, 1), "C cap only": (B36[0], chosen[1], chosen[2])}
    for name, config in halves.items():
        if config not in out.values():
            out[name] = config
    return out


def verdict(rows: Rows) -> str:
    """Rule 1 (a gain on MDA-QA or the long PDFs) and rule 2 (no benchmark loses)."""
    low = {
        b: paired_bootstrap_ci(
            rows[b]["C"]["primary"], rows[b]["B36"]["primary"], n_resamples=N_RESAMPLES
        )[1]
        for b in BENCHMARKS
    }
    gain = any(low[b] > 0 for b in TARGETS)
    no_loss = all(low[b] > MARGINS[b] for b in BENCHMARKS)
    return "default for paged corpora" if gain and no_loss else "not adopted"


def cap_verdict(rows: Rows, types: Sequence[str]) -> str:
    """P-5 (docs/paper_cap_protocol.md): rule 1 on cross questions, rule 2 adds the guards."""

    def low(bench: str, keep: Sequence[bool] | None = None) -> float:
        w, s = (rows[bench][name]["primary"] for name in ("W18", "S30"))
        if keep is not None:
            w, s = ([v for v, k in zip(side, keep, strict=True) if k] for side in (w, s))
        return paired_bootstrap_ci(w, s, n_resamples=N_RESAMPLES)[1]

    cross = [t == "cross" for t in types]
    if low("long", cross) <= 0:
        return "not adopted"
    guards = low("long", [not c for c in cross]) > MARGINS["long"] and all(
        low(b) > MARGINS[b] for b in ("qasper", "mdaqa")
    )
    return "default for paged corpora" if guards else "recommended for comparison questions"


def save_chosen(chosen: Config, scores: Mapping[Config, float], path: Path = DEV_PATH) -> None:
    data = {"chosen": list(chosen), "scores": [[*c, round(s, 6)] for c, s in scores.items()]}
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def load_chosen(path: Path = DEV_PATH) -> Config:
    k, cap, slices = json.loads(path.read_text(encoding="utf-8"))["chosen"]
    return (k, cap, slices)


def render_dev(rows: Rows, chosen: Config, notes: Sequence[str]) -> str:
    scores = dev_scores(rows)
    lines = ["### ragkit paperpool dev: P-4 (docs/paper_pool_protocol.md)", ""]
    lines += [f"- {n}" for n in notes]
    lines += ["", "| configuration | QASPER | MDA-QA | long PDFs | dev score | |"]
    lines.append("|---|---|---|---|---|---|")
    for c in CONFIGS:
        cells = " | ".join(f"{_mean(rows[b][c]['primary']):.3f}" for b in BENCHMARKS)
        mark = " ".join(n for n, on in (("B36", c == B36), ("**C**", c == chosen)) if on)
        lines.append(f"| {label(c)} | {cells} | {scores[c]:.4f} | {mark} |")
    return "\n".join(lines)


def _fmt_mean(values: Sequence[float], metric: str) -> str:
    if metric in ("packed", "tokens"):
        return f"{_mean(values):.2f}" if metric == "packed" else f"{_mean(values):.0f}"
    m, lo, hi = bootstrap_ci(values, n_resamples=N_RESAMPLES)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_diff(a: Sequence[float], b: Sequence[float]) -> str:
    m, lo, hi = paired_bootstrap_ci(a, b, n_resamples=N_RESAMPLES)
    better = sum(x > y for x, y in zip(a, b, strict=True))
    worse = sum(x < y for x, y in zip(a, b, strict=True))
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({better} better, {worse} worse of {len(a)})"


def render_test(
    rows: Rows,
    configs: Mapping[str, Config],
    types: Sequence[str],
    notes: Sequence[str],
    *,
    title: str = "### ragkit paperpool score: P-4 (docs/paper_pool_protocol.md)",
    base: str = "B36",
    new: str = "C",
) -> str:
    lines = [title, ""]
    lines += [f"- {name} = {label(c)}" for name, c in configs.items()]
    primaries = {"qasper": "evidence recall", "mdaqa": "needed papers packed"}
    for bench in BENCHMARKS:
        by_system = rows[bench]
        n = len(by_system[base]["primary"])
        primary = primaries.get(bench, "evidence recall")
        lines += ["", f"#### {bench} test ({n} questions; primary = {primary})", ""]
        lines.append("| system | " + " | ".join(METRICS.values()) + " |")
        lines.append("|---|" + "---|" * len(METRICS))
        for name in configs:
            cells = " | ".join(_fmt_mean(by_system[name][m], m) for m in METRICS)
            lines.append(f"| {name} | {cells} |")
        lines.append("")
        for name in configs:
            if name != base:
                diff = _fmt_diff(by_system[name]["primary"], by_system[base]["primary"])
                lines.append(f"- {name} - {base}: {diff}")
    groups = {**{t: (t,) for t in pb.TYPES}, "single": ("named", "described")}
    for kind, members in groups.items():
        keep = [i for i, t in enumerate(types) if t in members]
        c = [rows["long"][new]["primary"][i] for i in keep]
        b = [rows["long"][base]["primary"][i] for i in keep]
        diff = _fmt_diff(c, b)
        lines.append(f"- long PDFs, {kind}: {base} {_mean(b):.3f}, {new} - {base} {diff}")
    outcome = verdict(rows) if new == "C" else cap_verdict(rows, types)
    lines += ["", "#### Verdict", "", f"- {new}: **{outcome}**", "", "#### Not gating", ""]
    lines += [f"- {n}" for n in notes]
    return "\n".join(lines)
