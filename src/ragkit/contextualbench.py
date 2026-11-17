"""Larger chunks and Anthropic's contextual retrieval, the published industry recipe (K-R7).

Protocol, systems and decision rule: ``docs/contextual_protocol.md`` (committed before this code).
Every system is ``vocabbench``'s ``hybrid+ce@100`` engine; only the chunks change. Contexts are
read from a call cache.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ragkit.chunking import Chunk, recursive
from ragkit.eval import paired_bootstrap_ci
from ragkit.legalbench import MiniTest
from ragkit.lexical import Question
from ragkit.llm import CachedMessages
from ragkit.vocabbench import (
    BASELINE,
    METRIC_LABELS,
    TIE,
    Engine,
    Result,
    _ci,
    _delta,
    pool,
    score_set,
)

CHUNK_SIZE = 1600
OVERLAP = 400
CONTEXT_MODEL = "qwen3.6:27b-ctx4k"
CONTEXT_NUM_CTX = 24576
SMALL_NUM_CTX = 8192  # K-R8: EU articles only, the longest is 13 253 characters
CONTEXT_MAX_TOKENS = 200
CONTEXT_PROMPT = (
    "<document>\n{document}\n</document>\n"
    "Here is the chunk we want to situate within the whole document\n"
    "<chunk>\n{chunk}\n</chunk>\n"
    "Please give a short succinct context to situate this chunk within the overall document for "
    "the purposes of improving search retrieval of the chunk. Answer only with the succinct "
    "context and nothing else."
)
LEGAL_SOURCES = ("privacy_qa", "contractnli")
BUDGET = 8000
BUDGETS = (4000, BUDGET, 16000)
SYSTEMS = ("B0", "S", "SC")
CANDIDATES = ("S", "SC")
EU_METRICS = ("mrr", "r5", "g5_prod", "pack")

Span = tuple[str, int, int]


def large_chunks(documents: Sequence[tuple[str, str]]) -> list[Chunk]:
    """The recursive 1600 / 400 chunks of every document (S and SC)."""
    return [
        chunk
        for doc_id, text in documents
        for chunk in recursive(text, doc_id, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    ]


def contexts(
    client: CachedMessages, documents: Mapping[str, str], chunks: Sequence[Chunk]
) -> dict[str, str]:
    """One context per chunk id (``""`` for an empty reply), in document order.

    A failed live call raises; the stage is simply run again and replays what is cached.
    """
    return {
        c.id: client.complete(
            "context",
            [],
            CONTEXT_PROMPT.format(document=documents[c.doc_id], chunk=c.text),
            CONTEXT_MAX_TOKENS,
        ).text.strip()
        for c in chunks
    }


def with_contexts(chunks: Sequence[Chunk], by_id: Mapping[str, str]) -> list[Chunk]:
    """SC chunks: context, a blank line, then the chunk; spans still locate the chunk text."""
    return [
        dataclasses.replace(c, text=f"{by_id[c.id]}\n\n{c.text}") if by_id.get(c.id) else c
        for c in chunks
    ]


def budget_recall(spans: Sequence[Span], truth: Sequence[Span], budget: int) -> float:
    """Gold characters covered by the first ``budget`` distinct retrieved characters, over all gold.

    Spans are taken in rank order and the last one is cut at the budget; context text has no span,
    so it never counts.
    """
    covered: dict[str, set[int]] = {}
    used = 0
    for path, start, end in spans:
        seen = covered.setdefault(path, set())
        for pos in range(start, end):
            if used == budget:
                break
            if pos not in seen:
                seen.add(pos)
                used += 1
    gold = {(path, pos) for path, start, end in truth for pos in range(start, end)}
    if not gold:
        return 0.0
    return sum(1 for path, pos in gold if pos in covered.get(path, ())) / len(gold)


def score_legal(engine: Engine, tests: Sequence[MiniTest]) -> dict[int, list[float]]:
    """Budget recall of every test at every budget (shipped ``hybrid+ce@100``)."""
    engine.prepare([BASELINE], [t.query for t in tests], {})
    out: dict[int, list[float]] = {b: [] for b in BUDGETS}
    for test in tests:
        hits = engine.rank(BASELINE, test.query, "")
        found = [engine.chunks[i] for i, _ in hits]
        spans = [(c.doc_id, c.start, c.end) for c in found]
        for budget in BUDGETS:
            out[budget].append(budget_recall(spans, test.truth, budget))
    return out


@dataclass
class Row:
    """One system on every EU set and every LegalBench source."""

    eu: dict[str, Result]
    legal: dict[str, dict[int, list[float]]]

    def eu_pooled(self) -> Result:
        return pool(list(self.eu.values()))

    def legal_pooled(self, budget: int = BUDGET) -> list[float]:
        return [v for source in LEGAL_SOURCES for v in self.legal[source][budget]]


def _mean_diff(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x - y for x, y in zip(a, b, strict=True)) / len(a)


@dataclass(frozen=True)
class Verdict:
    """Outcome of the decision rule for one candidate."""

    status: str
    d_mrr: tuple[float, float, float]
    wrong: int
    base_wrong: int
    d_pack: float
    d_legal: float
    failed: list[str]

    @property
    def adopted(self) -> bool:
        return self.status == "adopted"


def decide(base: Row, cand: Row) -> Verdict:
    """The protocol's conditions: EU gain, EU serving context, LegalBench not hurt (K-R7 only).

    K-R8 rows have no LegalBench results, so the third condition does not apply to them.
    """
    b, c = base.eu_pooled(), cand.eu_pooled()
    d_mrr = paired_bootstrap_ci(c.scores["mrr"], b.scores["mrr"])
    wrong, base_wrong = sum(c.wrong_act), sum(b.wrong_act)
    d_pack = _mean_diff(c.scores["pack"], b.scores["pack"])
    d_legal = _mean_diff(cand.legal_pooled(), base.legal_pooled()) if base.legal else 0.0
    failed = []
    if d_mrr[1] <= 0:
        failed.append(f"1: EU MRR CI lower bound {d_mrr[1]:+.3f} <= 0")
    if wrong > base_wrong:
        failed.append(f"2: EU wrong-act {wrong} > {base_wrong}")
    if d_pack < -TIE:
        failed.append(f"2: EU pack recall difference {d_pack:+.3f} < 0")
    if d_legal < -TIE:
        failed.append(f"3: LegalBench budget recall difference {d_legal:+.4f} < 0")
    if not failed:
        status = "adopted"
    elif d_mrr[2] < 0:
        status = "rejected"
    else:
        status = "not adopted (no evidence)"
    return Verdict(status, d_mrr, wrong, base_wrong, d_pack, d_legal, failed)


def choose(rows: Mapping[str, Row], verdicts: Mapping[str, Verdict]) -> str | None:
    """What ships: SC only if both pass and SC beats S on EU MRR with a CI above 0."""
    adopted = [k for k in CANDIDATES if verdicts[k].adopted]
    if len(adopted) < 2:
        return adopted[0] if adopted else None
    sc, s = rows["SC"].eu_pooled().scores["mrr"], rows["S"].eu_pooled().scores["mrr"]
    return "SC" if paired_bootstrap_ci(sc, s)[1] > 0 else "S"


def run(
    engines: Mapping[str, Mapping[str, Engine]],
    eu_sets: Mapping[str, Sequence[Question]],
    legal: Mapping[str, Sequence[MiniTest]],
) -> dict[str, Row]:
    """Score every system once (``engines[system]`` has an ``eu`` and a ``legal`` engine)."""
    return {
        name: Row(
            {s: score_set(engines[name]["eu"], BASELINE, qs, {}) for s, qs in eu_sets.items()},
            {s: score_legal(engines[name]["legal"], tests) for s, tests in legal.items()},
        )
        for name in SYSTEMS
    }


def _eu_table(rows: Mapping[str, Row], pick: str | None) -> list[str]:
    header = " | ".join(METRIC_LABELS[m] for m in EU_METRICS)
    lines = [f"| system | {header} | wrong-act | MRR difference vs B0 |"]
    lines.append("|---|" + "---|" * (len(EU_METRICS) + 2))
    for name, row in rows.items():
        r = row.eu_pooled() if pick is None else row.eu[pick]
        b = rows["B0"].eu_pooled() if pick is None else rows["B0"].eu[pick]
        cells = " | ".join(_ci(r.scores[m]) for m in EU_METRICS)
        delta = "" if name == "B0" else _delta(r.scores["mrr"], b.scores["mrr"])
        lines.append(f"| {name} | {cells} | {sum(r.wrong_act)} | {delta} |")
    return lines


def _eu_section(rows: Mapping[str, Row]) -> list[str]:
    base = rows["B0"]
    pooled = len(base.eu_pooled().wrong_act)
    lines = [f"#### EU pooled ({pooled} questions, consolidated corpus)", ""]
    lines += _eu_table(rows, None)
    for name in base.eu:
        lines += ["", f"#### EU {name} ({len(base.eu[name].wrong_act)})", ""]
        lines += _eu_table(rows, name)
    return lines


def render_small(rows: Mapping[str, Row], notes: Sequence[str]) -> str:
    """K-R8: the EU tables of B0 and C8, the verdict and the non-gating notes."""
    lines = ["### ragkit contextual score-small: K-R8 (docs/contextual_small_protocol.md)", ""]
    lines += _eu_section(rows)
    lines += ["", "#### Verdict", ""]
    for name in ("C8", "P0"):
        v = decide(rows["B0"], rows[name])
        m, lo, hi = v.d_mrr
        label = "" if name == "C8" else " (control, not gating)"
        lines.append(
            f"- {name}{label}: **{v.status}**. EU MRR {m:+.3f} [{lo:+.3f}, {hi:+.3f}], wrong-act "
            f"{v.wrong} vs {v.base_wrong}, pack recall {v.d_pack:+.3f}."
        )
        lines += [f"  - failed {f}" for f in v.failed]
    lines += ["", "#### Not gating", ""]
    lines += [f"- {note}" for note in notes]
    return "\n".join(lines)


def render(rows: Mapping[str, Row], notes: Sequence[str]) -> str:
    """EU and LegalBench tables with paired differences, the verdicts and the non-gating notes."""
    base = rows["B0"]
    lines = ["### ragkit contextual score: K-R7 (docs/contextual_protocol.md)", ""]
    lines += _eu_section(rows)
    for budget in BUDGETS:
        size = len(base.legal_pooled(budget))
        lines += ["", f"#### LegalBench {' + '.join(LEGAL_SOURCES)} ({size}), budget {budget}", ""]
        lines += ["| system | budget recall | difference vs B0 (better/worse/tied) | by source |"]
        lines += ["|---|---|---|---|"]
        for name, row in rows.items():
            values = row.legal_pooled(budget)
            delta = "" if name == "B0" else _delta(values, base.legal_pooled(budget))
            by_source = ", ".join(
                f"{s} {sum(row.legal[s][budget]) / len(row.legal[s][budget]):.3f}"
                for s in LEGAL_SOURCES
            )
            lines.append(f"| {name} | {_ci(values)} | {delta} | {by_source} |")
    verdicts = {name: decide(base, rows[name]) for name in CANDIDATES}
    lines += ["", "#### Verdicts", ""]
    for name, v in verdicts.items():
        m, lo, hi = v.d_mrr
        lines.append(
            f"- {name}: **{v.status}**. EU MRR {m:+.3f} [{lo:+.3f}, {hi:+.3f}], wrong-act "
            f"{v.wrong} vs {v.base_wrong}, pack recall {v.d_pack:+.3f}; LegalBench budget recall "
            f"({BUDGET}) {v.d_legal:+.4f}."
        )
        lines += [f"  - failed {f}" for f in v.failed]
    lines.append(f"- Ships: **{choose(rows, verdicts) or 'nothing'}**.")
    lines += ["", "#### Not gating", ""]
    lines += [f"- {note}" for note in notes]
    return "\n".join(lines)
