"""Post-hoc diagnosis of why the lexical variants (A2 to A4) moved Recall@5 (model-free).

``ragkit lexical --explain`` prints this. It re-scores the protocol configurations at chunk
level, follows every (question, relevant article) pair through the stages

    A0 (pinned BM25) -> A1 (full chunk list) -> S0 (structured chunks, body only)
       -> A3 (BM25F heading field) -> A4 (query expansion)

and reports where each pair crossed the top-5 boundary and which side moved. ``S0`` is a
diagnostic stage, not a protocol configuration: ``BM25FIndex(title_weight=0)`` on the structured
chunks, i.e. plain BM25 on the structured body text. It separates the chunking effect from the
heading effect, which A2 mixes (A2 prepends the heading to the chunk text). Nothing here changes a
shipped parameter; the stage indexes are built from the same constants as ``ragkit.lexical``.
"""

from __future__ import annotations

import itertools
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from ragkit.bm25 import BM25Index, tokenize
from ragkit.bm25f import BM25FIndex
from ragkit.chunking import Chunk
from ragkit.expand import DEFAULT_EXPANSION_WEIGHT, SynonymExpander
from ragkit.lexical import (
    K1,
    METRIC_LABELS,
    TITLE_WEIGHT,
    TOP_DOCS,
    B,
    Question,
    _recursive_chunks,
    _structured_chunks,
    score_question,
)

CUT = 5  # Recall@5
A0_CHUNKS = 10  # the pinned baseline keeps only the top 10 chunks
# Protocol configurations plus the diagnostic S0, in the order a pair is followed through.
STAGES = ("A0", "A1", "A2", "S0", "A3", "A4")
SEQUENCE = ("A0", "A1", "S0", "A3", "A4")
STAGE_LABELS = {
    "A0": "pinned BM25, top-10 chunks",
    "A1": "pinned BM25, full chunk list",
    "A2": "structured + prepended heading, BM25",
    "S0": "structured body only, BM25 (diagnostic)",
    "A3": "structured, BM25F heading field",
    "A4": "A3 + synonym expansion",
}
TRANSITION_NAMES = {
    ("A0", "A1"): "A0 top-10-chunk cut",
    ("A1", "S0"): "structured chunking",
    ("S0", "A3"): "heading field",
    ("A3", "A4"): "query expansion",
}
HEADING_EFFECT_MIN = 0.05  # a term counts as a heading hit when it adds at least this score
_TITLE = re.compile(r"^(?P<act>.*?), Article \S+: (?P<head>.*)$")

Terms = list[tuple[str, float]]


@dataclass(frozen=True)
class DocHit:
    """A document's best chunk under one configuration (rank 1 is the top document)."""

    doc_id: str
    rank: int
    score: float
    chunk_id: str
    chunk_chars: int


def split_title(title: str) -> tuple[str, str]:
    """``(act name, article heading)`` of a ``"<act>, Article N: <heading>"`` title."""
    m = _TITLE.match(title)
    return (m.group("act"), m.group("head")) if m else ("", title)


class _Stage:
    """One configuration at chunk level: search, per-term score tables, document ranking."""

    def __init__(
        self,
        stage_id: str,
        chunks: Sequence[Chunk],
        index: BM25Index | BM25FIndex,
        *,
        expand: bool = False,
        chunk_cap: int | None = None,
    ) -> None:
        self.id = stage_id
        self.chunks = list(chunks)
        self.index = index
        self.expander = SynonymExpander(weight=DEFAULT_EXPANSION_WEIGHT) if expand else None
        self.chunk_cap = chunk_cap
        self._tables: dict[str, dict[str, float]] = {}

    def terms(self, query: str) -> Terms:
        if self.expander is not None:
            return self.expander.expand(query)
        return [(t, 1.0) for t in tokenize(query)]

    def _search(self, terms: Terms, depth: int) -> list[tuple[Chunk, float]]:
        if isinstance(self.index, BM25FIndex):
            return self.index.search(terms, k=depth)
        return self.index.search(" ".join(t for t, _ in terms), k=depth)

    def chunk_ranking(self, terms: Terms) -> list[tuple[Chunk, float]]:
        """Positive-score chunks, best first (the first ``chunk_cap`` only, if capped)."""
        return self._search(terms, self.chunk_cap or len(self.chunks))

    def doc_hits(self, terms: Terms) -> list[DocHit]:
        """Documents by their best chunk, ranked from the chunk list (first occurrence)."""
        hits: list[DocHit] = []
        seen: set[str] = set()
        for chunk, score in self.chunk_ranking(terms):
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            hits.append(DocHit(chunk.doc_id, len(hits) + 1, score, chunk.id, len(chunk.text)))
        return hits

    def term_table(self, token: str) -> dict[str, float]:
        """Chunk id -> score of the single unit-weight term ``token`` (scores add over terms)."""
        if token not in self._tables:
            ranked = self._search([(token, 1.0)], len(self.chunks))
            self._tables[token] = {c.id: s for c, s in ranked}
        return self._tables[token]

    def contributions(self, terms: Terms, chunk_id: str) -> dict[str, float]:
        """Per-token share of the chunk's score for a weighted query."""
        weights: dict[str, float] = defaultdict(float)
        for token, w in terms:
            weights[token] += w
        out = {t: w * self.term_table(t).get(chunk_id, 0.0) for t, w in weights.items()}
        return {t: s for t, s in out.items() if s > 0.0}


@dataclass
class QuestionRun:
    """All stages' document rankings for one question."""

    question: Question
    terms: dict[str, Terms]
    hits: dict[str, list[DocHit]]

    def hit(self, stage: str, doc_id: str) -> DocHit | None:
        return next((h for h in self.hits[stage] if h.doc_id == doc_id), None)

    def rank(self, stage: str, doc_id: str) -> int | None:
        h = self.hit(stage, doc_id)
        return h.rank if h else None

    def ratio(self, stage: str, doc_id: str) -> float:
        """Score of ``doc_id`` relative to the top document of the stage (0 if unranked)."""
        h = self.hit(stage, doc_id)
        return h.score / self.hits[stage][0].score if h else 0.0


def _in_top(rank: int | None) -> bool:
    return rank is not None and rank <= CUT


def _inf(rank: int | None) -> float:
    return float("inf") if rank is None else float(rank)


@dataclass(frozen=True)
class Crossing:
    """A relevant article entering or leaving the top 5 between two consecutive stages."""

    question: int
    doc_id: str
    transition: tuple[str, str]
    gained: bool
    side: str  # "relevant" or "competitors": whose relative score moved more; "" for the cut
    movers: tuple[str, ...]  # documents that passed the relevant one (lost) or were passed (gained)

    @property
    def cause(self) -> str:
        name = TRANSITION_NAMES[self.transition]
        if not self.side:
            return name
        if self.side == "relevant":
            verb = "relevant doc scored higher" if self.gained else "relevant doc scored lower"
        else:
            verb = "competitors scored lower" if self.gained else "competitors scored higher"
        return f"{name}: {verb}"


@dataclass(frozen=True)
class PairTrace:
    """One (question, relevant article) pair followed through the stages."""

    question: int
    doc_id: str
    ranks: dict[str, int | None]
    crossings: tuple[Crossing, ...]
    sum_rank_a0: int | None  # counterfactual: article score = sum of its chunk scores
    sum_rank_a4: int | None

    @property
    def net(self) -> str:
        """``lost`` / ``gained`` if top-5 membership differs between A0 and A4, else ``""``."""
        a0, a4 = _in_top(self.ranks["A0"]), _in_top(self.ranks["A4"])
        return "" if a0 == a4 else ("gained" if a4 else "lost")


def classify_crossing(
    transition: tuple[str, str],
    gained: bool,
    rel_before: float,
    rel_after: float,
    movers_before: Sequence[float],
    movers_after: Sequence[float],
) -> str:
    """Which side moved more, in the direction of the crossing: ``relevant`` or ``competitors``.

    Scores are relative to each stage's top document, so stages with different idf scales are
    comparable. For a loss the relevant article's relative score falls, the movers' rises; the
    larger of the two shifts is the side. The A0 -> A1 step has no side: it only restores the
    chunks A0 discards.
    """
    if transition == ("A0", "A1"):
        return ""
    rel_shift = rel_after - rel_before
    mover_shift = (
        statistics.fmean(a - b for a, b in zip(movers_after, movers_before, strict=True))
        if movers_before
        else 0.0
    )
    if gained:
        return "relevant" if rel_shift > -mover_shift else "competitors"
    return "relevant" if -rel_shift > mover_shift else "competitors"


@dataclass
class Explanation:
    """Everything ``render_explanation`` prints; built by :func:`explain`."""

    stages: dict[str, _Stage]
    runs: list[QuestionRun]
    pairs: list[PairTrace]
    titles: dict[str, str] = field(default_factory=dict)

    @property
    def flips(self) -> list[PairTrace]:
        return [p for p in self.pairs if p.net]


def _build_stages(documents: Sequence[tuple[str, str]]) -> dict[str, _Stage]:
    rec = _recursive_chunks(documents)
    bm25 = BM25Index()
    bm25.add(rec)
    head = _structured_chunks(documents, prepend_heading=True)
    plain = BM25Index(k1=K1, b=B)
    plain.add(head)
    body = _structured_chunks(documents, prepend_heading=False)
    s0 = BM25FIndex(k1=K1, b=B, title_weight=0.0)
    s0.add(body)
    f = BM25FIndex(k1=K1, b=B, title_weight=TITLE_WEIGHT)
    f.add(body)
    return {
        "A0": _Stage("A0", rec, bm25, chunk_cap=A0_CHUNKS),
        "A1": _Stage("A1", rec, bm25),
        "A2": _Stage("A2", head, plain),
        "S0": _Stage("S0", body, s0),
        "A3": _Stage("A3", body, f),
        "A4": _Stage("A4", body, f, expand=True),
    }


def _sum_rank(stage: _Stage, terms: Terms, doc_id: str) -> int | None:
    totals: dict[str, float] = defaultdict(float)
    for chunk, score in stage.chunk_ranking(terms):
        totals[chunk.doc_id] += score
    order = sorted(totals, key=lambda d: -totals[d])
    return order.index(doc_id) + 1 if doc_id in totals else None


def _trace(stages: dict[str, _Stage], run: QuestionRun, index: int, doc_id: str) -> PairTrace:
    ranks = {s: run.rank(s, doc_id) for s in STAGES}
    crossings: list[Crossing] = []
    for a, b in itertools.pairwise(SEQUENCE):
        in_a, in_b = _in_top(ranks[a]), _in_top(ranks[b])
        if in_a == in_b:
            continue
        gained = in_b
        r_a, r_b = _inf(ranks[a]), _inf(ranks[b])
        if gained:  # documents the relevant article moved past
            movers = [
                h.doc_id
                for h in run.hits[b]
                if _inf(run.rank(a, h.doc_id)) < r_a and h.rank > r_b and h.doc_id != doc_id
            ]
        else:  # documents that moved past the relevant article
            movers = [
                h.doc_id
                for h in run.hits[b]
                if _inf(run.rank(a, h.doc_id)) > r_a and h.rank <= r_b and h.doc_id != doc_id
            ]
        side = classify_crossing(
            (a, b),
            gained,
            run.ratio(a, doc_id),
            run.ratio(b, doc_id),
            [run.ratio(a, d) for d in movers],
            [run.ratio(b, d) for d in movers],
        )
        crossings.append(Crossing(index, doc_id, (a, b), gained, side, tuple(movers)))
    return PairTrace(
        index,
        doc_id,
        ranks,
        tuple(crossings),
        _sum_rank(stages["A1"], run.terms["A1"], doc_id),
        _sum_rank(stages["A4"], run.terms["A4"], doc_id),
    )


def explain(documents: Sequence[tuple[str, str]], questions: Sequence[Question]) -> Explanation:
    """Follow every (question, relevant article) pair through the stages."""
    stages = _build_stages(documents)
    runs: list[QuestionRun] = []
    pairs: list[PairTrace] = []
    for i, question in enumerate(questions):
        terms = {s: stages[s].terms(question.text) for s in STAGES}
        hits = {s: stages[s].doc_hits(terms[s]) for s in STAGES}
        run = QuestionRun(question, terms, hits)
        runs.append(run)
        pairs.extend(_trace(stages, run, i, d) for d in question.relevant)
    titles = {c.doc_id: c.title for c in stages["S0"].chunks}
    return Explanation(stages, runs, pairs, titles)


# ---------------------------------------------------------------------------------------------
# Rendering


def _fmt_rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def heading_effect(expl: Explanation, run: QuestionRun, doc_id: str) -> list[tuple[str, float]]:
    """Per-term score change of the document's best A3 chunk caused by the heading field.

    A3 minus S0 on the same chunk, largest absolute change first. Positive: the term is in the
    heading. Negative: the heading field raised the term's document frequency (idf falls for
    every chunk) or the length statistics, and this chunk's body match is worth less.
    """
    hit = run.hit("A3", doc_id)
    if hit is None:
        return []
    with_head = expl.stages["A3"].contributions(run.terms["A3"], hit.chunk_id)
    without = expl.stages["S0"].contributions(run.terms["A3"], hit.chunk_id)
    effect = {t: with_head.get(t, 0.0) - without.get(t, 0.0) for t in with_head | without}
    return sorted(
        ((t, e) for t, e in effect.items() if abs(e) >= HEADING_EFFECT_MIN),
        key=lambda x: -abs(x[1]),
    )


def _terms_text(pairs: Sequence[tuple[str, float]], limit: int = 4) -> str:
    return ", ".join(f"{t} {v:+.2f}" for t, v in pairs[:limit]) or "none"


def _top_contributions(expl: Explanation, run: QuestionRun, stage: str, doc_id: str) -> str:
    hit = run.hit(stage, doc_id)
    if hit is None:
        return "not ranked"
    contrib = expl.stages[stage].contributions(run.terms[stage], hit.chunk_id)
    top = sorted(contrib.items(), key=lambda x: -x[1])
    return ", ".join(f"{t} {v:.2f}" for t, v in top[:4])


def _expansion_effect(expl: Explanation, run: QuestionRun, doc_id: str) -> list[tuple[str, float]]:
    hit = run.hit("A4", doc_id)
    if hit is None:
        return []
    extra: Terms = [(t, w) for t, w in run.terms["A4"] if w < 1.0]
    contrib = expl.stages["A4"].contributions(extra, hit.chunk_id)
    return sorted(contrib.items(), key=lambda x: -x[1])


def _doc_line(expl: Explanation, run: QuestionRun, stage: str, doc_id: str) -> str:
    hit = run.hit(stage, doc_id)
    assert hit is not None
    heading = split_title(expl.titles.get(doc_id, ""))[1]
    return (
        f"{hit.rank}. {doc_id} ({heading!r}) {hit.score:.2f}, best chunk {hit.chunk_id} "
        f"({hit.chunk_chars} chars); "
        f"heading field [{_terms_text(heading_effect(expl, run, doc_id))}]; "
        f"expansion [{_terms_text(_expansion_effect(expl, run, doc_id))}]"
    )


def _render_flip(expl: Explanation, pair: PairTrace) -> list[str]:
    run = expl.runs[pair.question]
    q = run.question
    doc_id = pair.doc_id
    heading = split_title(expl.titles.get(doc_id, ""))[1]
    lines = [
        f"#### Q{pair.question} ({q.qtype}): {q.text}",
        f"- relevant: `{doc_id}` ({heading!r}); top-5 membership {pair.net} (A0 -> A4)",
        "- rank: " + " | ".join(f"{s} {_fmt_rank(pair.ranks[s])}" for s in STAGES),
    ]
    for c in pair.crossings:
        lines.append(
            f"- crossed the top-5 line at {c.transition[0]} -> {c.transition[1]}: {c.cause}"
        )
    for stage in ("A0", "A4"):
        hit = run.hit(stage, doc_id)
        if hit is None:
            lines.append(f"- {stage}: relevant article not ranked")
            continue
        lines.append(
            f"- {stage}: relevant best chunk {hit.chunk_id} ({hit.chunk_chars} chars, "
            f"score {hit.score:.2f}, {hit.score / run.hits[stage][0].score:.0%} of the top "
            f"document); top terms [{_top_contributions(expl, run, stage, doc_id)}]"
        )
    lines.append(
        "- relevant heading field effect on A3 (A3 minus S0, same chunk): "
        f"[{_terms_text(heading_effect(expl, run, doc_id))}]; "
        f"expansion adds to A4: [{_terms_text(_expansion_effect(expl, run, doc_id))}]"
    )
    for c in pair.crossings:
        if c.movers:
            lines.append(
                f"- documents that {'were passed by' if c.gained else 'passed'} it at "
                f"{c.transition[0]} -> {c.transition[1]}:"
            )
            for d in c.movers[:3]:
                lines.append("    - " + _doc_line(expl, run, c.transition[1], d))
    lines.append(
        f"- sum-of-chunks article score (counterfactual, not shipped): rank A0 "
        f"{_fmt_rank(pair.sum_rank_a0)} -> A4 {_fmt_rank(pair.sum_rank_a4)}"
    )
    relevant = set(q.relevant)
    rivals = [h.doc_id for h in run.hits["A4"] if h.doc_id not in relevant][:3]
    lines.append("- top-3 competing articles under A4:")
    lines.extend("    - " + _doc_line(expl, run, "A4", d) for d in rivals)
    return lines


def _cause_table(expl: Explanation) -> list[str]:
    net: Counter[tuple[str, bool]] = Counter()
    allx: Counter[tuple[str, bool]] = Counter()
    for pair in expl.pairs:
        for c in pair.crossings:
            allx[(c.cause, c.gained)] += 1
            # A crossing is part of the net flip when it ends in the A4 membership and the pair
            # flipped; the last crossing of a flipped pair is the one that decides it.
        if pair.net and pair.crossings:
            last = pair.crossings[-1]
            net[(last.cause, last.gained)] += 1
    causes = sorted({c for c, _ in allx} | {c for c, _ in net})
    lines = [
        "| cause (stage: side that moved) | net lost A0->A4 | net gained A0->A4 "
        "| all crossings out | all crossings in |",
        "|---|---|---|---|---|",
    ]
    for cause in causes:
        lines.append(
            f"| {cause} | {net[(cause, False)]} | {net[(cause, True)]} "
            f"| {allx[(cause, False)]} | {allx[(cause, True)]} |"
        )
    n_lost = sum(1 for p in expl.pairs if p.net == "lost")
    n_gained = sum(1 for p in expl.pairs if p.net == "gained")
    lines.append(
        f"| **total** | {n_lost} | {n_gained} | {sum(v for (_, g), v in allx.items() if not g)} "
        f"| {sum(v for (_, g), v in allx.items() if g)} |"
    )
    return lines


def _counts_text(counts: Counter[str]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "-"


def _top_set(run: QuestionRun, stage: str) -> set[str]:
    return {h.doc_id for h in run.hits[stage][:CUT]}


def _stage_table(expl: Explanation) -> list[str]:
    lines = [
        "| transition | relevant out (by type) | relevant in (by type) "
        "| all articles out of / into a top 5 | of those, relevant out / in |",
        "|---|---|---|---|---|",
    ]
    for (a, b), name in TRANSITION_NAMES.items():
        out_t: Counter[str] = Counter()
        in_t: Counter[str] = Counter()
        for pair in expl.pairs:
            for c in pair.crossings:
                if c.transition == (a, b):
                    qtype = expl.runs[pair.question].question.qtype or "-"
                    (in_t if c.gained else out_t)[qtype] += 1
        n_out = n_in = 0
        for run in expl.runs:
            n_out += len(_top_set(run, a) - _top_set(run, b))
            n_in += len(_top_set(run, b) - _top_set(run, a))
        lines.append(
            f"| {a} -> {b} ({name}) | {sum(out_t.values())} ({_counts_text(out_t)}) "
            f"| {sum(in_t.values())} ({_counts_text(in_t)}) | {n_out} / {n_in} "
            f"| {sum(out_t.values())} / {sum(in_t.values())} |"
        )
    return lines


def _is_definition(title: str) -> bool:
    return split_title(title)[1].strip().lower() == "definitions"


def _spread_lines(expl: Explanation) -> list[str]:
    """How flat the A1 document scores are near the top-5 line, per question type."""
    by_type: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for run in expl.runs:
        hits = run.hits["A1"]
        if len(hits) < 10:
            continue
        by_type[run.question.qtype or "-"].append(
            (hits[9].score / hits[0].score, 1.0 - hits[5].score / hits[4].score)
        )
    lines = [
        "| type | questions | median score of rank 10 / rank 1 | median gap rank 5 -> 6 | "
        "questions with that gap under 5% |",
        "|---|---|---|---|---|",
    ]
    for qtype, vals in sorted(by_type.items()):
        gaps = [g for _, g in vals]
        lines.append(
            f"| {qtype} | {len(vals)} | {statistics.median(v for v, _ in vals):.2f} "
            f"| {statistics.median(gaps):.1%} | {sum(g < 0.05 for g in gaps)} |"
        )
    return lines


def _boundary_lines(expl: Explanation) -> list[str]:
    """Where relevant articles sat under A0, and how many are in the top 5 under A4."""
    buckets = (
        ("1-3", range(1, 4)),
        ("4", range(4, 5)),
        ("5", range(5, 6)),
        ("6-7", range(6, 8)),
        ("8-10", range(8, 11)),
    )
    lines = ["| A0 rank of the relevant article | pairs | in the top 5 under A4 |", "|---|---|---|"]
    seen = 0
    for label, ranks in buckets:
        members = [p for p in expl.pairs if p.ranks["A0"] in ranks]
        seen += len(members)
        lines.append(
            f"| {label} | {len(members)} | {sum(_in_top(p.ranks['A4']) for p in members)} |"
        )
    rest = [p for p in expl.pairs if p.ranks["A0"] is None]
    lines.append(
        f"| not in A0's top-10 chunks | {len(rest)} | {sum(_in_top(p.ranks['A4']) for p in rest)} |"
    )
    return lines


def _accounting_lines(expl: Explanation) -> list[str]:
    """Split the A4 minus A0 Recall@5 and MRR change into flipped-pair questions and the rest."""
    flipped = {p.question for p in expl.flips}
    parts = {m: [0.0, 0.0] for m in ("r5", "mrr")}
    for i, run in enumerate(expl.runs):
        rel = run.question.relevant
        before = score_question([h.doc_id for h in run.hits["A0"][:TOP_DOCS]], rel)
        after = score_question([h.doc_id for h in run.hits["A4"][:TOP_DOCS]], rel)
        for m, acc in parts.items():
            acc[0 if i in flipped else 1] += after[m] - before[m]
    n = len(expl.runs)
    return [
        f"- {METRIC_LABELS[m]} A4 minus A0 = {sum(acc) / n:+.3f}: {acc[0] / n:+.3f} from the "
        f"{len(flipped)} questions with a flipped pair, {acc[1] / n:+.3f} from the other "
        f"{n - len(flipped)}"
        for m, acc in parts.items()
    ]


def _entry_lines(expl: Explanation) -> list[str]:
    """Which articles enter a top 5, on any question, at the heading and expansion steps."""
    titles = {c.doc_id: c.title for c in expl.stages["S0"].chunks}
    n_defs = sum(_is_definition(t) for t in titles.values())
    heading_entries: Counter[str] = Counter()
    fired = changed = entries = definition_entries = 0
    for run in expl.runs:
        for doc in _top_set(run, "A3") - _top_set(run, "S0"):
            gains = [(t, e) for t, e in heading_effect(expl, run, doc) if e > 0.0]
            act, head = split_title(titles.get(doc, ""))
            best = max(gains, key=lambda x: x[1])[0] if gains else ""
            if best in set(tokenize(head)):
                heading_entries["article-heading term"] += 1
            elif best in set(tokenize(act)):
                heading_entries["act-name term only"] += 1
            else:
                heading_entries["no heading term gains (idf / length shift only)"] += 1
        fired += any(w < 1.0 for _, w in run.terms["A4"])
        changed += [h.doc_id for h in run.hits["A3"][:TOP_DOCS]] != [
            h.doc_id for h in run.hits["A4"][:TOP_DOCS]
        ]
        new = _top_set(run, "A4") - _top_set(run, "A3")
        entries += len(new)
        definition_entries += sum(_is_definition(titles.get(d, "")) for d in new)
    slots = {
        s: sum(_is_definition(titles.get(d, "")) for run in expl.runs for d in _top_set(run, s))
        for s in ("A0", "S0", "A3", "A4")
    }
    n_slots = CUT * len(expl.runs)
    return [
        f"- Heading step (S0 -> A3): {sum(heading_entries.values())} article entries into a top 5 "
        f"over {len(expl.runs)} questions; the term gaining most from the heading is: "
        + ", ".join(f"{k} {v}" for k, v in sorted(heading_entries.items(), key=lambda x: -x[1])),
        f"- Expansion fires on {fired} of {len(expl.runs)} questions and changes the top-10 "
        f"article order on {changed}; {entries} articles enter a top 5 because of it, "
        f"{definition_entries} of them definition articles.",
        f"- Definition articles ('Definitions' heading; {n_defs} of {len(titles)} articles) in "
        f"the top-5 slots of the {len(expl.runs)} questions ({n_slots} slots): "
        + ", ".join(f"{s} {slots[s]}" for s in ("A0", "S0", "A3", "A4")),
    ]


def _heading_match_lines(expl: Explanation) -> list[str]:
    """Share of relevant articles and of non-relevant top-5 articles whose heading adds score."""
    rel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    other: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for run in expl.runs:
        qtype = run.question.qtype or "-"
        relevant = set(run.question.relevant)
        for doc_id in [d for d in relevant if run.hit("A3", d)]:
            rel[qtype][1] += 1
            rel[qtype][0] += any(
                e >= HEADING_EFFECT_MIN for _, e in heading_effect(expl, run, doc_id)
            )
        for doc_id in _top_set(run, "A3") - relevant:
            other[qtype][1] += 1
            other[qtype][0] += any(
                e >= HEADING_EFFECT_MIN for _, e in heading_effect(expl, run, doc_id)
            )
    lines = [
        "| type | relevant articles with a heading match | non-relevant top-5 articles "
        "with a heading match |",
        "|---|---|---|",
    ]
    for qtype in sorted(rel):
        lines.append(
            f"| {qtype} | {rel[qtype][0]} of {rel[qtype][1]} "
            f"| {other[qtype][0]} of {other[qtype][1]} |"
        )
    return lines


def render_explanation(expl: Explanation) -> str:
    """Markdown report: flipped pairs with worked detail, cause tables, global statistics."""
    flips = expl.flips
    lost = [p for p in flips if p.net == "lost"]
    gained = [p for p in flips if p.net == "gained"]
    lines = [
        f"{len(expl.pairs)} (question, relevant article) pairs on {len(expl.runs)} questions; "
        f"top-5 membership differs between A0 and A4 for {len(flips)}: {len(lost)} lost, "
        f"{len(gained)} gained. Stages: " + "; ".join(f"{s} = {STAGE_LABELS[s]}" for s in STAGES),
        "",
        *_accounting_lines(expl),
        "",
        "### Flipped pairs (A0 vs A4)",
        "",
    ]
    for pair in flips:
        lines.extend(_render_flip(expl, pair))
        lines.append("")
    lines += [
        "### Cause table (a crossing is a relevant article entering or leaving the top 5 "
        "between two consecutive stages)",
        "",
        *_cause_table(expl),
        "",
        "### Crossings by stage",
        "",
        *_stage_table(expl),
        "",
        "### Where the relevant articles sat under A0",
        "",
        *_boundary_lines(expl),
        "",
        "### How flat the scores are near the top-5 line (A1 document scores)",
        "",
        *_spread_lines(expl),
        "",
        "### Who enters the top 5 at the heading and expansion steps",
        "",
        *_entry_lines(expl),
        "",
        "### Does the heading field match the question? (A3 top 5)",
        "",
        *_heading_match_lines(expl),
    ]
    return "\n".join(lines)
