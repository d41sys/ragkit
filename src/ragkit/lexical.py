"""Model-free lexical retrieval experiment (protocol: ``docs/lexical_protocol.md``).

Compares the pinned BM25 baseline against structure-aware chunking, BM25F with a heading field
and synonym/acronym query expansion, plus a cross-validated (k1, b, title weight) variant. Only
numpy and the standard library are used: no embedder, cross-encoder or other model is built.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ragkit.bm25 import BM25Index
from ragkit.bm25f import BM25FIndex
from ragkit.chunking import Chunk, recursive, structured
from ragkit.eval import bootstrap_ci, mrr, ndcg_at_k, paired_bootstrap_ci, recall_at_k
from ragkit.expand import DEFAULT_EXPANSION_WEIGHT, SynonymExpander

# Frozen by the protocol, committed before any scoring.
K1 = 1.5
B = 0.75
TITLE_WEIGHT = 2.0
MIN_CHARS = 250
MAX_CHARS = 1000
CV_FOLDS = 2
CV_SEEDS = (0, 1, 2, 3, 4)
GRID_K1 = (1.5, 0.9, 1.2, 2.0)  # default first: ties keep the earlier point
GRID_B = (0.75, 0.5, 0.9)
GRID_TITLE_WEIGHT = (2.0, 1.0, 4.0)

METRICS = ("r5", "mrr", "ndcg")
METRIC_LABELS = {"r5": "Recall@5", "mrr": "MRR", "ndcg": "nDCG@10"}
TOP_DOCS = 10

Retriever = Callable[[str], list[str]]
GridPoint = tuple[float, float, float]


@dataclass(frozen=True)
class Question:
    """One answerable question with its relevant document ids and optional type."""

    text: str
    relevant: tuple[str, ...]
    qtype: str


def questions_from_records(records: Sequence[Mapping[str, Any]]) -> list[Question]:
    """Build questions from ``qa*.jsonl`` records (``q``, ``relevant_doc_ids``, ``type``)."""
    return [
        Question(
            text=str(r["q"]),
            relevant=tuple(str(d) for d in r["relevant_doc_ids"]),
            qtype=str(r.get("type", "")),
        )
        for r in records
    ]


def rank_documents(
    ranked_chunks: Sequence[tuple[Chunk, float]], limit: int | None = TOP_DOCS
) -> list[str]:
    """Document ids of a ranked chunk list, first occurrence only, at most ``limit``."""
    seen: list[str] = []
    for chunk, _ in ranked_chunks:
        if chunk.doc_id not in seen:
            seen.append(chunk.doc_id)
            if limit is not None and len(seen) == limit:
                break
    return seen


def _structured_chunks(
    documents: Sequence[tuple[str, str]], *, prepend_heading: bool
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, text in documents:
        chunks.extend(
            structured(
                text,
                doc_id,
                min_chars=MIN_CHARS,
                max_chars=MAX_CHARS,
                prepend_heading=prepend_heading,
            )
        )
    return chunks


def _recursive_chunks(documents: Sequence[tuple[str, str]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, text in documents:
        chunks.extend(recursive(text, doc_id, chunk_size=500, overlap=50))
    return chunks


def pinned_baseline(documents: Sequence[tuple[str, str]]) -> Retriever:
    """A0: the existing ``BM25Index`` on 500-char recursive chunks, top 10 chunks then dedupe."""
    index = BM25Index()
    index.add(_recursive_chunks(documents))
    return lambda q: rank_documents(index.search(q, k=10))


def baseline_doc_depth(documents: Sequence[tuple[str, str]]) -> Retriever:
    """A1: A0's index, but documents are ranked from the full positive-score chunk list."""
    chunks = _recursive_chunks(documents)
    index = BM25Index()
    index.add(chunks)
    return lambda q: rank_documents(index.search(q, k=len(chunks)))


def structure_bm25(documents: Sequence[tuple[str, str]]) -> Retriever:
    """A2: structured chunks with the heading prepended, plain BM25."""
    chunks = _structured_chunks(documents, prepend_heading=True)
    index = BM25Index(k1=K1, b=B)
    index.add(chunks)
    return lambda q: rank_documents(index.search(q, k=len(chunks)))


def bm25f_retriever(
    documents: Sequence[tuple[str, str]],
    *,
    k1: float = K1,
    b: float = B,
    title_weight: float = TITLE_WEIGHT,
    expansion_weight: float | None = None,
) -> Retriever:
    """A3 (``expansion_weight=None``) and A4: BM25F over the structured chunks' heading field."""
    chunks = _structured_chunks(documents, prepend_heading=False)
    index = BM25FIndex(k1=k1, b=b, title_weight=title_weight)
    index.add(chunks)
    return _bm25f_search(index, expansion_weight, len(chunks))


def _bm25f_search(index: BM25FIndex, expansion_weight: float | None, depth: int) -> Retriever:
    if expansion_weight is None:
        return lambda q: rank_documents(index.search(q, k=depth))
    expander = SynonymExpander(weight=expansion_weight)
    return lambda q: rank_documents(index.search(expander.expand(q), k=depth))


@dataclass(frozen=True)
class Config:
    """A named retrieval configuration of the protocol."""

    id: str
    name: str
    build: Callable[[Sequence[tuple[str, str]]], Retriever]


CONFIGS: tuple[Config, ...] = (
    Config("A0", "bm25-pinned", pinned_baseline),
    Config("A1", "+doc-depth", baseline_doc_depth),
    Config("A2", "+structure", structure_bm25),
    Config("A3", "+bm25f", lambda docs: bm25f_retriever(docs)),
    Config(
        "A4",
        "+expansion",
        lambda docs: bm25f_retriever(docs, expansion_weight=DEFAULT_EXPANSION_WEIGHT),
    ),
)
BASELINE_ID = "A0"
HEADLINE_ID = "A4"
CV_ID = "A5"
CV_NAME = "+cv-params"


def score_question(retrieved: Sequence[str], relevant: Sequence[str]) -> dict[str, float]:
    """Recall@5, MRR and nDCG@10 of one ranked document list."""
    return {
        "r5": recall_at_k(retrieved, relevant, k=5),
        "mrr": mrr(retrieved, relevant),
        "ndcg": ndcg_at_k(retrieved, relevant, k=10),
    }


def score_retriever(retriever: Retriever, questions: Sequence[Question]) -> dict[str, list[float]]:
    """Per-question metric lists, in question order."""
    out: dict[str, list[float]] = {m: [] for m in METRICS}
    for question in questions:
        for metric, value in score_question(retriever(question.text), question.relevant).items():
            out[metric].append(value)
    return out


def stratified_folds(strata: Sequence[str], n_folds: int, seed: int) -> list[int]:
    """Fold index per item; each stratum is shuffled and dealt round-robin over the folds."""
    rng = np.random.default_rng(seed)
    folds = [0] * len(strata)
    for stratum in sorted(set(strata)):
        members = [i for i, s in enumerate(strata) if s == stratum]
        order = rng.permutation(len(members))
        for rank, pos in enumerate(order):
            folds[members[int(pos)]] = rank % n_folds
    return folds


def grid_points() -> list[GridPoint]:
    """The (k1, b, title weight) grid, default point first."""
    return list(itertools.product(GRID_K1, GRID_B, GRID_TITLE_WEIGHT))


def score_grid(
    documents: Sequence[tuple[str, str]], questions: Sequence[Question]
) -> dict[GridPoint, dict[str, list[float]]]:
    """Per-question metrics of the A4 pipeline at every grid point."""
    chunks = _structured_chunks(documents, prepend_heading=False)
    results: dict[GridPoint, dict[str, list[float]]] = {}
    for title_weight in GRID_TITLE_WEIGHT:
        index = BM25FIndex(title_weight=title_weight)
        index.add(chunks)
        for k1, b in itertools.product(GRID_K1, GRID_B):
            index.k1, index.b = k1, b
            retriever = _bm25f_search(index, DEFAULT_EXPANSION_WEIGHT, len(chunks))
            results[(k1, b, title_weight)] = score_retriever(retriever, questions)
    return results


def cross_validate(
    grid: Mapping[GridPoint, Mapping[str, Sequence[float]]],
    strata: Sequence[str],
    *,
    n_folds: int = CV_FOLDS,
    seeds: Sequence[int] = CV_SEEDS,
) -> tuple[dict[str, list[float]], Counter[GridPoint]]:
    """Held-out per-question metrics when the grid point is picked on the training folds.

    For each seed and fold the point with the highest training-fold MRR is chosen (the earliest
    point wins ties) and applied to the held-out fold. A question's score is the mean of its
    held-out scores over the seeds. Returns those scores and how often each point was chosen.
    """
    points = list(grid)
    n = len(strata)
    held_out = {m: np.zeros(n) for m in METRICS}
    chosen: Counter[GridPoint] = Counter()
    for seed in seeds:
        folds = stratified_folds(strata, n_folds, seed)
        for fold in range(n_folds):
            train = [i for i in range(n) if folds[i] != fold]
            test = [i for i in range(n) if folds[i] == fold]
            best = points[0]
            best_mrr = float(np.mean([grid[best]["mrr"][i] for i in train]))
            for point in points[1:]:
                value = float(np.mean([grid[point]["mrr"][i] for i in train]))
                if value > best_mrr:
                    best, best_mrr = point, value
            chosen[best] += 1
            for metric in METRICS:
                for i in test:
                    held_out[metric][i] += grid[best][metric][i] / len(seeds)
    return {m: [float(v) for v in held_out[m]] for m in METRICS}, chosen


@dataclass
class LexicalReport:
    """Per-question scores of every configuration on one dataset."""

    questions: list[Question]
    names: dict[str, str]
    scores: dict[str, dict[str, list[float]]]
    cv_choices: Counter[GridPoint] = field(default_factory=Counter)

    @property
    def types(self) -> list[str]:
        return sorted({q.qtype for q in self.questions if q.qtype})


def evaluate(
    documents: Sequence[tuple[str, str]],
    questions: Sequence[Question],
    *,
    cv_seeds: Sequence[int] = CV_SEEDS,
) -> LexicalReport:
    """Score A0 to A5 on the questions."""
    names = {c.id: c.name for c in CONFIGS} | {CV_ID: CV_NAME}
    scores = {c.id: score_retriever(c.build(documents), questions) for c in CONFIGS}
    grid = score_grid(documents, questions)
    scores[CV_ID], choices = cross_validate(grid, [q.qtype for q in questions], seeds=cv_seeds)
    return LexicalReport(list(questions), names, scores, choices)


def _cell(values: Sequence[float]) -> str:
    mean, lo, hi = bootstrap_ci(values)
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def _delta(a: Sequence[float], b: Sequence[float]) -> str:
    mean, lo, hi = paired_bootstrap_ci(a, b)
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _subset(values: Sequence[float], idx: Sequence[int]) -> list[float]:
    return [values[i] for i in idx]


def _win_loss(a: Sequence[float], b: Sequence[float]) -> str:
    better = sum(1 for x, y in zip(a, b, strict=True) if x > y + 1e-9)
    worse = sum(1 for x, y in zip(a, b, strict=True) if x < y - 1e-9)
    return f"{better} / {worse} / {len(a) - better - worse}"


def render_report(report: LexicalReport) -> str:
    """Markdown tables: overall, paired vs A0, by type, and the cross-validated choices."""
    ids = list(report.scores)
    base = report.scores[BASELINE_ID]
    n = len(report.questions)
    lines = [f"{n} questions; mean [bootstrap 95% CI, 2000 resamples, seed 0]", ""]

    lines += ["| Config | Recall@5 | MRR | nDCG@10 |", "|---|---|---|---|"]
    for cid in ids:
        cells = " | ".join(_cell(report.scores[cid][m]) for m in METRICS)
        lines.append(f"| {cid} {report.names[cid]} | {cells} |")

    lines += ["", f"Paired difference vs {BASELINE_ID} {report.names[BASELINE_ID]}:", ""]
    lines += ["| Config | dRecall@5 | dMRR | dnDCG@10 |", "|---|---|---|---|"]
    for cid in ids:
        if cid == BASELINE_ID:
            continue
        cells = " | ".join(_delta(report.scores[cid][m], base[m]) for m in METRICS)
        lines.append(f"| {cid} {report.names[cid]} | {cells} |")

    lines += ["", f"Questions better / worse / tied than {BASELINE_ID} (per question):", ""]
    lines += ["| Config | Recall@5 | MRR | nDCG@10 |", "|---|---|---|---|"]
    for cid in ids:
        if cid == BASELINE_ID:
            continue
        cells = " | ".join(_win_loss(report.scores[cid][m], base[m]) for m in METRICS)
        lines.append(f"| {cid} {report.names[cid]} | {cells} |")

    types = report.types
    if types:
        by_type = {t: [i for i, q in enumerate(report.questions) if q.qtype == t] for t in types}
        header = "| Config | " + " | ".join(f"{t} (n={len(by_type[t])})" for t in types) + " |"
        rule = "|---|" + "---|" * len(types)
        for metric in METRICS:
            lines += ["", f"{METRIC_LABELS[metric]} by question type:", "", header, rule]
            for cid in ids:
                cells = " | ".join(
                    _cell(_subset(report.scores[cid][metric], by_type[t])) for t in types
                )
                lines.append(f"| {cid} {report.names[cid]} | {cells} |")
        for metric in METRICS:
            lines += [
                "",
                f"Paired d{METRIC_LABELS[metric]} vs {BASELINE_ID} by question type:",
                "",
                header,
                rule,
            ]
            for cid in ids:
                if cid == BASELINE_ID:
                    continue
                cells = " | ".join(
                    _delta(
                        _subset(report.scores[cid][metric], by_type[t]),
                        _subset(base[metric], by_type[t]),
                    )
                    for t in types
                )
                lines.append(f"| {cid} {report.names[cid]} | {cells} |")

    lines += ["", f"{CV_ID} chosen (k1, b, title weight) per fold and seed:", ""]
    for point, count in report.cv_choices.most_common():
        lines.append(f"- {point}: {count}")
    lines.append("")
    return "\n".join(lines)


def decision_inputs(report: LexicalReport) -> dict[str, tuple[float, float, float]]:
    """Paired A4 minus A0 differences (mean, lo, hi) of Recall@5 and MRR."""
    head, base = report.scores[HEADLINE_ID], report.scores[BASELINE_ID]
    return {
        "r5": paired_bootstrap_ci(head["r5"], base["r5"]),
        "mrr": paired_bootstrap_ci(head["mrr"], base["mrr"]),
    }


def render_decision_inputs(report: LexicalReport) -> str:
    """The A4-minus-A0 quantities the protocol's decision rule reads."""
    inputs = decision_inputs(report)
    lines = [f"Decision-rule inputs, {HEADLINE_ID} minus {BASELINE_ID} (paired, 95% CI):", ""]
    for metric in ("r5", "mrr"):
        mean, lo, hi = inputs[metric]
        lines.append(f"- d{METRIC_LABELS[metric]}: {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    lines.append("")
    return "\n".join(lines)
