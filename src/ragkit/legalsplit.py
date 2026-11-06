"""LegalBench-RAG-mini: query-prefix removal (arm A) and document routing (arm B).

Protocol, split, design choices and the decision rule: ``docs/legalsplit_protocol.md`` (committed
before any number). Every mini query is ``Consider ... ; <question>``; arm A searches with the
question only, arm B routes the prefix to documents by file name / opening text and searches only
the top ``n`` of them (it uses document metadata, so it is not comparable to published rows).
Only the cached document matrices of ``e5-small-v2`` are read; a missing matrix is an error.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ragkit.analysis import lucene_english
from ragkit.beir import LUCENE_B, LUCENE_K1
from ragkit.bm25 import BM25Index, top_k_indices
from ragkit.chunking import Chunk
from ragkit.legalbench import (
    KS,
    SOURCES,
    LegalResult,
    MiniBenchmark,
    MiniTest,
    Span,
    char_precision,
    char_recall,
    chunk_corpus,
)
from ragkit.neural import DenseSide, corpus_cache_path, e5_small

SPLIT_SEED = 20261106
DEV_FRACTION = 0.2
PROFILES = ("name", "name+open500", "name+open2000", "open2000")
N_DOCS = (1, 2)
QUERY_MODES = ("full", "question")
RETRIEVERS = ("bm25", "dense-e5")
CHUNKERS = ("naive500", "recursive500")
PRIMARY = (("bm25", "naive500"), ("dense-e5", "recursive500"))
DEPTH = max(KS)
K8 = KS.index(8)
# Committed full-query rows (docs/sota-runs/legalbench_mini.txt, neural_legalbench_mini.txt), %.
EXPECTED_BASELINE_R8 = {
    ("bm25", "naive500"): 13.37,
    ("bm25", "recursive500"): 11.84,
    ("dense-e5", "naive500"): 19.69,
    ("dense-e5", "recursive500"): 20.40,
}
REPRODUCTION_TOLERANCE = 0.05


def split_query(query: str) -> tuple[str, str]:
    """``(prefix, question)``: text before / after the first ``;``; no ``;`` keeps it whole."""
    if ";" not in query:
        return "", query.strip()
    prefix, question = query.split(";", 1)
    prefix = prefix.strip()
    if prefix.startswith("Consider"):
        prefix = prefix[len("Consider") :].strip()
    return prefix, question.strip()


def dev_test_split(tests: Sequence[MiniTest]) -> tuple[list[int], list[int]]:
    """Indices of the dev (20 % per source, seed-fixed) and test queries, in benchmark order."""
    rng = random.Random(SPLIT_SEED)
    dev: list[int] = []
    test: list[int] = []
    for source in SOURCES:
        indices = [i for i, t in enumerate(tests) if t.source == source]
        rng.shuffle(indices)
        n_dev = round(DEV_FRACTION * len(indices))
        dev += indices[:n_dev]
        test += indices[n_dev:]
    return sorted(dev), sorted(test)


# ---------------------------------------------------------------------------------------------
# Document routing
# ---------------------------------------------------------------------------------------------


def doc_profile(path: str, text: str, profile: str) -> str:
    """The text a document is matched by: file name and/or its opening characters."""
    name = PurePosixPath(path).name.removesuffix(".txt")
    if profile == "name":
        return name
    if profile == "name+open500":
        return f"{name}\n{text[:500]}"
    if profile == "name+open2000":
        return f"{name}\n{text[:2000]}"
    if profile == "open2000":
        return text[:2000]
    raise ValueError(f"unknown profile {profile!r}")


class DocRouter:
    """Ranks documents against a query prefix by BM25 over one profile text per document."""

    def __init__(self, corpus: dict[str, str], profile: str) -> None:
        self.paths = sorted(corpus)
        self.index = BM25Index(k1=LUCENE_K1, b=LUCENE_B, analyzer=lucene_english)
        profiles = [doc_profile(p, corpus[p], profile) for p in self.paths]
        self.index.add(
            [
                Chunk(f"profile-{i}", path, text, 0, len(text))
                for i, (path, text) in enumerate(zip(self.paths, profiles, strict=True))
            ]
        )
        self._cache: dict[str, list[str]] = {}

    def rank(self, prefix: str) -> list[str]:
        """Documents with a positive routing score, best first (ties: lower sorted index)."""
        cached = self._cache.get(prefix)
        if cached is None:
            hits = top_k_indices(self.index.score_all(prefix), len(self.paths))
            cached = self._cache[prefix] = [self.paths[i] for i, _ in hits]
        return cached


def routing_accuracy(
    router: DocRouter, tests: Sequence[MiniTest], indices: Sequence[int]
) -> dict[str, tuple[float, float]]:
    """Per source and ``ALL`` (mean of the four): share with the gold document at top-1, top-2."""
    hits: dict[str, list[tuple[bool, bool]]] = {s: [] for s in SOURCES}
    for i in indices:
        test = tests[i]
        ranked = router.rank(split_query(test.query)[0])
        gold = test.truth[0][0]
        hits[test.source].append((ranked[:1] == [gold], gold in ranked[:2]))
    out = {
        s: (float(np.mean([h[0] for h in rows])), float(np.mean([h[1] for h in rows])))
        for s, rows in hits.items()
    }
    out["ALL"] = (
        float(np.mean([out[s][0] for s in SOURCES])),
        float(np.mean([out[s][1] for s in SOURCES])),
    )
    return out


# ---------------------------------------------------------------------------------------------
# Retrieval cells and arms
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One way of querying a cell: search text and (optionally) routing to ``n_docs`` documents."""

    name: str
    query: str  # "full" or "question"
    n_docs: int | None = None  # None: all documents, no routing


BASELINE = Arm("baseline", "full")
ARM_A = Arm("A", "question")


def arm_b(n_docs: int, query: str) -> Arm:
    """Arm B with the dev-chosen ``n_docs`` and in-document query."""
    return Arm("B", query, n_docs)


@dataclass
class Cell:
    """One retriever on one chunker: BM25 index or dense side over the chunks."""

    retriever: str
    chunker: str
    chunks: list[Chunk]
    bm25: BM25Index | None = None
    dense: DenseSide | None = None
    _by_doc: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        by_doc: dict[str, list[int]] = {}
        for i, chunk in enumerate(self.chunks):
            by_doc.setdefault(chunk.doc_id, []).append(i)
        self._by_doc = {d: np.asarray(v, dtype=np.int64) for d, v in by_doc.items()}

    @property
    def name(self) -> str:
        """``<retriever>-<chunker>``."""
        return f"{self.retriever}-{self.chunker}"

    def prepare(self, queries: Sequence[str]) -> None:
        """Embed the dense queries in one batch (no-op for BM25)."""
        if self.dense is not None:
            self.dense.prepare(queries)

    def rank(self, query: str, docs: Sequence[str] | None) -> list[Span]:
        """Top ``DEPTH`` chunk spans, over all documents or only the chunks of ``docs``."""
        allowed = None if docs is None else np.concatenate([self._by_doc[d] for d in docs])
        order: list[int]
        if self.bm25 is not None:
            scores = self.bm25.score_all(query)
            if allowed is not None:
                kept = np.zeros_like(scores)
                kept[allowed] = scores[allowed]
                scores = kept
            order = [i for i, _ in top_k_indices(scores, DEPTH)]
        else:
            assert self.dense is not None
            self.dense.prepare([query])
            vec = self.dense.queries[query]
            if allowed is None:
                allowed = np.arange(len(self.chunks))
                sims = self.dense.matrix @ vec
            else:
                sims = self.dense.matrix[allowed] @ vec
            top = np.lexsort((allowed, -sims))[:DEPTH]
            order = [int(allowed[j]) for j in top]
        return [(self.chunks[i].doc_id, self.chunks[i].start, self.chunks[i].end) for i in order]


@dataclass
class ArmRun:
    """Per-query scores of one arm on one cell for a set of queries."""

    result: LegalResult
    fallbacks: int = 0

    def recall8(self) -> float:
        """Recall@8 ALL in percent."""
        return 100.0 * self.result.all_mean("recall")[K8]


def evaluate(
    cell: Cell,
    tests: Sequence[MiniTest],
    indices: Sequence[int],
    arm: Arm,
    router: DocRouter | None = None,
) -> ArmRun:
    """Score ``arm`` on the queries ``tests[i]``, ``i`` in ``indices`` (in that order)."""
    plans: list[tuple[MiniTest, str, list[str] | None]] = []
    fallbacks = 0
    for i in indices:
        test = tests[i]
        prefix, question = split_query(test.query)
        text = question if arm.query == "question" else test.query
        docs: list[str] | None = None
        if arm.n_docs is not None:
            if router is None:
                raise ValueError("a routed arm needs a router")
            docs = router.rank(prefix)[: arm.n_docs]
            if not docs:
                docs, text = None, test.query
                fallbacks += 1
        plans.append((test, text, docs))
    cell.prepare([text for _, text, _ in plans])
    result = LegalResult(f"{arm.name}-{cell.name}", n_chunks=len(cell.chunks))
    for test, text, docs in plans:
        ranked = cell.rank(text, docs)
        if len(ranked) < DEPTH:
            result.short_lists += 1
        result.precision.setdefault(test.source, []).append(
            [char_precision(ranked[:k], test.truth) for k in KS]
        )
        result.recall.setdefault(test.source, []).append(
            [char_recall(ranked[:k], test.truth) for k in KS]
        )
    return ArmRun(result, fallbacks)


def build_cells(
    bench: MiniBenchmark, cache_dir: Path, device: str | None = None
) -> dict[tuple[str, str], Cell]:
    """The four cells; the e5 document matrices must already be in ``cache_dir``."""
    embedder = e5_small(device)
    cells: dict[tuple[str, str], Cell] = {}
    for chunker in CHUNKERS:
        chunks = chunk_corpus(bench.corpus, chunker)
        bm25 = BM25Index(k1=LUCENE_K1, b=LUCENE_B, analyzer=lucene_english)
        bm25.add(chunks)
        path = corpus_cache_path(embedder, [c.text for c in chunks], cache_dir)
        if not path.is_file():
            raise FileNotFoundError(f"cached e5 document matrix not found: {path}")
        matrix: np.ndarray = np.load(path)
        if len(matrix) != len(chunks):
            raise ValueError(f"{path} has {len(matrix)} rows for {len(chunks)} chunks")
        cells[("bm25", chunker)] = Cell("bm25", chunker, chunks, bm25=bm25)
        cells[("dense-e5", chunker)] = Cell(
            "dense-e5", chunker, chunks, dense=DenseSide(embedder, matrix)
        )
    return cells


# ---------------------------------------------------------------------------------------------
# Statistics and the rule
# ---------------------------------------------------------------------------------------------


def paired_all_ci(
    arm: LegalResult,
    base: LegalResult,
    *,
    k: int = 8,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean, 95 % CI of ``arm - base`` Recall@``k`` ALL in points.

    Queries are resampled within each source with the same resample for both systems, and the
    four source means are averaged with equal weight (as ``LegalResult.all_ci``).
    """
    col = KS.index(k)
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_resamples)
    point = 0.0
    for source in SOURCES:
        diff = np.asarray(arm.recall[source])[:, col] - np.asarray(base.recall[source])[:, col]
        idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
        boot += diff[idx].mean(axis=1) / len(SOURCES)
        point += float(diff.mean()) / len(SOURCES)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return 100.0 * point, 100.0 * float(lo), 100.0 * float(hi)


def verdict(lo: float, hi: float) -> str:
    """The pre-registered rule: ``improvement`` (lo > 0), ``worse`` (hi < 0), else null."""
    if lo > 0.0:
        return "improvement"
    if hi < 0.0:
        return "worse"
    return "no detectable effect"


def overall_verdict(primary: dict[tuple[str, str], str]) -> str:
    """An arm is an improvement overall only if it is one in every primary cell."""
    improved = [f"{r}-{c}" for (r, c), v in primary.items() if v == "improvement"]
    if len(improved) == len(primary):
        return "improvement overall (both primary cells)"
    if improved:
        return "improvement in " + ", ".join(improved) + " only"
    worse = [f"{r}-{c}" for (r, c), v in primary.items() if v == "worse"]
    if worse:
        return "worse in " + ", ".join(worse) + "; otherwise no detectable effect"
    return "no detectable effect"


# ---------------------------------------------------------------------------------------------
# Tuning (dev queries only)
# ---------------------------------------------------------------------------------------------


@dataclass
class Tuning:
    """Dev-chosen settings: profile, and per retriever ``(n_docs, in-document query)``."""

    profile: str
    profile_table: dict[str, dict[str, float]]
    settings: dict[str, tuple[int, str]]
    grid: dict[str, dict[str, float]]
    n_dev: int

    def to_json(self) -> dict[str, Any]:
        """JSON form read back by ``ragkit legalsplit score``."""
        return {
            "split_seed": SPLIT_SEED,
            "dev_fraction": DEV_FRACTION,
            "n_dev": self.n_dev,
            "chosen": {
                "profile": self.profile,
                "settings": {r: {"n": n, "query": q} for r, (n, q) in self.settings.items()},
            },
            "profile_table": self.profile_table,
            "grid": self.grid,
        }


def chosen_from_json(data: dict[str, Any]) -> tuple[str, dict[str, tuple[int, str]]]:
    """``(profile, settings)`` of a tuning JSON written by ``Tuning.to_json``."""
    chosen = data["chosen"]
    return str(chosen["profile"]), {
        r: (int(v["n"]), str(v["query"])) for r, v in chosen["settings"].items()
    }


def choose_profile(
    bench: MiniBenchmark, dev: Sequence[int]
) -> tuple[str, dict[str, dict[str, float]]]:
    """Profile with the best dev document top-1 accuracy (ties: earlier in ``PROFILES``)."""
    table: dict[str, dict[str, float]] = {}
    for profile in PROFILES:
        acc = routing_accuracy(DocRouter(bench.corpus, profile), bench.tests, dev)
        table[profile] = {"top1": acc["ALL"][0], "top2": acc["ALL"][1]}
    best = max(PROFILES, key=lambda p: (table[p]["top1"], -PROFILES.index(p)))
    return best, table


def tune(bench: MiniBenchmark, cells: dict[tuple[str, str], Cell], dev: Sequence[int]) -> Tuning:
    """Choose the profile, then ``n`` and the in-document query per retriever, on ``dev`` only."""
    profile, table = choose_profile(bench, dev)
    router = DocRouter(bench.corpus, profile)
    settings: dict[str, tuple[int, str]] = {}
    grid: dict[str, dict[str, float]] = {}
    for retriever in RETRIEVERS:
        scores: dict[tuple[int, str], float] = {}
        for n in N_DOCS:
            for query in QUERY_MODES:
                runs = [
                    evaluate(cells[(retriever, c)], bench.tests, dev, arm_b(n, query), router)
                    for c in CHUNKERS
                ]
                scores[(n, query)] = float(np.mean([r.recall8() for r in runs]))
        grid[retriever] = {f"n={n},{q}": v for (n, q), v in scores.items()}
        settings[retriever] = max(scores, key=lambda s: (scores[s], s[0] == 1, s[1] == "question"))
    return Tuning(profile, table, settings, grid, len(dev))


def _source_counts(bench: MiniBenchmark, indices: Sequence[int]) -> str:
    counts = {s: sum(1 for i in indices if bench.tests[i].source == s) for s in SOURCES}
    return ", ".join(f"{s} {n}" for s, n in counts.items())


def render_tuning(tuning: Tuning, bench: MiniBenchmark, dev: Sequence[int]) -> str:
    """Markdown report of the dev run."""
    lines = [
        "### ragkit legalsplit tune: dev queries only",
        "",
        f"Dev split: seed {SPLIT_SEED}, {DEV_FRACTION:.0%} per source: {len(dev)} queries "
        f"({_source_counts(bench, dev)}). No test query is scored here.",
        "",
        "**Document routing on dev** (gold document at top-1 / top-2; mean of the four sources)",
        "",
        "| profile | top-1 | top-2 |",
        "|---|---|---|",
    ]
    for profile in PROFILES:
        row = tuning.profile_table[profile]
        mark = " (chosen)" if profile == tuning.profile else ""
        lines.append(f"| {profile}{mark} | {100 * row['top1']:.1f} | {100 * row['top2']:.1f} |")
    lines += [
        "",
        "**Arm B on dev: Recall@8 ALL (%), mean of the two chunkers**",
        "",
        "| retriever | " + " | ".join(f"n={n}, {q}" for n in N_DOCS for q in QUERY_MODES) + " |",
        "|---|" + "---|" * (len(N_DOCS) * len(QUERY_MODES)),
    ]
    for retriever in RETRIEVERS:
        row_cells = []
        for n in N_DOCS:
            for q in QUERY_MODES:
                mark = "*" if tuning.settings[retriever] == (n, q) else ""
                row_cells.append(f"{tuning.grid[retriever][f'n={n},{q}']:.2f}{mark}")
        lines.append(f"| {retriever} | " + " | ".join(row_cells) + " |")
    lines += ["", "`*` = chosen (best dev Recall@8; ties: n = 1, then question)."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Held-out scoring
# ---------------------------------------------------------------------------------------------


@dataclass
class CellScore:
    """Baseline, A and B of one cell on the test queries and on all 776 (post-hoc)."""

    test: dict[str, ArmRun]
    full: dict[str, ArmRun]


def score(
    bench: MiniBenchmark,
    cells: dict[tuple[str, str], Cell],
    profile: str,
    settings: dict[str, tuple[int, str]],
    test: Sequence[int],
) -> tuple[dict[tuple[str, str], CellScore], DocRouter]:
    """Score baseline, A and B in every cell with the frozen settings."""
    router = DocRouter(bench.corpus, profile)
    everyone = list(range(len(bench.tests)))
    out: dict[tuple[str, str], CellScore] = {}
    for key, cell in cells.items():
        n, query = settings[cell.retriever]
        arms = {"baseline": BASELINE, "A": ARM_A, "B": arm_b(n, query)}
        out[key] = CellScore(
            test={a: evaluate(cell, bench.tests, test, arm, router) for a, arm in arms.items()},
            full={a: evaluate(cell, bench.tests, everyone, arm, router) for a, arm in arms.items()},
        )
    return out, router


ARM_LABEL = {
    "baseline": "baseline (full query, all documents)",
    "A": "A: question only, all documents",
    "B": "B: routed to top-n documents (uses document metadata, not comparable to published rows)",
}


def _recall_cells(run: ArmRun) -> list[str]:
    cis = run.result.all_ci("recall")
    return [
        f"{100 * m:.2f} [{100 * lo:.1f}, {100 * hi:.1f}]"
        for m, (lo, hi) in zip(run.result.all_mean("recall"), cis, strict=True)
    ]


def render_score(
    bench: MiniBenchmark,
    scores: dict[tuple[str, str], CellScore],
    router: DocRouter,
    profile: str,
    settings: dict[str, tuple[int, str]],
    test: Sequence[int],
) -> str:
    """Markdown report of the held-out run, with the pre-registered verdicts."""
    chosen = ", ".join(f"{r}: n = {n}, in-document query = {q}" for r, (n, q) in settings.items())
    lines = [
        "### ragkit legalsplit score: test queries (query-prefix removal and document routing)",
        "",
        f"Test split: {len(test)} queries ({_source_counts(bench, test)}); dev-chosen settings: "
        f"profile `{profile}`, {chosen}. Values in %. Delta = arm minus the same-cell baseline on "
        "Recall@8 ALL, 95% paired bootstrap CI (2000 resamples, seed 0, queries resampled within "
        "each source).",
    ]
    verdicts: dict[str, dict[tuple[str, str], str]] = {"A": {}, "B": {}}
    for key, cs in scores.items():
        base = cs.test["baseline"]
        lines += [
            "",
            f"#### {key[0]}-{key[1]}, test",
            "",
            "| arm | " + " | ".join(f"R@{k}" for k in KS) + " | P@8 | delta R@8 [CI] | rule |",
            "|---|" + "---|" * (len(KS) + 3),
        ]
        for arm, run in cs.test.items():
            prec8 = 100 * run.result.all_mean("precision")[K8]
            if arm == "baseline":
                delta, rule = "baseline", ""
            else:
                mean, lo, hi = paired_all_ci(run.result, base.result)
                rule = verdict(lo, hi)
                verdicts[arm][key] = rule
                delta = f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            recall = " | ".join(_recall_cells(run))
            lines.append(f"| {ARM_LABEL[arm]} | {recall} | {prec8:.2f} | {delta} | {rule} |")
        per_source = "; ".join(
            f"{arm}: "
            + ", ".join(f"{s} {100 * run.result.source_mean('recall', s)[K8]:.2f}" for s in SOURCES)
            for arm, run in cs.test.items()
        )
        short = ", ".join(f"{arm} {run.result.short_lists}" for arm, run in cs.test.items())
        lines += [
            "",
            f"Recall@8 per source: {per_source}.",
            "",
            f"B fallbacks (no routed document): {cs.test['B'].fallbacks}; "
            f"queries with < {DEPTH} hits: {short}.",
        ]
    primary_label = ", ".join(f"{r}-{c}" for r, c in PRIMARY)
    lines += [
        "",
        f"**Pre-registered verdicts** (primary cells: {primary_label})",
        "",
        "| arm | " + " | ".join(f"{r}-{c}" for r, c in scores) + " | overall |",
        "|---|" + "---|" * (len(scores) + 1),
    ]
    for arm in ("A", "B"):
        row = " | ".join(verdicts[arm][c] + (" (primary)" if c in PRIMARY else "") for c in scores)
        overall = overall_verdict({c: verdicts[arm][c] for c in PRIMARY})
        lines.append(f"| {arm} | {row} | {overall} |")

    acc = routing_accuracy(router, bench.tests, test)
    columns = (*SOURCES, "ALL")
    lines += [
        "",
        f"**Document routing on test** (profile `{profile}`; gold document at top-1 / top-2, %)",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
        "| " + " | ".join(f"{100 * acc[s][0]:.1f} / {100 * acc[s][1]:.1f}" for s in columns) + " |",
        "",
        "**Post-hoc, all 776 queries (includes the 156 dev queries; only to line the rows up with "
        "the published full-sample numbers). Recall@8 ALL, %**",
        "",
        "| cell | baseline | committed row | A | B |",
        "|---|---|---|---|---|",
    ]
    for key, cs in scores.items():
        gap = cs.full["baseline"].recall8() - EXPECTED_BASELINE_R8[key]
        status = "reproduced" if abs(gap) <= REPRODUCTION_TOLERANCE else "NOT reproduced"
        row_cells = []
        for run in cs.full.values():
            lo, hi = run.result.all_ci("recall")[K8]
            row_cells.append(f"{run.recall8():.2f} [{100 * lo:.1f}, {100 * hi:.1f}]")
        lines.append(
            f"| {key[0]}-{key[1]} | {row_cells[0]} | {EXPECTED_BASELINE_R8[key]:.2f}, gap "
            f"{gap:+.2f}, {status} | {row_cells[1]} | {row_cells[2]} |"
        )
    return "\n".join(lines)
