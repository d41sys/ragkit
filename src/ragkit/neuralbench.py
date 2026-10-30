"""Benchmark runners for the neural systems (protocol: ``docs/neural_protocol.md``).

BEIR (four sets, nDCG@10 and Recall@100), LegalBench-RAG-mini (character-overlap Recall and
Precision@k) and the four EU question sets (Recall@5, MRR, nDCG@10, wrong-act count with paired
CIs against BM25). The hybrid convex weight is chosen on each benchmark's dev split only.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ragkit.analysis import lucene_english
from ragkit.beir import (
    LUCENE_B,
    LUCENE_K1,
    NDCG_CUT,
    PYSERINI_2CR,
    RECALL_CUT,
    BeirData,
    SystemResult,
    _flat_chunks,
    load_beir,
    ndcg_cut,
    recall_at,
)
from ragkit.bm25 import BM25Index
from ragkit.chunking import Chunk
from ragkit.ctxprefix import SEARCH_CHUNKS, VariantResult, build_chunks, is_wrong_act
from ragkit.eval import bootstrap_ci, paired_bootstrap_ci
from ragkit.legalbench import (
    KS,
    SOURCES,
    LegalResult,
    MiniBenchmark,
    MiniTest,
    char_precision,
    char_recall,
    chunk_corpus,
)
from ragkit.lexical import METRIC_LABELS, METRICS, Question, rank_documents, score_question
from ragkit.neural import (
    ALPHA_GRID,
    LEVEL_SYSTEMS,
    NeuralEngine,
    alpha_of_best,
    build_engine,
    convex_fuse,
)

NEURAL_BEIR = ("scifact", "nfcorpus", "fiqa", "arguana")
# Dev split of the hybrid convex weight; ArguAna ships a test split only (RRF only).
BEIR_DEV_SPLIT = {"scifact": "train", "nfcorpus": "dev", "fiqa": "dev"}
REPRODUCTION_TOLERANCE = 0.015

# Published, same split and metric (docs/neural_protocol.md): (nDCG@10, Recall@100).
# MTEB results repo, commit 7818c6fe5639c8396c2c62879c137077e9a9c60b.
MTEB_BGE_SMALL = {
    "scifact": (0.7127, 0.945),
    "nfcorpus": (0.3426, 0.3116),
    "fiqa": (0.4034, 0.6955),
    "arguana": (0.6035, 0.9879),
}
MTEB_E5_SMALL = {
    "scifact": (0.6885, 0.9277),
    "nfcorpus": (0.3245, 0.298),
    "fiqa": (0.3743, 0.6819),
    "arguana": (0.4179, 0.9239),
}
BEIR_PAPER_BM25_CE = {"scifact": 0.688, "nfcorpus": 0.350, "fiqa": 0.347, "arguana": 0.311}

_PYSERINI_BM25_FLAT, _PYSERINI_BGE_BASE = 0, 4


def _delta_cell(values: Sequence[float], base: Sequence[float]) -> str:
    mean, lo, hi = paired_bootstrap_ci(values, base)
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _cell(values: Sequence[float]) -> str:
    mean, lo, hi = bootstrap_ci(values)
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def _systems(level: str, alpha: float | None) -> list[str]:
    return [s for s in LEVEL_SYSTEMS[level] if s != "hybrid-convex" or alpha is not None]


# ---------------------------------------------------------------------------------------------
# BEIR
# ---------------------------------------------------------------------------------------------


@dataclass
class NeuralBeirRun:
    """Per-query scores of every system on one BEIR test set, with the dev-chosen alpha."""

    data: BeirData
    results: dict[str, SystemResult]
    alpha: float | None = None
    dev_name: str = ""
    dev_scores: dict[float, float] = field(default_factory=dict)
    encode_seconds: float = 0.0


def _beir_engine(
    data: BeirData, level: str, cache_dir: Path | None, device: str | None
) -> tuple[NeuralEngine, float]:
    chunks = _flat_chunks(data)
    bm25 = BM25Index(k1=LUCENE_K1, b=LUCENE_B, analyzer=lucene_english)
    bm25.add(chunks)
    started = time.perf_counter()
    engine = build_engine(chunks, bm25, level, cache_dir=cache_dir, device=device)
    return engine, time.perf_counter() - started


def choose_beir_alpha(
    engine: NeuralEngine, dev: BeirData, max_queries: int | None = None
) -> dict[float, float]:
    """Mean dev nDCG@10 of ``hybrid-convex`` for every alpha of the grid."""
    items = list(dev.queries.items())[:max_queries]
    engine.prepare([text for _, text in items])
    totals = dict.fromkeys(ALPHA_GRID, 0.0)
    for qid, text in items:
        cands = engine.candidates(text, exclude_id=qid)
        for alpha in ALPHA_GRID:
            fused = convex_fuse(cands.bm25, cands.dense["bge"], alpha)
            ranked = [engine.chunks[i].id for i, _ in fused[:NDCG_CUT]]
            totals[alpha] += ndcg_cut(ranked, dev.qrels[qid])
    return {a: t / max(1, len(items)) for a, t in totals.items()}


def run_beir_neural(
    name: str,
    root: Path,
    level: str,
    *,
    cache_dir: Path | None = None,
    device: str | None = None,
    max_queries: int | None = None,
    download: bool = True,
) -> NeuralBeirRun:
    """Score every system of ``level`` on the test split of one BEIR dataset."""
    data = load_beir(name, root, split="test", download=download)
    engine, encode_seconds = _beir_engine(data, level, cache_dir, device)
    run = NeuralBeirRun(data, {}, encode_seconds=encode_seconds)
    if level in ("hybrid", "hybrid+ce") and name in BEIR_DEV_SPLIT:
        split = BEIR_DEV_SPLIT[name]
        run.dev_name = f"{name} {split}"
        run.dev_scores = choose_beir_alpha(
            engine, load_beir(name, root, split=split, download=download), max_queries
        )
        run.alpha = alpha_of_best(run.dev_scores)
    systems = _systems(level, run.alpha)
    run.results = {s: SystemResult(s) for s in systems}
    items = list(data.queries.items())[:max_queries]
    engine.prepare([text for _, text in items])
    for qid, text in items:
        cands = engine.candidates(text, exclude_id=qid)
        rels = data.qrels[qid]
        for system in systems:
            started = time.perf_counter()
            hits = engine.system(system, text, cands, run.alpha)
            ranked = [engine.chunks[i].id for i, _ in hits[:RECALL_CUT]]
            result = run.results[system]
            result.ndcg.append(ndcg_cut(ranked, rels))
            result.recall.append(recall_at(ranked, rels))
            result.seconds += time.perf_counter() - started
    return run


def render_neural_beir(run: NeuralBeirRun) -> str:
    """Our systems with CIs and paired nDCG@10 differences, then the published rows."""
    name = run.data.name
    lines = [
        f"### ragkit beir --retriever: {name} (test)",
        "",
        f"{len(next(iter(run.results.values())).ndcg)} test queries, {len(run.data.doc_ids)} "
        f"documents; mean [bootstrap 95% CI over queries]; dense encoding "
        f"{run.encode_seconds:.0f} s (cached matrices reload in seconds).",
        "",
        "| System | nDCG@10 | paired dnDCG@10 vs bm25 | Recall@100 | s (ranking, incl. CE) |",
        "|---|---|---|---|---|",
    ]
    base = run.results["bm25"].ndcg
    for system, res in run.results.items():
        delta = "baseline" if system == "bm25" else _delta_cell(res.ndcg, base)
        lines.append(
            f"| {system} | {_cell(res.ndcg)} | {delta} | {_cell(res.recall)} | {res.seconds:.0f} |"
        )
    if run.alpha is not None:
        best = run.dev_scores[run.alpha]
        lines += [
            "",
            f"`hybrid-convex` alpha = {run.alpha:.1f} (weight of BM25), chosen on {run.dev_name} "
            f"dev nDCG@10 {best:.3f} (grid "
            + ", ".join(f"{a:.1f}: {s:.3f}" for a, s in sorted(run.dev_scores.items()))
            + ").",
        ]
    elif "hybrid" in run.results:
        lines += ["", "No dev split for this dataset: RRF only, no `hybrid-convex` row."]
    lines += [
        "",
        "Published, same dataset, split and metrics (nDCG@10 / Recall@100):",
        "",
        "| System | nDCG@10 | Recall@100 |",
        "|---|---|---|",
    ]
    pyserini = PYSERINI_2CR[name]
    bm25_n, bm25_r = pyserini[_PYSERINI_BM25_FLAT]
    base_n, base_r = pyserini[_PYSERINI_BGE_BASE]
    lines += [
        f"| Pyserini BM25 flat | {bm25_n:.3f} | {bm25_r:.3f} |",
        f"| bge-small-en-v1.5 (MTEB) | {MTEB_BGE_SMALL[name][0]:.4f} | "
        f"{MTEB_BGE_SMALL[name][1]:.4f} |",
        f"| e5-small-v2 (MTEB) | {MTEB_E5_SMALL[name][0]:.4f} | {MTEB_E5_SMALL[name][1]:.4f} |",
        f"| BM25+CE (BEIR paper; rerank of BM25 top 100, nDCG@10 only) | "
        f"{BEIR_PAPER_BM25_CE[name]:.3f} | = BM25 |",
        f"| BGE-base-en-v1.5 (Pyserini) | {base_n:.3f} | {base_r:.3f} |",
    ]
    if "dense" in run.results:
        ours = sum(run.results["dense"].ndcg) / len(run.results["dense"].ndcg)
        gap = ours - MTEB_BGE_SMALL[name][0]
        verdict = "within" if abs(gap) <= REPRODUCTION_TOLERANCE else "OUTSIDE"
        lines += [
            "",
            f"Reproduction check: our bge-small `dense` nDCG@10 {ours:.4f} vs MTEB "
            f"{MTEB_BGE_SMALL[name][0]:.4f}, gap {gap:+.4f}, {verdict} the pre-registered "
            f"{REPRODUCTION_TOLERANCE} tolerance.",
        ]
    return "\n".join(lines)


def beir_mean_ci(
    runs: Sequence[NeuralBeirRun],
    system: str,
    base: str | None = None,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean over datasets of mean nDCG@10 (or of the paired ``system - base`` difference), 95% CI.

    The queries of each dataset are resampled independently (same resample for ``system`` and
    ``base``) and the dataset means are averaged, so the interval is that of the unweighted
    mean over datasets.
    """
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_resamples)
    point = 0.0
    for run in runs:
        values = np.asarray(run.results[system].ndcg, dtype=np.float64)
        if base is not None:
            values = values - np.asarray(run.results[base].ndcg, dtype=np.float64)
        idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
        boot += values[idx].mean(axis=1) / len(runs)
        point += float(values.mean()) / len(runs)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return point, float(lo), float(hi)


def render_beir_mean(runs: Sequence[NeuralBeirRun]) -> str:
    """Mean nDCG@10 over the datasets run, with paired bootstrap CIs against ``bm25``."""
    names = ", ".join(run.data.name for run in runs)
    lines = [
        f"### ragkit beir --retriever: mean over {len(runs)} datasets ({names})",
        "",
        "Unweighted mean of the per-dataset nDCG@10; 95% bootstrap CI (2000 resamples, seed 0) "
        "resampling the queries of each dataset independently; the paired difference uses the "
        "same resample for both systems.",
        "",
        "| System | mean nDCG@10 [95% CI] | paired d vs bm25 [95% CI] |",
        "|---|---|---|",
    ]
    for system in runs[0].results:
        mean = beir_mean_ci(runs, system)
        delta = (
            "baseline"
            if system == "bm25"
            else "{:+.3f} [{:+.3f}, {:+.3f}]".format(*beir_mean_ci(runs, system, "bm25"))
        )
        lines.append(f"| {system} | {mean[0]:.3f} [{mean[1]:.3f}, {mean[2]:.3f}] | {delta} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# LegalBench-RAG-mini
# ---------------------------------------------------------------------------------------------


def _limit_per_source(tests: Sequence[MiniTest], max_queries: int | None) -> list[MiniTest]:
    if max_queries is None:
        return list(tests)
    count = dict.fromkeys(SOURCES, 0)
    kept: list[MiniTest] = []
    for test in tests:
        if count[test.source] < max_queries:
            count[test.source] += 1
            kept.append(test)
    return kept


def run_legal_neural(
    bench: MiniBenchmark,
    level: str,
    *,
    chunkers: Sequence[str] = ("naive500", "recursive500"),
    cache_dir: Path | None = None,
    device: str | None = None,
    max_queries: int | None = None,
) -> list[LegalResult]:
    """Every system of ``level`` (no convex row: no dev split) with each chunker."""
    out: list[LegalResult] = []
    for chunker in chunkers:
        chunks = chunk_corpus(bench.corpus, chunker)
        bm25 = BM25Index(k1=LUCENE_K1, b=LUCENE_B, analyzer=lucene_english)
        bm25.add(chunks)
        engine = build_engine(chunks, bm25, level, cache_dir=cache_dir, device=device)
        systems = _systems(level, None)
        results = {s: LegalResult(f"{s}-{chunker}", n_chunks=len(chunks)) for s in systems}
        tests = _limit_per_source(bench.tests, max_queries)
        engine.prepare([t.query for t in tests])
        for test in tests:
            cands = engine.candidates(test.query)
            for system in systems:
                started = time.perf_counter()
                hits = engine.system(system, test.query, cands)
                ranked = [
                    (engine.chunks[i].doc_id, engine.chunks[i].start, engine.chunks[i].end)
                    for i, _ in hits[: max(KS)]
                ]
                res = results[system]
                if len(ranked) < max(KS):
                    res.short_lists += 1
                res.precision.setdefault(test.source, []).append(
                    [char_precision(ranked[:k], test.truth) for k in KS]
                )
                res.recall.setdefault(test.source, []).append(
                    [char_recall(ranked[:k], test.truth) for k in KS]
                )
                res.seconds += time.perf_counter() - started
        out.extend(results.values())
    return out


# ---------------------------------------------------------------------------------------------
# EU question sets
# ---------------------------------------------------------------------------------------------


@dataclass
class EuRun:
    """Per-question outcomes of every system on the dev set and each held-out set."""

    alpha: float | None
    dev_scores: dict[float, float]
    dev: dict[str, VariantResult]
    sets: dict[str, dict[str, VariantResult]]
    questions: dict[str, list[Question]]
    dev_label: str = "original 42 (dev)"


def _eu_docs(engine: NeuralEngine, hits: Sequence[tuple[int, float]]) -> list[str]:
    ranked = [(engine.chunks[i], s) for i, s in hits[:SEARCH_CHUNKS]]
    return rank_documents(ranked)


def _eu_scores(
    engine: NeuralEngine,
    questions: Sequence[Question],
    systems: Sequence[str],
    alpha: float | None,
) -> dict[str, VariantResult]:
    out = {s: VariantResult({m: [] for m in METRICS}, []) for s in systems}
    engine.prepare([q.text for q in questions])
    for question in questions:
        cands = engine.candidates(question.text)
        for system in systems:
            docs = _eu_docs(engine, engine.system(system, question.text, cands, alpha))
            for metric, value in score_question(docs, question.relevant).items():
                out[system].scores[metric].append(value)
            out[system].wrong_act.append(is_wrong_act(docs, question.relevant))
    return out


def run_eu_neural(
    documents: Sequence[tuple[str, str]],
    dev: Sequence[Question],
    sets: dict[str, Sequence[Question]],
    level: str,
    *,
    cache_dir: Path | None = None,
    device: str | None = None,
) -> EuRun:
    """Score the systems of ``level`` on the EU sets; ``alpha`` is chosen on ``dev`` only."""
    chunks: list[Chunk] = build_chunks(documents, prefix=False)
    bm25 = BM25Index()
    bm25.add(chunks)
    engine = build_engine(chunks, bm25, level, cache_dir=cache_dir, device=device)
    alpha: float | None = None
    dev_scores: dict[float, float] = {}
    if level in ("hybrid", "hybrid+ce"):
        engine.prepare([q.text for q in dev])
        totals = dict.fromkeys(ALPHA_GRID, 0.0)
        for question in dev:
            cands = engine.candidates(question.text)
            for a in ALPHA_GRID:
                hits = convex_fuse(cands.bm25, cands.dense["bge"], a)
                docs = _eu_docs(engine, hits)
                totals[a] += score_question(docs, question.relevant)["ndcg"]
        dev_scores = {a: t / len(dev) for a, t in totals.items()}
        alpha = alpha_of_best(dev_scores)
    systems = _systems(level, alpha)
    return EuRun(
        alpha=alpha,
        dev_scores=dev_scores,
        dev=_eu_scores(engine, dev, systems, alpha),
        sets={n: _eu_scores(engine, qs, systems, alpha) for n, qs in sets.items()},
        questions={"dev": list(dev), **{n: list(qs) for n, qs in sets.items()}},
    )


def pool(results: Sequence[dict[str, VariantResult]]) -> dict[str, VariantResult]:
    """Concatenate per-set outcomes (same systems) into one pooled outcome per system."""
    pooled: dict[str, VariantResult] = {}
    for system in results[0]:
        scores = {m: [v for r in results for v in r[system].scores[m]] for m in METRICS}
        pooled[system] = VariantResult(scores, [w for r in results for w in r[system].wrong_act])
    return pooled


def recommendation(pooled: dict[str, VariantResult], system: str) -> tuple[bool, str]:
    """The decision rule: paired MRR CI lower bound above 0 and wrong-act not above BM25's."""
    mean, lo, hi = paired_bootstrap_ci(pooled[system].scores["mrr"], pooled["bm25"].scores["mrr"])
    wrong, base_wrong = sum(pooled[system].wrong_act), sum(pooled["bm25"].wrong_act)
    ok = lo > 0.0 and wrong <= base_wrong
    return ok, (
        f"dMRR {mean:+.3f} [{lo:+.3f}, {hi:+.3f}], wrong-act {wrong} vs bm25 {base_wrong}: "
        + ("recommended" if ok else "not recommended")
    )


def _render_eu_set(label: str, n: int, results: dict[str, VariantResult]) -> list[str]:
    base = results["bm25"]
    lines = [
        f"#### {label} ({n} questions)",
        "",
        "| System | Recall@5 | MRR | nDCG@10 | paired dRecall@5 vs bm25 | paired dMRR vs bm25 "
        "| wrong-act top-1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for system, res in results.items():
        is_base = system == "bm25"
        d5 = "baseline" if is_base else _delta_cell(res.scores["r5"], base.scores["r5"])
        dm = "baseline" if is_base else _delta_cell(res.scores["mrr"], base.scores["mrr"])
        lines.append(
            f"| {system} | {_cell(res.scores['r5'])} | {_cell(res.scores['mrr'])} | "
            f"{_cell(res.scores['ndcg'])} | {d5} | {dm} | {sum(res.wrong_act)} / {n} |"
        )
    return [*lines, ""]


def render_eu_neural(run: EuRun) -> str:
    """Tables per set, the pooled held-out table and the decision rule's outcome."""
    lines = [
        "### ragkit neural: EU retrieval (chunks 500/50, top-10 chunks deduplicated to articles)",
        "",
        "mean [bootstrap 95% CI over questions, 2000 resamples, seed 0]; "
        "paired differences are system minus bm25 on the same questions.",
        "",
    ]
    if run.alpha is not None:
        lines += [
            f"`hybrid-convex` alpha = {run.alpha:.1f} (weight of BM25), chosen on the "
            f"{run.dev_label} set by nDCG@10 (grid "
            + ", ".join(f"{a:.1f}: {s:.3f}" for a, s in sorted(run.dev_scores.items()))
            + ").",
            "",
        ]
    n_of = {name: len(qs) for name, qs in run.questions.items()}
    lines += _render_eu_set(f"{run.dev_label}: dev only, not a test", n_of["dev"], run.dev)
    for name, results in run.sets.items():
        lines += _render_eu_set(f"{name} (held out)", n_of[name], results)
    pooled = pool(list(run.sets.values()))
    n_pool = sum(n_of[n] for n in run.sets)
    lines += _render_eu_set("pooled held-out sets", n_pool, pooled)
    lines += ["Decision rule (pooled held-out; MRR CI lower bound > 0 and wrong-act <= bm25):", ""]
    lines += [f"- `{s}`: {recommendation(pooled, s)[1]}" for s in pooled if s != "bm25"]
    lines += ["", "Metrics: " + ", ".join(METRIC_LABELS[m] for m in METRICS) + "."]
    return "\n".join(lines)
