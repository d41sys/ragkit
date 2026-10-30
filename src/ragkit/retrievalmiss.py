"""Why gold articles miss the top-5 chunks, and candidate fixes (retrieval_miss_protocol.md).

Retrieval only: no generation and no LLM call. A *unit* is a chunk or a whole article; a
``Config`` fixes the unit, the dense query instruction, the fusion (RRF or convex), the first-stage
depth and the stage (``bm25``, ``dense``, ``hybrid``, ``hybrid+ce``). The default retriever of
``ragkit answer --retriever hybrid+ce`` is ``BASELINE``. Models are injected, so offline tests use
fakes.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ragkit.bm25 import BM25Index, tokenize, top_k_indices
from ragkit.chunking import Chunk, recursive
from ragkit.ctxprefix import (
    SEARCH_CHUNKS,
    SYNTHETIC_MAX_DROP,
    VariantResult,
    act_of,
    build_chunks,
    is_wrong_act,
)
from ragkit.eval import bootstrap_ci, mrr, paired_bootstrap_ci, recall_at_k
from ragkit.lexical import Question
from ragkit.modelsignals import PassageScorer
from ragkit.neural import (
    CE_SLICE,
    DenseSide,
    Hits,
    SentenceEmbedder,
    convex_fuse,
    embed_corpus,
    rerank_head,
    rrf_fuse,
)
from ragkit.pipeline import STOPWORDS

# Frozen by the protocol, committed before any scoring.
WINDOW_CHARS = 1800
WINDOW_OVERLAP = 200
PROD_UNITS = 5  # the first five units are what ``ragkit answer`` turns into article contexts
DEPTHS = (50, 100, 200)
ALPHAS = tuple(i / 10 for i in range(1, 10))
STAGES = ("bm25", "dense", "hybrid", "hybrid+ce")
METRICS = ("mrr", "r5", "g3_list", "g5_list", "g3_prod", "g5_prod")
METRIC_LABELS = {
    "mrr": "MRR",
    "r5": "Recall@5",
    "g3_list": "G@3 (list)",
    "g5_list": "G@5 (list)",
    "g3_prod": "G@3 (prod)",
    "g5_prod": "G@5 (prod)",
}
JACCARD_NEAR_DUPLICATE = 0.30
VOCAB_MISMATCH_BELOW = 0.50
SPLIT_GAP = 0.25
LENGTH_PERCENTILES = (10.0, 90.0)
CANDIDATE_IDS = ("F1", "F2", "F3", "F4", "S")
CANDIDATE_NAMES = {
    "F1": "article units",
    "F2": "depth",
    "F3": "fusion",
    "F4": "no query instruction",
    "S": "joint stack",
}


@dataclass(frozen=True)
class Config:
    """One retriever: the unit, the bge query instruction, the fusion, the depth and the stage.

    ``alpha`` is the convex weight of BM25, ``None`` meaning RRF (k = 60). The defaults are the
    shipped ``hybrid+ce@100`` over chunks (B0 of the protocol).
    """

    unit: str = "chunk"
    instruction: bool = True
    alpha: float | None = None
    depth: int = 100
    stage: str = "hybrid+ce"

    def __post_init__(self) -> None:
        if self.unit not in ("chunk", "article"):
            raise ValueError(f"unit must be chunk or article, got {self.unit!r}")
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {self.stage!r}")
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if self.alpha is not None and not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")

    def at_stage(self, stage: str) -> Config:
        """The same configuration read out at another stage."""
        return Config(self.unit, self.instruction, self.alpha, self.depth, stage)

    def at_depth(self, depth: int) -> Config:
        """The same configuration with another first-stage depth."""
        return Config(self.unit, self.instruction, self.alpha, depth, self.stage)

    def distance(self) -> int:
        """How many of unit, instruction, fusion and depth differ from ``BASELINE``."""
        return sum(
            (
                self.unit != BASELINE.unit,
                self.instruction != BASELINE.instruction,
                self.alpha != BASELINE.alpha,
                self.depth != BASELINE.depth,
            )
        )

    def label(self) -> str:
        """Compact name, e.g. ``article/inst=off/convex0.4/D50/hybrid+ce``."""
        fusion = "rrf" if self.alpha is None else f"convex{self.alpha:.1f}"
        inst = "on" if self.instruction else "off"
        return f"{self.unit}/inst={inst}/{fusion}/D{self.depth}/{self.stage}"

    def to_json(self) -> dict[str, Any]:
        """Plain-dict form for ``retrieval_miss_tuning.json``."""
        return {
            "unit": self.unit,
            "instruction": self.instruction,
            "alpha": self.alpha,
            "depth": self.depth,
            "stage": self.stage,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> Config:
        """Inverse of :meth:`to_json`."""
        alpha = data["alpha"]
        return Config(
            str(data["unit"]),
            bool(data["instruction"]),
            None if alpha is None else float(alpha),
            int(data["depth"]),
            str(data["stage"]),
        )


BASELINE = Config()


def grid() -> list[Config]:
    """The 2 x 2 x 3 x 10 tuning grid (unit, instruction, depth, fusion) at ``hybrid+ce``."""
    return [
        Config(unit, instruction, alpha, depth)
        for unit in ("chunk", "article")
        for instruction in (True, False)
        for depth in DEPTHS
        for alpha in (None, *ALPHAS)
    ]


# ---------------------------------------------------------------------------------------------
# Units, pooled dense side, index
# ---------------------------------------------------------------------------------------------


@dataclass
class PooledDenseSide(DenseSide):
    """Dense side over text windows whose unit score is the maximum over its windows."""

    owner: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    n_units: int = 0

    def unit_scores(self, query: str) -> np.ndarray:
        window_scores = self.matrix @ self.queries[query]
        scores = np.full(self.n_units, -np.inf, dtype=np.float64)
        np.maximum.at(scores, self.owner, window_scores)
        return scores


class UnitIndex:
    """BM25, dense (instruction on / off) and cross-encoder over one list of units.

    ``windows[i]`` holds the texts the cross-encoder sees for unit ``i`` (its own text for a
    chunk; the 1800-character windows for an article); the unit's logit is their maximum.
    """

    def __init__(
        self,
        kind: str,
        units: Sequence[Chunk],
        windows: Sequence[Sequence[str]],
        bm25: BM25Index,
        sides: Mapping[bool, DenseSide],
        ce: PassageScorer | None,
    ) -> None:
        if len(bm25.chunks) != len(units) or len(windows) != len(units):
            raise ValueError("BM25 index and windows must match the units")
        self.kind = kind
        self.units = list(units)
        self.windows = [list(w) for w in windows]
        self.bm25 = bm25
        self.sides = dict(sides)
        self.ce = ce
        self.ce_cache: dict[tuple[str, int], float] = {}
        self._lists: dict[tuple[Any, ...], Hits] = {}

    def prepare(self, queries: Sequence[str]) -> None:
        """Batch-embed the queries on both dense sides."""
        for side in self.sides.values():
            side.prepare(queries)

    def _bm25(self, query: str, depth: int) -> Hits:
        key = ("bm25", query, depth)
        if key not in self._lists:
            self._lists[key] = top_k_indices(self.bm25.score_all(query), depth)
        return self._lists[key]

    def _dense(self, query: str, instruction: bool, depth: int) -> Hits:
        key = ("dense", query, instruction, depth)
        if key not in self._lists:
            self._lists[key] = self.sides[instruction].search(query, depth)
        return self._lists[key]

    def ce_scores(self, query: str, indices: Sequence[int]) -> dict[int, float]:
        """Cross-encoder logit of every unit (max over its windows); each pair scored once."""
        if self.ce is None:
            raise RuntimeError("this index was built without a cross-encoder")
        todo = [i for i in indices if (query, i) not in self.ce_cache]
        flat = [(i, w) for i in todo for w in self.windows[i]]
        best: dict[int, float] = {}
        for start in range(0, len(flat), CE_SLICE):
            part = flat[start : start + CE_SLICE]
            for (i, _), score in zip(part, self.ce.score(query, [w for _, w in part]), strict=True):
                best[i] = max(score, best.get(i, -np.inf))
        for i in todo:
            self.ce_cache[(query, i)] = best[i]
        return {i: self.ce_cache[(query, i)] for i in indices}

    def rank(self, cfg: Config, query: str) -> Hits:
        """Ranked ``(unit index, score)`` of ``cfg``; the list is cut to ``cfg.depth``."""
        first = self._bm25(query, cfg.depth)
        if cfg.stage == "bm25":
            return first
        second = self._dense(query, cfg.instruction, cfg.depth)
        if cfg.stage == "dense":
            return second
        if cfg.alpha is None:
            fused = rrf_fuse([first, second])[: cfg.depth]
        else:
            fused = convex_fuse(first, second, cfg.alpha)[: cfg.depth]
        if cfg.stage == "hybrid":
            return fused
        scores = self.ce_scores(query, [i for i, _ in fused])
        return rerank_head(fused, scores, cfg.depth)

    def doc_ids(self, hits: Sequence[tuple[int, float]]) -> list[str]:
        """Article id of every ranked unit, in rank order (not deduplicated)."""
        return [self.units[i].doc_id for i, _ in hits]


def dedup(ids: Sequence[str]) -> list[str]:
    """First occurrence of every id, in order."""
    return list(dict.fromkeys(ids))


def article_windows(text: str, doc_id: str) -> list[str]:
    """The 1800-character, 200-overlap windows of an article (one window if it is short)."""
    windows = [
        c.text for c in recursive(text, doc_id, chunk_size=WINDOW_CHARS, overlap=WINDOW_OVERLAP)
    ]
    return windows or [text]


def build_indexes(
    documents: Sequence[tuple[str, str]],
    embedder: SentenceEmbedder,
    embedder_off: SentenceEmbedder,
    ce: PassageScorer | None,
    cache_dir: Path | None = None,
) -> dict[str, UnitIndex]:
    """Chunk-unit and article-unit indexes of one corpus (``embedder_off``: no query prefix)."""
    chunks = build_chunks(documents, prefix=False)
    chunk_bm25 = BM25Index()
    chunk_bm25.add(chunks)
    matrix = embed_corpus(embedder, [c.text for c in chunks], cache_dir)
    chunk_sides = {True: DenseSide(embedder, matrix), False: DenseSide(embedder_off, matrix)}
    out = {
        "chunk": UnitIndex("chunk", chunks, [[c.text] for c in chunks], chunk_bm25, chunk_sides, ce)
    }

    articles = [Chunk(doc_id, doc_id, text, 0, len(text)) for doc_id, text in documents]
    windows = [article_windows(text, doc_id) for doc_id, text in documents]
    flat = [w for ws in windows for w in ws]
    owner = np.array([i for i, ws in enumerate(windows) for _ in ws], dtype=np.int64)
    art_matrix = embed_corpus(embedder, flat, cache_dir)
    art_bm25 = BM25Index()
    art_bm25.add(articles)
    art_sides = {
        True: PooledDenseSide(embedder, art_matrix, owner=owner, n_units=len(articles)),
        False: PooledDenseSide(embedder_off, art_matrix, owner=owner, n_units=len(articles)),
    }
    out["article"] = UnitIndex("article", articles, windows, art_bm25, art_sides, ce)
    return out


# ---------------------------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------------------------


def article_lists(
    index: UnitIndex, hits: Sequence[tuple[int, float]]
) -> tuple[list[str], list[str]]:
    """``(list, prod)``: articles of the first 10 units and of the first 5 units, deduplicated."""
    ids = index.doc_ids(hits)
    return dedup(ids[:SEARCH_CHUNKS]), dedup(ids[:PROD_UNITS])


def _gold_in(articles: Sequence[str], relevant: Sequence[str], k: int) -> float:
    return float(bool(set(articles[:k]) & set(relevant)))


def score_one(index: UnitIndex, cfg: Config, question: Question) -> tuple[dict[str, float], bool]:
    """Metric values and the wrong-act flag of one question under ``cfg``."""
    listed, prod = article_lists(index, index.rank(cfg, question.text))
    rel = question.relevant
    values = {
        "mrr": mrr(listed, rel),
        "r5": recall_at_k(listed, rel, k=5),
        "g3_list": _gold_in(listed, rel, 3),
        "g5_list": _gold_in(listed, rel, 5),
        "g3_prod": _gold_in(prod, rel, 3),
        "g5_prod": _gold_in(prod, rel, 5),
    }
    return values, is_wrong_act(listed, rel)


def score_questions(index: UnitIndex, cfg: Config, questions: Sequence[Question]) -> VariantResult:
    """Per-question metrics of ``cfg`` on ``questions`` (same order)."""
    index.prepare([q.text for q in questions])
    out = VariantResult({m: [] for m in METRICS}, [])
    for question in questions:
        values, wrong = score_one(index, cfg, question)
        for metric in METRICS:
            out.scores[metric].append(values[metric])
        out.wrong_act.append(wrong)
    return out


def pool_results(results: Sequence[VariantResult]) -> VariantResult:
    """Concatenate per-set outcomes of one system."""
    scores = {m: [v for r in results for v in r.scores[m]] for m in METRICS}
    return VariantResult(scores, [w for r in results for w in r.wrong_act])


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean (``0.0`` for an empty list)."""
    return float(np.mean(values)) if values else 0.0


# ---------------------------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One grid cell on the tuning sets."""

    config: Config
    mrr_fresh: float
    mrr_synthetic: float
    g5_prod_fresh: float
    g5_prod_synthetic: float

    @property
    def objective(self) -> float:
        """Mean of the two set-level MRR values (the sets weigh equally)."""
        return (self.mrr_fresh + self.mrr_synthetic) / 2.0

    @property
    def g5_prod(self) -> float:
        """Mean of the two set-level ``G@5 (prod)`` values (first tie-break)."""
        return (self.g5_prod_fresh + self.g5_prod_synthetic) / 2.0


def tie_key(cell: Cell) -> tuple[float, float, int, int, int, float]:
    """Sort key: higher objective, higher G@5, nearer B0, smaller depth, RRF, alpha 0.5."""
    cfg = cell.config
    alpha_gap = 0.0 if cfg.alpha is None else round(abs(cfg.alpha - 0.5), 9)
    return (
        -round(cell.objective, 9),
        -round(cell.g5_prod, 9),
        cfg.distance(),
        cfg.depth,
        0 if cfg.alpha is None else 1,
        alpha_gap,
    )


def choose(cells: Mapping[Config, Cell], options: Sequence[Config]) -> Config:
    """The best of ``options`` by ``tie_key`` (the protocol's tuning and tie-break order)."""
    return min((cells[c] for c in options), key=tie_key).config


@dataclass
class Tuning:
    """Every grid cell and the setting each candidate chose."""

    cells: list[Cell]
    chosen: dict[str, Config]
    n_fresh: int
    n_synthetic: int

    def to_json(self) -> dict[str, Any]:
        """Serialisable form committed as ``retrieval_miss_tuning.json``."""
        return {
            "baseline": BASELINE.to_json(),
            "n_fresh": self.n_fresh,
            "n_synthetic": self.n_synthetic,
            "chosen": {k: v.to_json() for k, v in self.chosen.items()},
            "cells": [
                {
                    "config": c.config.to_json(),
                    "mrr_fresh": c.mrr_fresh,
                    "mrr_synthetic": c.mrr_synthetic,
                    "g5_prod_fresh": c.g5_prod_fresh,
                    "g5_prod_synthetic": c.g5_prod_synthetic,
                }
                for c in self.cells
            ],
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> Tuning:
        """Inverse of :meth:`to_json`."""
        cells = [
            Cell(
                Config.from_json(c["config"]),
                float(c["mrr_fresh"]),
                float(c["mrr_synthetic"]),
                float(c["g5_prod_fresh"]),
                float(c["g5_prod_synthetic"]),
            )
            for c in data["cells"]
        ]
        chosen = {k: Config.from_json(v) for k, v in data["chosen"].items()}
        return Tuning(cells, chosen, int(data["n_fresh"]), int(data["n_synthetic"]))


def choose_candidates(cells: Mapping[Config, Cell]) -> dict[str, Config]:
    """F1 to F4 and S from the grid, each chosen on the tuning sets only (protocol, Tuning)."""
    return {
        "F1": Config(unit="article"),
        "F2": choose(cells, [BASELINE.at_depth(d) for d in DEPTHS]),
        "F3": choose(cells, [Config(alpha=a) for a in (None, *ALPHAS)]),
        "F4": Config(instruction=False),
        "S": choose(cells, list(cells)),
    }


def tune(
    eu: Mapping[str, UnitIndex],
    synthetic: Mapping[str, UnitIndex],
    fresh: Sequence[Question],
    synthetic_questions: Sequence[Question],
) -> Tuning:
    """Score the whole grid on the fresh and the synthetic questions and choose the candidates."""
    cells: list[Cell] = []
    for cfg in grid():
        a = score_questions(eu[cfg.unit], cfg, fresh)
        b = score_questions(synthetic[cfg.unit], cfg, synthetic_questions)
        cells.append(
            Cell(
                cfg,
                mean(a.scores["mrr"]),
                mean(b.scores["mrr"]),
                mean(a.scores["g5_prod"]),
                mean(b.scores["g5_prod"]),
            )
        )
    by_config = {c.config: c for c in cells}
    return Tuning(cells, choose_candidates(by_config), len(fresh), len(synthetic_questions))


def render_tuning(tuning: Tuning) -> str:
    """Grid summary and the chosen settings (what the held-out run reads from the JSON)."""
    by_config = {c.config: c for c in tuning.cells}
    base = by_config[BASELINE]
    lines = [
        "### ragkit retrieval-miss: tuning (fresh 29 + synthetic 40; hybrid+ce; no held-out data)",
        "",
        f"Objective = mean of the two set-level MRR values. B0 ({BASELINE.label()}): "
        f"fresh {base.mrr_fresh:.3f}, synthetic {base.mrr_synthetic:.3f}, objective "
        f"{base.objective:.4f}, G@5 (prod) {base.g5_prod_fresh:.3f} / "
        f"{base.g5_prod_synthetic:.3f}.",
        "",
        "| candidate | chosen configuration | fresh MRR | synthetic MRR | objective | "
        "d objective vs B0 | G@5 (prod) fresh / synthetic |",
        "|---|---|---|---|---|---|---|",
    ]
    for cid in CANDIDATE_IDS:
        cfg = tuning.chosen[cid]
        cell = by_config[cfg]
        note = " (= B0: tuning selected the baseline)" if cfg == BASELINE else ""
        lines.append(
            f"| {cid} {CANDIDATE_NAMES[cid]} | `{cfg.label()}`{note} | {cell.mrr_fresh:.3f} | "
            f"{cell.mrr_synthetic:.3f} | {cell.objective:.4f} | "
            f"{cell.objective - base.objective:+.4f} | {cell.g5_prod_fresh:.3f} / "
            f"{cell.g5_prod_synthetic:.3f} |"
        )
    lines += ["", "One-factor grids (every other field at B0), objective:", ""]
    lines.append("| depth | " + " | ".join(str(d) for d in DEPTHS) + " |")
    lines.append("|---|" + "---|" * len(DEPTHS))
    lines.append(
        "| objective | "
        + " | ".join(f"{by_config[BASELINE.at_depth(d)].objective:.4f}" for d in DEPTHS)
        + " |"
    )
    lines += ["", "| fusion | rrf | " + " | ".join(f"{a:.1f}" for a in ALPHAS) + " |"]
    lines.append("|---|---|" + "---|" * len(ALPHAS))
    lines.append(
        "| objective | "
        + " | ".join(f"{by_config[Config(alpha=a)].objective:.4f}" for a in (None, *ALPHAS))
        + " |"
    )
    for unit in ("chunk", "article"):
        for instruction in (True, False):
            cfg = Config(unit, instruction)
            cell = by_config[cfg]
            lines.append(
                f"\n`{cfg.label()}`: fresh {cell.mrr_fresh:.3f}, synthetic "
                f"{cell.mrr_synthetic:.3f}, objective {cell.objective:.4f}"
            )
    top = sorted(tuning.cells, key=tie_key)[:5]
    lines += ["", "Five best of the 120 cells by the protocol's order:", ""]
    lines += [f"- `{c.config.label()}`: objective {c.objective:.4f}" for c in top]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Held-out scoring and the decision rule
# ---------------------------------------------------------------------------------------------


@dataclass
class Row:
    """One scored system: its configuration and its outcomes on every held-out set."""

    name: str
    config: Config
    sets: dict[str, VariantResult]
    synthetic: VariantResult | None = None

    def pooled(self, exclude: Sequence[str] = ()) -> VariantResult:
        """Outcomes pooled over the held-out sets (minus ``exclude``)."""
        return pool_results([r for n, r in self.sets.items() if n not in exclude])


@dataclass
class HeldOut:
    """Every row of the single held-out run."""

    candidates: dict[str, Row]
    sensitivity: list[Row]
    stage_rows: list[Row]
    questions: dict[str, int]
    n_synthetic: int


@dataclass(frozen=True)
class Verdict:
    """Outcome of the decision rule for one candidate."""

    status: str
    d_mrr: tuple[float, float, float]
    wrong: int
    base_wrong: int
    d_g5: float
    d_syn: tuple[float, float, float]
    reasons: tuple[str, ...]

    @property
    def adopted(self) -> bool:
        """True when all four conditions of the protocol hold."""
        return self.status == "adopted"


def decide(base: Row, cand: Row) -> Verdict:
    """The protocol's four conditions on the pooled held-out sets and the synthetic guard."""
    if base.synthetic is None or cand.synthetic is None:
        raise ValueError("both rows need synthetic outcomes for the guard")
    b, c = base.pooled(), cand.pooled()
    d_mrr = paired_bootstrap_ci(c.scores["mrr"], b.scores["mrr"])
    d_g5 = mean(c.scores["g5_prod"]) - mean(b.scores["g5_prod"])
    d_syn = paired_bootstrap_ci(cand.synthetic.scores["mrr"], base.synthetic.scores["mrr"])
    wrong, base_wrong = sum(c.wrong_act), sum(b.wrong_act)
    reasons: list[str] = []
    if d_mrr[1] <= 0.0:
        reasons.append("pooled MRR CI lower bound is not above 0")
    if wrong > base_wrong:
        reasons.append(f"wrong-act {wrong} above B0's {base_wrong}")
    if d_g5 < 0.0:
        reasons.append("pooled G@5 (prod) falls")
    guard_fails = d_syn[0] < -SYNTHETIC_MAX_DROP or d_syn[2] < 0.0
    if guard_fails:
        reasons.append("synthetic guard fails")
    if not reasons:
        status = "adopted"
    elif d_mrr[2] < 0.0 or guard_fails:
        status = "rejected"
    else:
        status = "not adopted (no evidence)"
    return Verdict(status, d_mrr, wrong, base_wrong, d_g5, d_syn, tuple(reasons))


def winner(verdicts: Mapping[str, Verdict]) -> str | None:
    """Adopted candidate with the highest pooled MRR difference, ``None`` when none is adopted."""
    adopted = [k for k, v in verdicts.items() if v.adopted]
    return max(adopted, key=lambda k: verdicts[k].d_mrr[0]) if adopted else None


def run_heldout(
    eu: Mapping[str, UnitIndex],
    synthetic: Mapping[str, UnitIndex],
    sets: Mapping[str, Sequence[Question]],
    synthetic_questions: Sequence[Question],
    tuning: Tuning,
) -> HeldOut:
    """Score B0 and every candidate once on the held-out sets (plus the non-gating rows)."""

    def row(name: str, cfg: Config, *, guard: bool = False) -> Row:
        result = {n: score_questions(eu[cfg.unit], cfg, qs) for n, qs in sets.items()}
        syn = score_questions(synthetic[cfg.unit], cfg, synthetic_questions) if guard else None
        return Row(name, cfg, result, syn)

    candidates: dict[str, Row] = {"B0": row("B0", BASELINE, guard=True)}
    seen: dict[Config, Row] = {BASELINE: candidates["B0"]}
    for cid in CANDIDATE_IDS:
        cfg = tuning.chosen[cid]
        if cfg not in seen:
            seen[cfg] = row(cid, cfg, guard=True)
        if cfg != BASELINE:
            candidates[cid] = replace(seen[cfg], name=cid)
    sensitivity = [
        row(f"depth {d} (chunks, RRF)", BASELINE.at_depth(d)) for d in DEPTHS if d != 100
    ]
    f3 = tuning.chosen["F3"]
    sensitivity += [
        row("hybrid RRF (no CE)", BASELINE.at_stage("hybrid")),
        row(f"hybrid {_fusion_name(f3)} (no CE)", Config(alpha=f3.alpha, stage="hybrid")),
    ]
    stage_rows = [
        row(f"{unit} {stage}", Config(unit=unit, stage=stage))
        for unit in ("chunk", "article")
        for stage in STAGES
    ]
    return HeldOut(
        candidates,
        sensitivity,
        stage_rows,
        {n: len(qs) for n, qs in sets.items()},
        len(synthetic_questions),
    )


def _fusion_name(cfg: Config) -> str:
    return "RRF" if cfg.alpha is None else f"convex {cfg.alpha:.1f}"


def _cell(values: Sequence[float]) -> str:
    m, lo, hi = bootstrap_ci(values)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _delta(values: Sequence[float], base: Sequence[float]) -> str:
    m, lo, hi = paired_bootstrap_ci(values, base)
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _counts(values: Sequence[float], base: Sequence[float]) -> str:
    better = sum(1 for v, b in zip(values, base, strict=True) if v > b + 1e-12)
    worse = sum(1 for v, b in zip(values, base, strict=True) if v < b - 1e-12)
    return f"{better}/{worse}/{len(values) - better - worse}"


def _table(rows: Sequence[Row], base: Row, exclude: Sequence[str] = ()) -> list[str]:
    n = len(base.pooled(exclude).wrong_act)
    lines = [
        "| system | configuration | "
        + " | ".join(METRIC_LABELS[m] for m in METRICS)
        + " | wrong-act |",
        "|---|---|" + "---|" * (len(METRICS) + 1),
    ]
    for r in rows:
        p = r.pooled(exclude)
        cells = " | ".join(_cell(p.scores[m]) for m in METRICS)
        lines.append(f"| {r.name} | `{r.config.label()}` | {cells} | {sum(p.wrong_act)} / {n} |")
    return lines


def _diff_table(rows: Sequence[Row], base: Row, exclude: Sequence[str] = ()) -> list[str]:
    b = base.pooled(exclude)
    lines = [
        "| system | "
        + " | ".join(f"d {METRIC_LABELS[m]}" for m in METRICS)
        + " | MRR better/worse/tied | d wrong-act |",
        "|---|" + "---|" * (len(METRICS) + 2),
    ]
    for r in rows:
        p = r.pooled(exclude)
        cells = " | ".join(_delta(p.scores[m], b.scores[m]) for m in METRICS)
        lines.append(
            f"| {r.name} | {cells} | {_counts(p.scores['mrr'], b.scores['mrr'])} | "
            f"{sum(p.wrong_act) - sum(b.wrong_act):+d} |"
        )
    return lines


def render_heldout(run: HeldOut, tuning: Tuning) -> str:
    """Per-set and pooled tables, paired differences, the guard and the decision rule's verdicts."""
    base = run.candidates["B0"]
    cand_rows = list(run.candidates.values())
    verdicts = {k: decide(base, r) for k, r in run.candidates.items() if k != "B0"}
    n_pool = sum(run.questions.values())
    lines = [
        "### ragkit retrieval-miss: held-out scoring (hybrid+ce; one run)",
        "",
        "mean [bootstrap 95% CI over questions, 2000 resamples, seed 0]; paired differences are "
        "system minus B0 on the same questions. `list` = first 10 units deduplicated to articles "
        "(MRR, Recall@5, wrong-act); `prod` = first 5 units deduplicated (the article context of "
        "`ragkit answer`). G@k = a gold article among the first k articles.",
        "",
        "Chosen on fresh 29 + synthetic 40 only (`retrieval_miss_tuning.json`):",
        "",
    ]
    lines += [
        f"- {cid} {CANDIDATE_NAMES[cid]}: `{tuning.chosen[cid].label()}`"
        + (" (= B0)" if tuning.chosen[cid] == BASELINE else "")
        for cid in CANDIDATE_IDS
    ]
    lines += [
        "",
        f"#### Pooled held-out ({n_pool} questions: "
        + ", ".join(f"{n} {c}" for n, c in run.questions.items())
        + ")",
        "",
    ]
    lines += _table(cand_rows, base)
    lines += ["", "Paired differences against B0, pooled:", ""]
    lines += _diff_table([r for r in cand_rows if r.name != "B0"], base)
    for name in run.questions:
        lines += ["", f"#### {name} ({run.questions[name]} questions)", ""]
        single = [Row(r.name, r.config, {name: r.sets[name]}) for r in cand_rows]
        lines += _table(single, single[0])
        lines += ["", *_diff_table(single[1:], single[0])]
    others = [n for n in run.questions if n != "original 42"]
    if "original 42" in run.questions:
        lines += [
            "",
            "#### Sensitivity (non-gating): pooled without the original 42 "
            f"({sum(run.questions[n] for n in others)} questions; the original 42 are where the "
            "misses were found)",
            "",
        ]
        lines += _table(cand_rows, base, exclude=("original 42",))
        lines += [
            "",
            *_diff_table([r for r in cand_rows if r.name != "B0"], base, exclude=("original 42",)),
        ]
    lines += ["", "#### Synthetic guard (tuning set; consistency check, not evidence)", ""]
    lines += [
        "| candidate | synthetic MRR | paired d MRR vs B0 |",
        "|---|---|---|",
    ]
    assert base.synthetic is not None
    for key, r in run.candidates.items():
        assert r.synthetic is not None
        diff = (
            "baseline"
            if key == "B0"
            else _delta(r.synthetic.scores["mrr"], base.synthetic.scores["mrr"])
        )
        lines.append(f"| {key} | {_cell(r.synthetic.scores['mrr'])} | {diff} |")
    lines += ["", "#### Decision rule (pooled held-out; four conditions, protocol)", ""]
    for key, v in verdicts.items():
        name = CANDIDATE_NAMES.get(key, key)
        why = "; ".join(v.reasons) if v.reasons else "all four conditions hold"
        lines.append(
            f"- **{key} {name}** (`{run.candidates[key].config.label()}`): {v.status}. "
            f"dMRR {v.d_mrr[0]:+.3f} [{v.d_mrr[1]:+.3f}, {v.d_mrr[2]:+.3f}], wrong-act "
            f"{v.wrong} vs {v.base_wrong}, dG@5 (prod) {v.d_g5:+.3f}, synthetic dMRR "
            f"{v.d_syn[0]:+.3f} [{v.d_syn[1]:+.3f}, {v.d_syn[2]:+.3f}]. {why}."
        )
    best = winner(verdicts)
    lines += [
        "",
        f"Default: {'**' + best + '**' if best else 'unchanged (B0)'}"
        + (f" (`{run.candidates[best].config.label()}`)." if best else "."),
        "",
        "#### Sensitivity rows (non-gating; nothing is selected on them)",
        "",
    ]
    lines += _table(run.sensitivity, base)
    lines += ["", *_diff_table(run.sensitivity, base)]
    lines += ["", "#### Stage rows at B0's settings (both unit types; non-gating)", ""]
    lines += _table(run.stage_rows, base)
    lines += ["", *_diff_table(run.stage_rows, base)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------------------------


def content_terms(text: str) -> set[str]:
    """Lowercase alphanumeric tokens minus ``STOPWORDS`` (as ``query_support``)."""
    return {t for t in tokenize(text) if t not in STOPWORDS}


def bigrams(text: str) -> set[tuple[str, str]]:
    """Word bigrams of a text."""
    return set(itertools.pairwise(tokenize(text)))


def jaccard(a: set[Any], b: set[Any]) -> float:
    """Jaccard similarity of two sets (0.0 when both are empty)."""
    return len(a & b) / len(a | b) if a | b else 0.0


def first_rank(items: Sequence[str], gold: Sequence[str]) -> int | None:
    """1-based position of the first gold item, ``None`` when absent."""
    wanted = set(gold)
    for i, item in enumerate(items):
        if item in wanted:
            return i + 1
    return None


def length_bounds(questions: Sequence[Question]) -> tuple[float, float]:
    """10th and 90th percentile of the word counts of ``questions``."""
    words = [len(q.text.split()) for q in questions]
    lo, hi = np.percentile(words, LENGTH_PERCENTILES)
    return float(lo), float(hi)


@dataclass(frozen=True)
class SystemRank:
    """Rank of the best gold article and of its best unit under one system."""

    article: int | None
    unit: int | None

    def cell(self, absent: str) -> str:
        """``article (unit)`` or ``absent``."""
        if self.article is None:
            return absent
        return f"{self.article} (c{self.unit})"


@dataclass
class Miss:
    """One question whose gold articles are all outside the first five units of B0."""

    set_name: str
    question: Question
    ranks: dict[str, SystemRank]
    flags: dict[str, bool]
    primary: str
    coverage_article: float
    coverage_chunk: float
    jaccard_any: float
    jaccard_top5: float
    same_act: bool
    words: int


def rank_in(index: UnitIndex, cfg: Config, question: Question) -> SystemRank:
    """Rank of the best gold article (and unit) under ``cfg`` over the whole returned list."""
    ids = index.doc_ids(index.rank(cfg, question.text))
    art = first_rank(dedup(ids), question.relevant)
    unit = first_rank(ids, question.relevant)
    return SystemRank(art, unit)


def primary_cause(flags: Mapping[str, bool]) -> str:
    """First applicable cause in the protocol's order, else ``unexplained``."""
    for name in ("vocabulary mismatch", "chunk boundary split", "near-duplicate outranked"):
        if flags[name]:
            return name
    if flags["query too short"] or flags["query too long"]:
        return "query too short" if flags["query too short"] else "query too long"
    return "unexplained"


def diagnose_question(
    set_name: str,
    question: Question,
    chunk: UnitIndex,
    article: UnitIndex,
    texts: Mapping[str, str],
    grams: Mapping[str, set[tuple[str, str]]],
    bounds: tuple[float, float],
) -> Miss | None:
    """Diagnosis of one question; ``None`` when a gold article is among B0's first five units."""
    hits = chunk.rank(BASELINE, question.text)
    _, prod = article_lists(chunk, hits)
    if first_rank(prod, question.relevant) is not None:
        return None
    n = len(chunk.units)
    ranks = {
        "bm25": rank_in(chunk, Config(stage="bm25", depth=n), question),
        "dense": rank_in(chunk, Config(stage="dense", depth=n), question),
        "hybrid": rank_in(chunk, BASELINE.at_stage("hybrid"), question),
        "hybrid+ce": rank_in(chunk, BASELINE, question),
        "article hybrid+ce (F1)": rank_in(article, Config(unit="article"), question),
    }
    terms = content_terms(question.text)
    gold = [g for g in question.relevant if g in texts]

    def cover(text_terms: set[str]) -> float:
        return len(terms & text_terms) / len(terms) if terms else 0.0

    art_cov = {g: cover(content_terms(texts[g])) for g in gold}
    best_gold = max(gold, key=lambda g: art_cov[g])
    chunk_cov = max(
        (cover(content_terms(u.text)) for u in chunk.units if u.doc_id == best_gold), default=0.0
    )
    above = dedup(chunk.doc_ids(hits))
    rank = first_rank(above, question.relevant)
    outrankers = above if rank is None else above[: rank - 1]
    top5 = prod

    def best_pair(pool: Sequence[str]) -> tuple[float, bool]:
        best, same = 0.0, False
        for o in pool:
            for g in gold:
                sim = jaccard(grams[o], grams[g])
                if sim > best:
                    best, same = sim, act_of(o) == act_of(g)
        return best, same

    j_any, same_act = best_pair(outrankers)
    j_top5, _ = best_pair(top5)
    words = len(question.text.split())
    flags = {
        "vocabulary mismatch": art_cov[best_gold] < VOCAB_MISMATCH_BELOW,
        "chunk boundary split": art_cov[best_gold] - chunk_cov >= SPLIT_GAP,
        "near-duplicate outranked": j_any >= JACCARD_NEAR_DUPLICATE,
        "query too short": words < bounds[0],
        "query too long": words > bounds[1],
    }
    return Miss(
        set_name,
        question,
        ranks,
        flags,
        primary_cause(flags),
        art_cov[best_gold],
        chunk_cov,
        j_any,
        j_top5,
        same_act,
        words,
    )


def diagnose(
    sets: Mapping[str, Sequence[Question]],
    chunk: UnitIndex,
    article: UnitIndex,
    documents: Sequence[tuple[str, str]],
    all_questions: Sequence[Question],
) -> tuple[list[Miss], dict[str, int]]:
    """Every miss of every set (in set order) and the number of questions per set."""
    texts = dict(documents)
    grams = {doc_id: bigrams(text) for doc_id, text in documents}
    bounds = length_bounds(all_questions)
    for index in (chunk, article):
        index.prepare([q.text for qs in sets.values() for q in qs])
    misses: list[Miss] = []
    for name, qs in sets.items():
        for q in qs:
            miss = diagnose_question(name, q, chunk, article, texts, grams, bounds)
            if miss is not None:
                misses.append(miss)
    return misses, {n: len(qs) for n, qs in sets.items()}


def render_diagnosis(
    misses: Sequence[Miss],
    sizes: Mapping[str, int],
    bounds: tuple[float, float],
    focus: str = "original 42",
) -> str:
    """Per-miss table for ``focus``, cause counts for every set."""
    lines = [
        "### ragkit retrieval-miss: diagnosis (descriptive; B0 = hybrid+ce@100 over chunks)",
        "",
        "A miss = no gold article among the first five chunks of B0. Cell = rank of the best gold "
        "article among distinct articles (rank of its best unit, c<n>); `bm25` and `dense` rank "
        "all chunks, `hybrid` and `hybrid+ce` the top-100 pool (absent = not in the pool), `F1` "
        "ranks article units. Query length bounds (10th / 90th percentile of the 159 EU "
        f"questions): {bounds[0]:.1f} / {bounds[1]:.1f} words.",
        "",
        "| set | misses | "
        + " | ".join(
            [
                "vocabulary mismatch",
                "chunk boundary split",
                "near-duplicate outranked",
                "query too short",
                "query too long",
                "unexplained",
            ]
        )
        + " |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, n in sizes.items():
        mine = [m for m in misses if m.set_name == name]
        primary = Counter(m.primary for m in mine)
        cells = " | ".join(
            str(primary.get(c, 0))
            for c in (
                "vocabulary mismatch",
                "chunk boundary split",
                "near-duplicate outranked",
                "query too short",
                "query too long",
                "unexplained",
            )
        )
        lines.append(f"| {name} ({n}) | {len(mine)} | {cells} |")
    lines += ["", "(cells above: primary cause, protocol order; flags below are not exclusive)", ""]
    focus_misses = [m for m in misses if m.set_name == focus]
    lines += [f"#### Misses on {focus} ({len(focus_misses)} of {sizes.get(focus, 0)})", ""]
    lines += [
        "| # | question | gold | bm25 | dense | hybrid | h+ce | F1 | words | flags | primary |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, m in enumerate(focus_misses, 1):
        flagged = ", ".join(k for k, v in m.flags.items() if v) or "none"
        lines.append(
            f"| {i} | {m.question.text[:90]} | {', '.join(m.question.relevant)} | "
            + " | ".join(
                m.ranks[s].cell("absent")
                for s in ("bm25", "dense", "hybrid", "hybrid+ce", "article hybrid+ce (F1)")
            )
            + f" | {m.words} | {flagged} | {m.primary} |"
        )
    lines += [
        "",
        "| # | term coverage gold article | best gold chunk | max bigram Jaccard (outrankers) "
        "| (top-5 articles) | same act |",
        "|---|---|---|---|---|---|",
    ]
    for i, m in enumerate(focus_misses, 1):
        lines.append(
            f"| {i} | {m.coverage_article:.2f} | {m.coverage_chunk:.2f} | {m.jaccard_any:.2f} | "
            f"{m.jaccard_top5:.2f} | {'yes' if m.same_act else 'no'} |"
        )
    lines += ["", "Which system finds the gold article at all (best rank, any set):", ""]
    for system in ("bm25", "dense", "hybrid", "hybrid+ce", "article hybrid+ce (F1)"):
        in5 = sum(1 for m in misses if (r := m.ranks[system].article) is not None and r <= 5)
        lines.append(
            f"- {system}: gold article within the first 5 articles for {in5} of "
            f"{len(misses)} misses"
        )
    return "\n".join(lines)


def dump_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write ``data`` as sorted, indented JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
