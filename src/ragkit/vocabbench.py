"""Vocabulary-mismatch fixes: larger models and LLM query expansion (K-R5).

Protocol, grid, decision rule and predictions: ``docs/vocab_protocol.md`` (committed before this
code). Every system is the shipped ``hybrid+ce@100`` over the K-R3 chunks with four fields varied:
the embedder, the cross-encoder, query2doc-style expansion of the first-stage query by an LLM, and
which query the cross-encoder scores. Retrieval only; expansions are read from a call cache.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ragkit.bm25 import BM25Index, top_k_indices
from ragkit.chunking import Chunk
from ragkit.ctxprefix import SEARCH_CHUNKS, SYNTHETIC_MAX_DROP, is_wrong_act
from ragkit.eval import bootstrap_ci, mrr, paired_bootstrap_ci, recall_at_k
from ragkit.lexical import Question
from ragkit.llm import CachedMessages
from ragkit.modelsignals import PassageScorer
from ragkit.neural import (
    BGE_QUERY_INSTRUCTION,
    CE_SLICE,
    DEPTH,
    DenseSide,
    Hits,
    SentenceEmbedder,
    rerank_head,
    rrf_fuse,
)
from ragkit.parent import (
    ARTICLE_BUDGET_TOKENS,
    PACK_CHUNKS,
    PACK_OVERFLOW,
    Passage,
    parent_articles,
)

# Frozen by the protocol (pinned Hugging Face revisions, read 2026-11-17).
BGE_BASE_MODEL = "BAAI/bge-base-en-v1.5"  # MIT
BGE_BASE_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-base"  # MIT
BGE_RERANKER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
EMBEDDERS = ("bge-small", "bge-base")
RERANKERS = ("minilm", "bge-reranker")
EXPANSION_MODEL = "qwen3.6:27b-ctx4k"
EXPANSION_NUM_CTX = 4096
EXPANSION_MAX_TOKENS = 256
EXPANSION_TEMPLATE = (
    "You write one short passage that answers the question as it would be worded in {domain}. "
    "Use the formal terms such a text would use. Write only the passage, 80 to 120 words: no "
    "preamble, no title, no citations, no mention that the passage is hypothetical."
)
EU_DOMAIN = "EU electricity-market legislation or in a technical document on energy flexibility"
EXPANSION_SYSTEM = EXPANSION_TEMPLATE.format(domain=EU_DOMAIN)
BM25_QUERY_REPEATS = 5  # query2doc's sparse form: the question five times, then the passage
PROD_UNITS = 5
METRICS = ("mrr", "r5", "g3_prod", "g5_prod", "pack")
METRIC_LABELS = {
    "mrr": "MRR",
    "r5": "Recall@5",
    "g3_prod": "G@3 (prod)",
    "g5_prod": "G@5 (prod)",
    "pack": "pack recall",
}
CANDIDATE_IDS = ("M1", "M2", "Q1", "S")
TIE = 1e-9


@dataclass(frozen=True)
class Cell:
    """One system of the grid; the defaults are the shipped ``hybrid+ce@100`` (B0)."""

    embedder: str = "bge-small"
    reranker: str = "minilm"
    expansion: bool = False
    ce_expanded: bool = False

    def __post_init__(self) -> None:
        if self.embedder not in EMBEDDERS:
            raise ValueError(f"embedder must be one of {EMBEDDERS}, got {self.embedder!r}")
        if self.reranker not in RERANKERS:
            raise ValueError(f"reranker must be one of {RERANKERS}, got {self.reranker!r}")
        if self.ce_expanded and not self.expansion:
            raise ValueError("ce_expanded needs expansion")

    def distance(self) -> int:
        """Fields different from B0 (``ce_query`` counts only through ``expansion``)."""
        return sum(
            (
                self.embedder != BASELINE.embedder,
                self.reranker != BASELINE.reranker,
                self.expansion != BASELINE.expansion,
            )
        )

    def label(self) -> str:
        """Compact name, e.g. ``bge-base/bge-reranker/exp=on/ce=original``."""
        exp = "on" if self.expansion else "off"
        ce = "expanded" if self.ce_expanded else "original"
        return f"{self.embedder}/{self.reranker}/exp={exp}/ce={ce}"

    def to_json(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder,
            "reranker": self.reranker,
            "expansion": self.expansion,
            "ce_expanded": self.ce_expanded,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> Cell:
        return Cell(
            str(data["embedder"]),
            str(data["reranker"]),
            bool(data["expansion"]),
            bool(data["ce_expanded"]),
        )


BASELINE = Cell()


def grid() -> list[Cell]:
    """The 2 x 2 x (1 + 2) = 12 cells."""
    return [
        Cell(e, r, exp, ce)
        for e in EMBEDDERS
        for r in RERANKERS
        for exp, ce in ((False, False), (True, False), (True, True))
    ]


# ---------------------------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------------------------


def expand(
    client: CachedMessages, questions: Sequence[str], system_text: str = EXPANSION_SYSTEM
) -> dict[str, str]:
    """One generated passage per distinct question (``""`` for an empty reply).

    A failed live call raises: the expansion stage runs before any score exists, so it is simply
    run again. An offline client raises ``CacheMiss`` for a question without a stored reply.
    """
    system = [{"type": "text", "text": system_text}]
    return {
        q: client.complete("expand", system, q, EXPANSION_MAX_TOKENS).text.strip()
        for q in dict.fromkeys(questions)
    }


def queries(cell: Cell, question: str, passage: str) -> tuple[str, str, str]:
    """``(bm25 query, dense query, cross-encoder query)``; an empty passage falls back to ``q``."""
    if not cell.expansion or not passage:
        return question, question, question
    expanded = f"{question}\n{passage}"
    bm25 = " ".join([question] * BM25_QUERY_REPEATS) + "\n" + passage
    return bm25, expanded, expanded if cell.ce_expanded else question


# ---------------------------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------------------------


class Engine:
    """BM25, one dense side per embedder and one cross-encoder per reranker over one corpus."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        articles: Mapping[str, str],
        sides: Mapping[str, DenseSide],
        rerankers: Mapping[str, PassageScorer],
    ) -> None:
        self.chunks = list(chunks)
        self.articles = dict(articles)
        self.bm25 = BM25Index()
        self.bm25.add(self.chunks)
        self.sides = dict(sides)
        self.rerankers = dict(rerankers)
        self.ce_cache: dict[tuple[str, str, int], float] = {}
        self.ce_seconds: dict[str, float] = dict.fromkeys(self.rerankers, 0.0)

    def prepare(
        self, cells: Sequence[Cell], texts: Sequence[str], passages: Mapping[str, str]
    ) -> None:
        """Batch-embed every dense query the cells need."""
        for name, side in self.sides.items():
            side.prepare(
                [
                    queries(c, t, passages.get(t, ""))[1]
                    for c in cells
                    if c.embedder == name
                    for t in texts
                ]
            )

    def ce_scores(self, reranker: str, query: str, indices: Sequence[int]) -> dict[int, float]:
        """Cross-encoder logits of ``(query, chunk)``; each pair is scored once per process."""
        todo = [i for i in indices if (reranker, query, i) not in self.ce_cache]
        scorer = self.rerankers[reranker]
        started = time.perf_counter()
        for start in range(0, len(todo), CE_SLICE):
            part = todo[start : start + CE_SLICE]
            texts = [self.chunks[i].text for i in part]
            for i, score in zip(part, scorer.score(query, texts), strict=True):
                self.ce_cache[(reranker, query, i)] = score
        self.ce_seconds[reranker] += time.perf_counter() - started
        return {i: self.ce_cache[(reranker, query, i)] for i in indices}

    def rank(self, cell: Cell, question: str, passage: str) -> Hits:
        """B0's pipeline with the cell's models and queries."""
        bm25_q, dense_q, ce_q = queries(cell, question, passage)
        first = top_k_indices(self.bm25.score_all(bm25_q), DEPTH)
        second = self.sides[cell.embedder].search(dense_q, DEPTH)
        fused = rrf_fuse([first, second])[:DEPTH]
        scores = self.ce_scores(cell.reranker, ce_q, [i for i, _ in fused])
        return rerank_head(fused, scores, DEPTH)


def _gold_in(articles: Sequence[str], relevant: Sequence[str], k: int) -> float:
    return float(bool(set(articles[:k]) & set(relevant)))


def score_one(
    engine: Engine, cell: Cell, question: Question, passage: str
) -> tuple[dict[str, float], bool]:
    """Metric values and the wrong-act flag of one question."""
    hits = engine.rank(cell, question.text, passage)
    ids = [engine.chunks[i].doc_id for i, _ in hits]
    listed = list(dict.fromkeys(ids[:SEARCH_CHUNKS]))
    prod = list(dict.fromkeys(ids[:PROD_UNITS]))
    pool = [Passage(engine.chunks[i].doc_id, engine.chunks[i].text) for i, _ in hits[:PACK_CHUNKS]]
    context = parent_articles(
        pool, engine.articles, k=None, budget_tokens=ARTICLE_BUDGET_TOKENS, overflow=PACK_OVERFLOW
    )
    rel = question.relevant
    values = {
        "mrr": mrr(listed, rel),
        "r5": recall_at_k(listed, rel, k=5),
        "g3_prod": _gold_in(prod, rel, 3),
        "g5_prod": _gold_in(prod, rel, 5),
        "pack": float(bool(set(rel) & {p.article_id for p in context})),
    }
    return values, is_wrong_act(listed, rel)


@dataclass
class Result:
    """Per-question metric values and wrong-act flags of one system on one set (or pooled)."""

    scores: dict[str, list[float]]
    wrong_act: list[bool]

    def mean(self, metric: str) -> float:
        values = self.scores[metric]
        return sum(values) / len(values) if values else 0.0


def score_set(
    engine: Engine, cell: Cell, questions: Sequence[Question], passages: Mapping[str, str]
) -> Result:
    """Scores of ``cell`` on ``questions`` (same order)."""
    engine.prepare([cell], [q.text for q in questions], passages)
    out = Result({m: [] for m in METRICS}, [])
    for q in questions:
        values, wrong = score_one(engine, cell, q, passages.get(q.text, ""))
        for m in METRICS:
            out.scores[m].append(values[m])
        out.wrong_act.append(wrong)
    return out


def pool(results: Sequence[Result]) -> Result:
    """Concatenation of per-set results."""
    return Result(
        {m: [v for r in results for v in r.scores[m]] for m in METRICS},
        [w for r in results for w in r.wrong_act],
    )


# ---------------------------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TuneCell:
    """One cell on the two tuning sets."""

    cell: Cell
    mrr_fresh: float
    mrr_synthetic: float
    pack_fresh: float
    pack_synthetic: float

    @property
    def objective(self) -> float:
        return (self.mrr_fresh + self.mrr_synthetic) / 2.0

    @property
    def pack(self) -> float:
        return (self.pack_fresh + self.pack_synthetic) / 2.0


def tie_key(tc: TuneCell) -> tuple[float, float, int, bool, int, int]:
    """Higher objective, higher pack recall, nearer B0, original CE query, smaller models."""
    return (
        -round(tc.objective / TIE),
        -round(tc.pack / TIE),
        tc.cell.distance(),
        tc.cell.ce_expanded,
        EMBEDDERS.index(tc.cell.embedder),
        RERANKERS.index(tc.cell.reranker),
    )


def choose_candidates(cells: Sequence[TuneCell]) -> dict[str, Cell]:
    """M1 and M2 are fixed; Q1 picks ``ce_query`` at B0's models; S is the grid argmax."""
    q1_options = [tc for tc in cells if tc.cell.expansion and tc.cell.distance() == 1]
    return {
        "M1": Cell(embedder="bge-base"),
        "M2": Cell(reranker="bge-reranker"),
        "Q1": min(q1_options, key=tie_key).cell,
        "S": min(cells, key=tie_key).cell,
    }


@dataclass
class Tuning:
    """Every grid cell on the tuning sets and the chosen candidates."""

    cells: list[TuneCell]
    chosen: dict[str, Cell]
    sizes: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "chosen": {k: c.to_json() for k, c in self.chosen.items()},
            "grid": [
                {
                    **tc.cell.to_json(),
                    "objective": tc.objective,
                    "mrr_fresh": tc.mrr_fresh,
                    "mrr_synthetic": tc.mrr_synthetic,
                    "pack_fresh": tc.pack_fresh,
                    "pack_synthetic": tc.pack_synthetic,
                }
                for tc in self.cells
            ],
            "sizes": self.sizes,
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> Tuning:
        cells = [
            TuneCell(
                Cell.from_json(g),
                float(g["mrr_fresh"]),
                float(g["mrr_synthetic"]),
                float(g["pack_fresh"]),
                float(g["pack_synthetic"]),
            )
            for g in data["grid"]
        ]
        chosen = {k: Cell.from_json(v) for k, v in data["chosen"].items()}
        return Tuning(cells, chosen, {k: int(v) for k, v in data["sizes"].items()})


def tune(
    eu: Engine,
    synthetic: Engine,
    fresh: Sequence[Question],
    synthetic_questions: Sequence[Question],
    passages: Mapping[str, str],
) -> Tuning:
    """Score the 12 cells on the tuning sets and choose the candidates."""
    cells = []
    for cell in grid():
        f = score_set(eu, cell, fresh, passages)
        s = score_set(synthetic, cell, synthetic_questions, passages)
        cells.append(TuneCell(cell, f.mean("mrr"), s.mean("mrr"), f.mean("pack"), s.mean("pack")))
    sizes = {"fresh": len(fresh), "synthetic": len(synthetic_questions)}
    return Tuning(cells, choose_candidates(cells), sizes)


def render_tuning(tuning: Tuning) -> str:
    """Grid table and the chosen settings."""
    lines = [
        "### ragkit vocab tune (docs/vocab_protocol.md): 12 cells on the tuning sets",
        "",
        f"Tuning sets: fresh {tuning.sizes['fresh']} (EU), synthetic {tuning.sizes['synthetic']}.",
        "Objective: mean of the two set-level MRR values.",
        "",
        "| cell | objective | MRR fresh | MRR synthetic | pack fresh | pack synthetic |",
        "|---|---|---|---|---|---|",
    ]
    for tc in sorted(tuning.cells, key=tie_key):
        lines.append(
            f"| `{tc.cell.label()}` | {tc.objective:.4f} | {tc.mrr_fresh:.3f} | "
            f"{tc.mrr_synthetic:.3f} | {tc.pack_fresh:.3f} | {tc.pack_synthetic:.3f} |"
        )
    lines += ["", "Chosen:"]
    for cid in CANDIDATE_IDS:
        cell = tuning.chosen[cid]
        note = " (= B0: not scored as a candidate)" if cell == BASELINE else ""
        lines.append(f"- {cid}: `{cell.label()}`{note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Held-out scoring and the decision rule
# ---------------------------------------------------------------------------------------------


@dataclass
class Row:
    """One system on every held-out set and on the synthetic guard."""

    name: str
    cell: Cell
    sets: dict[str, Result]
    synthetic: Result | None

    def pooled(self, exclude: Sequence[str] = ()) -> Result:
        return pool([r for n, r in self.sets.items() if n not in exclude])


@dataclass(frozen=True)
class Verdict:
    """Outcome of the decision rule for one candidate."""

    status: str
    d_mrr: tuple[float, float, float]
    wrong: int
    base_wrong: int
    d_g5: float
    d_pack: float
    d_synthetic: tuple[float, float, float]
    reasons: tuple[str, ...]

    @property
    def adopted(self) -> bool:
        return self.status == "adopted"


def decide(base: Row, cand: Row) -> Verdict:
    """The protocol's five conditions on the pooled held-out sets and the synthetic guard."""
    b, c = base.pooled(), cand.pooled()
    if base.synthetic is None or cand.synthetic is None:
        raise ValueError("the decision needs the synthetic guard of both rows")
    d_mrr = paired_bootstrap_ci(c.scores["mrr"], b.scores["mrr"])
    d_syn = paired_bootstrap_ci(cand.synthetic.scores["mrr"], base.synthetic.scores["mrr"])
    wrong, base_wrong = sum(c.wrong_act), sum(b.wrong_act)
    d_g5 = c.mean("g5_prod") - b.mean("g5_prod")
    d_pack = c.mean("pack") - b.mean("pack")
    checks = {
        "pooled MRR CI lower bound above 0": d_mrr[1] > 0,
        "wrong-act not above B0": wrong <= base_wrong,
        "G@5 (prod) difference at least 0": d_g5 >= -TIE,
        "pack-recall difference at least 0": d_pack >= -TIE,
        "synthetic guard": d_syn[0] >= -SYNTHETIC_MAX_DROP and d_syn[2] >= 0,
    }
    failed = tuple(name for name, ok in checks.items() if not ok)
    if not failed:
        status = "adopted"
    elif d_mrr[2] < 0 or not checks["synthetic guard"]:
        status = "rejected"
    else:
        status = "not adopted (no evidence)"
    return Verdict(status, d_mrr, wrong, base_wrong, d_g5, d_pack, d_syn, failed)


def winner(verdicts: Mapping[str, Verdict]) -> str | None:
    """Adopted candidate with the highest pooled MRR difference; ``None`` if none is adopted."""
    adopted = [k for k, v in verdicts.items() if v.adopted]
    return max(adopted, key=lambda k: verdicts[k].d_mrr[0]) if adopted else None


@dataclass
class HeldOut:
    """B0, the candidates (distinct from B0 and from each other) and the 12-cell sensitivity."""

    rows: dict[str, Row]
    grid_rows: list[Row]
    fallbacks: int
    n_questions: int


def run_heldout(
    eu: Engine,
    synthetic: Engine,
    sets: Mapping[str, Sequence[Question]],
    synthetic_questions: Sequence[Question],
    passages: Mapping[str, str],
    tuning: Tuning,
) -> HeldOut:
    """Score B0, every candidate and every grid cell once."""
    cache: dict[Cell, Row] = {}

    def row(name: str, cell: Cell) -> Row:
        if cell not in cache:
            results = {n: score_set(eu, cell, qs, passages) for n, qs in sets.items()}
            syn = score_set(synthetic, cell, synthetic_questions, passages)
            cache[cell] = Row(name, cell, results, syn)
        return Row(name, cell, cache[cell].sets, cache[cell].synthetic)

    rows = {"B0": row("B0", BASELINE)}
    for cid in CANDIDATE_IDS:
        cell = tuning.chosen[cid]
        if cell != BASELINE and all(r.cell != cell for r in rows.values()):
            rows[cid] = row(cid, cell)
    grid_rows = [row(c.label(), c) for c in grid()]
    texts = [q.text for qs in sets.values() for q in qs]
    fallbacks = sum(1 for t in texts if not passages.get(t))
    return HeldOut(rows, grid_rows, fallbacks, len(texts))


def _ci(values: Sequence[float]) -> str:
    m, lo, hi = bootstrap_ci(values)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def _delta(values: Sequence[float], base: Sequence[float]) -> str:
    m, lo, hi = paired_bootstrap_ci(values, base)
    better = sum(v > b + TIE for v, b in zip(values, base, strict=True))
    worse = sum(v < b - TIE for v, b in zip(values, base, strict=True))
    tied = len(values) - better - worse
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({better}/{worse}/{tied})"


def _tables(rows: Sequence[Row], base: Row, exclude: Sequence[str] = ()) -> list[str]:
    b = base.pooled(exclude)
    n = len(b.wrong_act)
    lines = [
        "| system | cell | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " | wrong-act |",
        "|---|---|" + "---|" * (len(METRICS) + 1),
    ]
    for r in rows:
        p = r.pooled(exclude)
        cells = " | ".join(_ci(p.scores[m]) for m in METRICS)
        lines.append(f"| {r.name} | `{r.cell.label()}` | {cells} | {sum(p.wrong_act)} / {n} |")
    lines += [
        "",
        "Paired differences against B0 (better / worse / tied):",
        "",
        "| system | d MRR | d G@5 (prod) | d pack recall |",
        "|---|---|---|---|",
    ]
    for r in rows:
        if r.name == base.name:
            continue
        p = r.pooled(exclude)
        lines.append(
            f"| {r.name} | {_delta(p.scores['mrr'], b.scores['mrr'])} | "
            f"{_delta(p.scores['g5_prod'], b.scores['g5_prod'])} | "
            f"{_delta(p.scores['pack'], b.scores['pack'])} |"
        )
    return lines


def render_heldout(run: HeldOut) -> str:
    """Per-set and pooled tables, the guard, the verdicts and the non-gating rows."""
    base = run.rows["B0"]
    candidates = list(run.rows.values())
    lines = [
        "### ragkit vocab score (docs/vocab_protocol.md): held-out, scored once",
        "",
        f"Expansion fallbacks (empty passage) on the held-out questions: {run.fallbacks} of "
        f"{run.n_questions}.",
    ]
    for name in base.sets:
        lines += ["", f"#### {name}", ""]
        only = [n for n in base.sets if n != name]
        lines += _tables(candidates, base, exclude=only)
    lines += ["", "#### pooled (gating)", ""]
    lines += _tables(candidates, base)
    lines += ["", "#### synthetic guard (paired MRR difference against B0)", ""]
    assert base.synthetic is not None
    lines.append(f"- B0 MRR {_ci(base.synthetic.scores['mrr'])}")
    for r in candidates[1:]:
        assert r.synthetic is not None
        lines.append(
            f"- {r.name}: {_delta(r.synthetic.scores['mrr'], base.synthetic.scores['mrr'])}"
        )
    verdicts = {r.name: decide(base, r) for r in candidates[1:]}
    lines += ["", "#### Decision rule", ""]
    for name, v in verdicts.items():
        why = "all five conditions hold" if v.adopted else "failed: " + "; ".join(v.reasons)
        lines.append(
            f"- **{name}** (`{run.rows[name].cell.label()}`): {v.status}. "
            f"d MRR {v.d_mrr[0]:+.3f} [{v.d_mrr[1]:+.3f}, {v.d_mrr[2]:+.3f}], wrong-act "
            f"{v.wrong} vs {v.base_wrong}, d G@5 (prod) {v.d_g5:+.3f}, d pack {v.d_pack:+.3f}, "
            f"synthetic d MRR {v.d_synthetic[0]:+.3f} [{v.d_synthetic[1]:+.3f}, "
            f"{v.d_synthetic[2]:+.3f}]; {why}."
        )
    best = winner(verdicts)
    lines += [
        "",
        f"**Verdict: {best} is adopted.**" if best else "**Verdict: nothing adopted; B0 stays.**",
    ]
    exclude = [n for n in base.sets if n.startswith("original")]
    lines += ["", "#### Non-gating: all 12 cells, pooled", ""]
    lines += _tables([base, *run.grid_rows], base)
    if exclude:
        lines += ["", f"#### Non-gating: pooled without {exclude[0]}", ""]
        lines += _tables(candidates, base, exclude=exclude)
    return "\n".join(lines)


def bge_base(device: str | None = None) -> SentenceEmbedder:
    """``BAAI/bge-base-en-v1.5`` with the bge retrieval query instruction."""
    return SentenceEmbedder(
        BGE_BASE_MODEL, BGE_BASE_REVISION, query_prefix=BGE_QUERY_INSTRUCTION, device=device
    )
