"""Budget packing of parent articles: retrieval-only coverage study (Phase N-1).

Protocol, grid, adoption rule and predictions: ``docs/packing_protocol.md`` (committed before any
number). No generation and no LLM call: for each question the context that ``ragkit answer``
would build is constructed, and we count whether it holds a gold article and how many estimated
tokens it has. ``k3`` is the shipped context, ``k5`` the K-R4 setting, ``pack`` budget packing.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ragkit.eval import bootstrap_ci, paired_bootstrap_ci
from ragkit.lexical import Question
from ragkit.parent import (
    ARTICLE_BUDGET_TOKENS,
    ARTICLE_K,
    OVERFLOW_POLICIES,
    Passage,
    estimate_tokens,
    parent_articles,
)

POOL_DEPTHS = (5, 10, 20)
K5_BUDGET_TOKENS = 20_000  # the post-hoc k = 5 setting of the Phase L addendum
TOP_CHUNKS = 5  # chunks behind the shipped k = 3 and the k = 5 contexts
RETRIEVERS = ("default", "hybrid+ce")
ARMS = ("k3", "k5", "pack")
TIE = 1e-9

Retrieve = Callable[[str, int], list[Passage]]
Pools = Mapping[int, Sequence[Sequence[Passage]]]  # pool depth -> ranked chunks per question


@dataclass(frozen=True)
class Arm:
    """One context builder: chunks read, article limit (``None``: budget only), budget, policy."""

    name: str
    chunks: int
    k: int | None
    budget_tokens: int
    overflow: str

    def build(self, ranked: Sequence[Passage], articles: Mapping[str, str]) -> list[Passage]:
        """The context of one question from its ranked chunks (at least ``chunks`` of them)."""
        return parent_articles(
            ranked[: self.chunks],
            articles,
            k=self.k,
            budget_tokens=self.budget_tokens,
            overflow=self.overflow,
        )


def make_arms(overflow: str, depth: int) -> dict[str, Arm]:
    """The three arms of the protocol with the dev-chosen packing setting."""
    if overflow not in OVERFLOW_POLICIES:
        raise ValueError(f"overflow must be one of {OVERFLOW_POLICIES}, got {overflow!r}")
    return {
        "k3": Arm("k3", TOP_CHUNKS, ARTICLE_K, ARTICLE_BUDGET_TOKENS, "stop"),
        "k5": Arm("k5", TOP_CHUNKS, 5, K5_BUDGET_TOKENS, "stop"),
        "pack": Arm("pack", depth, None, ARTICLE_BUDGET_TOKENS, overflow),
    }


def retrieve_pools(
    retrieve: Retrieve, questions: Sequence[Question], depths: Sequence[int] = POOL_DEPTHS
) -> dict[int, list[list[Passage]]]:
    """Ranked chunks of every question at each pool depth (one ``retrieve(q, depth)`` call each)."""
    return {d: [retrieve(q.text, d) for q in questions] for d in depths}


@dataclass(frozen=True)
class ArmScores:
    """Per-question outcome of one arm: gold in context, estimated tokens, article count."""

    hit: list[float]
    tokens: list[int]
    articles: list[int]
    over_budget: int

    @property
    def recall(self) -> float:
        return sum(self.hit) / len(self.hit)

    @property
    def mean_tokens(self) -> float:
        return sum(self.tokens) / len(self.tokens)

    def extend(self, other: ArmScores) -> ArmScores:
        """The scores of this arm's questions followed by ``other``'s (pooling sets)."""
        return ArmScores(
            self.hit + other.hit,
            self.tokens + other.tokens,
            self.articles + other.articles,
            self.over_budget + other.over_budget,
        )


def score_arm(
    arm: Arm,
    ranked: Sequence[Sequence[Passage]],
    questions: Sequence[Question],
    articles: Mapping[str, str],
) -> ArmScores:
    """Build the arm's context for every question and score it (retrieval only)."""
    if len(ranked) != len(questions):
        raise ValueError("one ranked list per question is required")
    hit: list[float] = []
    tokens: list[int] = []
    counts: list[int] = []
    over = 0
    for chunks, q in zip(ranked, questions, strict=True):
        context = arm.build(chunks, articles)
        used = sum(estimate_tokens(p.text) for p in context)
        hit.append(float(bool(set(q.relevant) & {p.article_id for p in context})))
        tokens.append(used)
        counts.append(len(context))
        over += used > arm.budget_tokens
    return ArmScores(hit, tokens, counts, over)


def pool_scores(parts: Sequence[ArmScores]) -> ArmScores:
    """Concatenate per-set scores in order."""
    out = parts[0]
    for p in parts[1:]:
        out = out.extend(p)
    return out


# ---------------------------------------------------------------------------------------------
# Dev selection
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DevCell:
    """One retriever on one dev set: its pools, questions and corpus."""

    retriever: str
    dataset: str
    pools: Pools
    questions: Sequence[Question]
    articles: Mapping[str, str]


@dataclass(frozen=True)
class Grid:
    """One ``(overflow, depth)`` setting with its dev recall and tokens per ``DevCell``."""

    overflow: str
    depth: int
    recall: dict[str, float]  # "<retriever>/<dataset>" -> recall
    mean_tokens: dict[str, float]

    @property
    def objective(self) -> float:
        return sum(self.recall.values()) / len(self.recall)

    @property
    def tokens(self) -> float:
        return sum(self.mean_tokens.values()) / len(self.mean_tokens)


def best_grid(grids: Sequence[Grid]) -> Grid:
    """The protocol's argmax: objective, then ties (below 1e-9) by tokens, policy, depth."""
    top = max(g.objective for g in grids)
    tied = [g for g in grids if top - g.objective < TIE]
    return min(tied, key=lambda g: (g.tokens, OVERFLOW_POLICIES.index(g.overflow), g.depth))


def tune(cells: Sequence[DevCell]) -> list[Grid]:
    """Every ``(overflow, depth)`` of the grid scored on the dev cells."""
    grids: list[Grid] = []
    for overflow in OVERFLOW_POLICIES:
        for depth in POOL_DEPTHS:
            pack = make_arms(overflow, depth)["pack"]
            recall: dict[str, float] = {}
            tokens: dict[str, float] = {}
            for c in cells:
                s = score_arm(pack, c.pools[depth], c.questions, c.articles)
                recall[f"{c.retriever}/{c.dataset}"] = s.recall
                tokens[f"{c.retriever}/{c.dataset}"] = s.mean_tokens
            grids.append(Grid(overflow, depth, recall, tokens))
    return grids


def tuning_json(grids: Sequence[Grid], dev_sizes: Mapping[str, int]) -> dict[str, Any]:
    """The frozen choice and the whole grid, committed as ``packing_tuning.json``."""
    chosen = best_grid(grids)
    return {
        "chosen": {"overflow": chosen.overflow, "depth": chosen.depth},
        "dev_sizes": dict(dev_sizes),
        "grid": [
            {
                "overflow": g.overflow,
                "depth": g.depth,
                "objective": g.objective,
                "mean_tokens": g.tokens,
                "recall": g.recall,
                "cell_mean_tokens": g.mean_tokens,
            }
            for g in grids
        ],
    }


def render_tuning(grids: Sequence[Grid], dev_sizes: Mapping[str, int]) -> str:
    """Markdown table of the dev grid."""
    chosen = best_grid(grids)
    keys = list(grids[0].recall)
    lines = [
        "### ragkit packing tune: dev grid (overflow policy x pool depth)",
        "",
        "Dev questions: " + ", ".join(f"{k} {v}" for k, v in dev_sizes.items()) + ". "
        "Objective = mean of the recall cells; ties go to fewer tokens, `stop`, smaller depth.",
        "",
        "| overflow | depth | objective | mean tokens | " + " | ".join(keys) + " |",
        "|---|---|---|---|" + "---|" * len(keys),
    ]
    for g in grids:
        cells = " | ".join(f"{g.recall[k]:.3f} ({g.mean_tokens[k]:.0f} tok)" for k in keys)
        lines.append(f"| {g.overflow} | {g.depth} | {g.objective:.4f} | {g.tokens:.0f} | {cells} |")
    lines += ["", f"Chosen on dev: overflow `{chosen.overflow}`, pool depth {chosen.depth}."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Held-out scoring
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldOutSet:
    """One held-out question set for one retriever."""

    name: str
    pools: Pools
    questions: Sequence[Question]


@dataclass(frozen=True)
class HeldOut:
    """Per retriever: per-set and pooled scores of every arm."""

    overflow: str
    depth: int
    by_set: dict[str, dict[str, dict[str, ArmScores]]]  # retriever -> set -> arm -> scores
    pooled: dict[str, dict[str, ArmScores]]  # retriever -> arm -> scores


def run_heldout(
    sets: Mapping[str, Sequence[HeldOutSet]],
    articles: Mapping[str, str],
    overflow: str,
    depth: int,
) -> HeldOut:
    """Score the three arms once on every held-out set of every retriever."""
    arms = make_arms(overflow, depth)
    by_set: dict[str, dict[str, dict[str, ArmScores]]] = {}
    pooled: dict[str, dict[str, ArmScores]] = {}
    for retriever, parts in sets.items():
        by_set[retriever] = {}
        for part in parts:
            by_set[retriever][part.name] = {
                name: score_arm(arm, part.pools[arm.chunks], part.questions, articles)
                for name, arm in arms.items()
            }
        pooled[retriever] = {
            name: pool_scores([by_set[retriever][p.name][name] for p in parts]) for name in arms
        }
    return HeldOut(overflow, depth, by_set, pooled)


@dataclass(frozen=True)
class Diff:
    """Paired recall difference ``a - b`` with its 95 % CI and gained / lost / tied counts."""

    mean: float
    lo: float
    hi: float
    gained: int
    lost: int
    tied: int


def paired_diff(a: Sequence[float], b: Sequence[float]) -> Diff:
    """Paired bootstrap (2000 resamples, seed 0) of the 0/1 recall difference."""
    mean, lo, hi = paired_bootstrap_ci(a, b)
    gained = sum(x > y for x, y in zip(a, b, strict=True))
    lost = sum(x < y for x, y in zip(a, b, strict=True))
    return Diff(mean, lo, hi, gained, lost, len(a) - gained - lost)


def passes_rule(run: HeldOut) -> bool:
    """Adoption rule: ``pack - k3`` CI lower bound above 0 on every retriever, none over budget."""
    for arms in run.pooled.values():
        if paired_diff(arms["pack"].hit, arms["k3"].hit).lo <= 0:
            return False
        if any(s.over_budget for s in arms.values()):
            return False
    return True


def _fmt_diff(d: Diff) -> str:
    return f"{d.mean:+.3f} [{d.lo:+.3f}, {d.hi:+.3f}]"


def _row(name: str, s: ArmScores) -> str:
    n = len(s.hit)
    ci = bootstrap_ci(s.hit)
    return (
        f"| {name} | {int(sum(s.hit))}/{n} | {ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}] "
        f"| {s.mean_tokens:.0f} | {statistics.median(s.tokens):.0f} | {max(s.tokens)} "
        f"| {s.over_budget} | {statistics.fmean(s.articles):.2f} "
        f"| {statistics.median(s.articles):g} |"
    )


def render_heldout(run: HeldOut) -> str:
    """Markdown report of the held-out run and the decision against the pre-registered rule."""
    n = len(next(iter(run.pooled.values()))["k3"].hit)
    lines = [
        "### ragkit packing score: gold article in context, held-out (retrieval only, no LLM call)",
        "",
        f"Pooled held-out questions: {n}. Packing setting chosen on dev: overflow "
        f"`{run.overflow}`, pool depth {run.depth} chunks, budget {ARTICLE_BUDGET_TOKENS} "
        "estimated tokens. Tokens are `estimate_tokens` (chars / 4) of the context.",
    ]
    for retriever, arms in run.pooled.items():
        lines += [
            "",
            f"**{retriever}**",
            "",
            "| arm | gold in context | recall [95 % CI] | mean tokens | median | max "
            "| over budget | mean articles | median articles |",
            "|---|---|---|---|---|---|---|---|---|",
            *(_row(name, s) for name, s in arms.items()),
            "",
            "| paired recall difference (pooled) | d [95 % CI] | gained / lost / tied |",
            "|---|---|---|",
        ]
        for a, b, tag in (("k5", "k3", ""), ("pack", "k3", ""), ("pack", "k5", " (secondary)")):
            d = paired_diff(arms[a].hit, arms[b].hit)
            lines.append(f"| {a} - {b}{tag} | {_fmt_diff(d)} | {d.gained} / {d.lost} / {d.tied} |")
        lines += [
            "",
            "Per set (descriptive): gold in context, k3 / k5 / pack.",
            "",
            "| set | questions | k3 | k5 | pack |",
            "|---|---|---|---|---|",
        ]
        for name, per_arm in run.by_set[retriever].items():
            cells = " | ".join(f"{int(sum(per_arm[a].hit))}" for a in ARMS)
            lines.append(f"| {name} | {len(per_arm['k3'].hit)} | {cells} |")
    verdicts = {r: paired_diff(a["pack"].hit, a["k3"].hit) for r, a in run.pooled.items()}
    over = sum(s.over_budget for arms in run.pooled.values() for s in arms.values())
    lines += [
        "",
        "**Decision rule** (`docs/packing_protocol.md`): packing becomes the `ragkit answer` "
        "default only if the paired CI lower bound of `pack - k3` is above 0 on BOTH "
        "retrievers and no context exceeds its budget.",
        "",
    ]
    for r, d in verdicts.items():
        word = (
            "above 0"
            if d.lo > 0
            else "not above 0 (no detectable effect)"
            if d.hi >= 0
            else "below 0"
        )
        lines.append(f"- {r}: lower bound {d.lo:+.3f}, {word}.")
    lines.append(f"- contexts over budget: {over}.")
    verdict = (
        "rule PASSES, packing becomes the default"
        if passes_rule(run)
        else "rule FAILS, packing stays opt-in (--pack)"
    )
    lines += [
        "",
        f"Result: {verdict}. Answer quality was not re-measured (needs the API).",
    ]
    return "\n".join(lines)
