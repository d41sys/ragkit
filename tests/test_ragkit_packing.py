"""Budget packing (docs/packing_protocol.md): policy, budget invariant, arms, rule, dev choice."""

from __future__ import annotations

import pytest

from ragkit.lexical import Question
from ragkit.packbench import (
    DevCell,
    HeldOut,
    HeldOutSet,
    best_grid,
    make_arms,
    paired_diff,
    passes_rule,
    run_heldout,
    score_arm,
    tune,
    tuning_json,
)
from ragkit.parent import (
    CHARS_PER_TOKEN,
    Passage,
    estimate_tokens,
    pack_articles,
    parent_articles,
)

# token costs: a1 100, a2 200, a3 1000, a4 10, a5 150
ARTICLES = {
    "a1": "A" * 400,
    "a2": "B" * 800,
    "a3": "C" * 4000,
    "a4": "D" * 40,
    "a5": "E" * 600,
}


def ranked(*ids: str) -> list[Passage]:
    return [Passage(i, "chunk") for i in ids]


def used(context: list[Passage]) -> int:
    return sum(estimate_tokens(p.text) for p in context)


def test_pack_fills_the_budget_without_an_article_limit() -> None:
    got = pack_articles(ranked("a1", "a2", "a1", "a4", "a5"), ARTICLES, budget_tokens=10_000)
    assert [p.article_id for p in got] == ["a1", "a2", "a4", "a5"]  # 4 > the shipped k = 3


def test_overflow_stop_drops_the_rest_skip_keeps_later_articles_that_fit() -> None:
    chunks = ranked("a1", "a3", "a4", "a5")  # a3 (1000) does not fit in 500
    stop = pack_articles(chunks, ARTICLES, budget_tokens=500, overflow="stop")
    skip = pack_articles(chunks, ARTICLES, budget_tokens=500, overflow="skip")
    assert [p.article_id for p in stop] == ["a1"]
    assert [p.article_id for p in skip] == ["a1", "a4", "a5"]
    assert used(skip) == 260 <= 500


def test_pack_never_exceeds_the_budget_even_when_the_first_article_is_too_big() -> None:
    for overflow in ("stop", "skip"):
        for budget in (1, 50, 99, 100, 101, 310, 1500):
            got = pack_articles(
                ranked("a3", "a1", "a2", "a4", "a5"),
                ARTICLES,
                budget_tokens=budget,
                overflow=overflow,
            )
            assert got and used(got) <= budget
    first = pack_articles(ranked("a3", "a1"), ARTICLES, budget_tokens=50, overflow="skip")
    assert [p.article_id for p in first] == ["a3"]  # cut, packing ends: a1 would have fitted
    assert len(first[0].text) == 50 * CHARS_PER_TOKEN


def test_default_mode_is_unchanged_and_policy_is_validated() -> None:
    chunks = ranked("a1", "a3", "a4")
    assert [p.article_id for p in parent_articles(chunks, ARTICLES, k=3, budget_tokens=500)] == [
        "a1"
    ]
    with pytest.raises(ValueError):
        parent_articles(chunks, ARTICLES, overflow="truncate")
    with pytest.raises(ValueError):
        pack_articles(chunks, ARTICLES, budget_tokens=0)


def test_arms_read_their_own_chunks_and_budgets() -> None:
    arms = make_arms("skip", 10)
    pool = ranked("a1", "a2", "a4", "a5", "a1", "a3")  # six chunks, five distinct articles
    q = Question("q", ("a3",), "")
    k3 = score_arm(arms["k3"], [pool], [q], ARTICLES)
    k5 = score_arm(arms["k5"], [pool], [q], ARTICLES)
    pack = score_arm(arms["pack"], [pool], [q], ARTICLES)
    assert k3.articles == [3] and k3.hit == [0.0]  # a3 is the sixth chunk: outside the top 5
    assert k5.articles == [4] and k5.hit == [0.0]  # k = 5 reads only the top-5 chunks too
    assert pack.articles == [5] and pack.hit == [1.0]  # packing reads ten chunks
    assert pack.over_budget == 0 and pack.tokens == [100 + 200 + 10 + 150 + 1000]


def _heldout(pack_hits: list[float], k3_hits: list[float]) -> HeldOut:
    def scores(hit: list[float], over: int = 0):
        from ragkit.packbench import ArmScores

        return ArmScores(hit, [10] * len(hit), [1] * len(hit), over)

    arms = {"k3": scores(k3_hits), "k5": scores(k3_hits), "pack": scores(pack_hits)}
    return HeldOut("stop", 5, {}, {"default": arms, "hybrid+ce": arms})


def test_rule_needs_a_ci_lower_bound_above_zero_on_every_retriever() -> None:
    n = 40
    assert passes_rule(_heldout([1.0] * n, [1.0] * 20 + [0.0] * 20))  # 20 gains, no loss
    assert not passes_rule(_heldout([1.0] * n, [1.0] * n))  # tied: CI is [0, 0]
    # one gain in 40 pairs has a lower bound of 0: "no detectable effect" is not a pass
    assert not passes_rule(_heldout([1.0] * n, [1.0] * (n - 1) + [0.0]))
    d = paired_diff([1.0] * n, [1.0] * 20 + [0.0] * 20)
    assert (d.gained, d.lost, d.tied) == (20, 0, 20) and d.lo > 0


def test_a_retriever_that_fails_or_a_budget_violation_blocks_adoption() -> None:
    from ragkit.packbench import ArmScores

    good = _heldout([1.0] * 40, [1.0] * 20 + [0.0] * 20)
    flat = ArmScores([1.0] * 40, [10] * 40, [1] * 40, 0)
    mixed = HeldOut(
        "stop",
        5,
        {},
        {"default": good.pooled["default"], "hybrid+ce": {"k3": flat, "k5": flat, "pack": flat}},
    )
    assert not passes_rule(mixed)
    over = ArmScores([1.0] * 40, [10] * 40, [1] * 40, 1)
    bad = HeldOut(
        "stop",
        5,
        {},
        {
            "default": {**good.pooled["default"], "pack": over},
            "hybrid+ce": good.pooled["hybrid+ce"],
        },
    )
    assert not passes_rule(bad)


def test_dev_choice_prefers_recall_then_fewer_tokens_then_stop_then_smaller_depth() -> None:
    q = [Question("q", ("a1",), "")]
    # gold a1 is the 12th chunk: reached only at depth 20; a3 (1000 tokens) comes first
    deep = ranked("a3", "a2", "a2", "a2", "a2", "a2", "a2", "a2", "a2", "a2", "a2", "a1")
    pools = {d: [deep[:d]] for d in (5, 10, 20)}
    grids = tune([DevCell("default", "dev", pools, q, ARTICLES)])
    assert len(grids) == 6
    chosen = best_grid(grids)
    assert (chosen.overflow, chosen.depth) == ("stop", 20)  # equal recall: stop beats skip
    out = tuning_json(grids, {"dev": 1})
    assert out["chosen"] == {"overflow": "stop", "depth": 20}
    # a1 is already in the top 5: every setting ties on recall, fewest tokens wins, then depth 5
    pools2 = {d: [ranked("a1", "a2", "a4", "a5", "a5", "a3")[:d]] for d in (5, 10, 20)}
    grids2 = tune([DevCell("default", "dev", pools2, q, ARTICLES)])
    assert (best_grid(grids2).overflow, best_grid(grids2).depth) == ("stop", 5)


def test_run_heldout_pools_sets_in_order() -> None:
    q1, q2 = [Question("x", ("a1",), "")], [Question("y", ("a2",), ""), Question("z", ("a9",), "")]
    p1 = {d: [ranked("a1")] for d in (5, 10, 20)}
    p2 = {d: [ranked("a1", "a2"), ranked("a4")] for d in (5, 10, 20)}
    run = run_heldout(
        {"default": [HeldOutSet("one", p1, q1), HeldOutSet("two", p2, q2)]},
        {**ARTICLES, "a9": "Z" * 40},
        "skip",
        10,
    )
    assert run.pooled["default"]["k3"].hit == [1.0, 1.0, 0.0]
    assert set(run.by_set["default"]) == {"one", "two"}
