"""Tests for abstention signals: AUROC, bootstrap, risk-coverage, CV thresholds, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ragkit.abstention import (
    SIGNALS,
    accept_counts,
    auroc,
    auroc_ci,
    best_threshold,
    calibrate_threshold,
    collect_signals,
    compute_signals,
    cross_validated_accept,
    paired_auroc_ci,
    risk_at_coverage,
    stratified_folds,
)
from ragkit.cli import main
from ragkit.pipeline import RAGPipeline

# Answerable scores 0.9, 0.8, 0.4 vs unanswerable 0.7, 0.4, 0.2. Of the 9 (pos, neg) pairs:
# 0.9 wins 3, 0.8 wins 3, 0.4 beats 0.2 (1), ties 0.4 (0.5) and loses to 0.7 -> 7.5 / 9.
SCORES = [0.9, 0.8, 0.4, 0.7, 0.4, 0.2]
LABELS = [True, True, True, False, False, False]


def test_auroc_matches_hand_computed_case_with_a_tie() -> None:
    assert auroc(SCORES, LABELS) == pytest.approx(7.5 / 9.0)


def test_auroc_extremes_and_all_ties() -> None:
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.2, 0.1, 0.9, 0.8], [True, True, False, False]) == 0.0
    assert auroc([0.5] * 4, [True, True, False, False]) == 0.5


def test_auroc_is_invariant_to_order_and_monotone_transforms() -> None:
    order = [4, 0, 5, 2, 1, 3]
    shuffled = auroc([SCORES[i] for i in order], [LABELS[i] for i in order])
    assert shuffled == pytest.approx(auroc(SCORES, LABELS))
    assert auroc([10 * s + 3 for s in SCORES], LABELS) == pytest.approx(auroc(SCORES, LABELS))


def test_auroc_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="at least one"):
        auroc([0.1, 0.2], [True, True])


def test_auroc_ci_is_deterministic_and_brackets_the_estimate() -> None:
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(1.0, 1.0, 40), rng.normal(0.0, 1.0, 12)])
    labels = [True] * 40 + [False] * 12
    a, lo, hi = auroc_ci(scores, labels)
    assert lo < a < hi
    assert auroc_ci(scores, labels) == (a, lo, hi)
    assert auroc_ci([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == (1.0, 1.0, 1.0)


def test_paired_auroc_difference() -> None:
    labels = [True] * 20 + [False] * 8
    rng = np.random.default_rng(2)
    good = np.concatenate([rng.normal(2.0, 1.0, 20), rng.normal(0.0, 1.0, 8)])
    assert paired_auroc_ci(good, good, labels) == (0.0, 0.0, 0.0)
    noise = rng.normal(0.0, 1.0, 28)
    d, lo, hi = paired_auroc_ci(good, noise, labels)
    assert d == pytest.approx(auroc(good, labels) - auroc(noise, labels))
    assert lo <= d <= hi
    # Antisymmetric: swapping the signals negates the point estimate.
    assert paired_auroc_ci(noise, good, labels)[0] == pytest.approx(-d)


def test_risk_at_coverage_splits_ties_fractionally_and_ignores_order() -> None:
    scores = [0.9, 0.8, 0.5, 0.5, 0.1]
    bad = [False, False, True, False, True]
    # Top 3 = 0.9, 0.8 and one of the two tied 0.5s (one bad, one good): 0.5 / 3.
    assert risk_at_coverage(scores, bad, 0.6) == pytest.approx(0.5 / 3.0)
    swapped_bad = [False, False, False, True, True]
    assert risk_at_coverage(scores, swapped_bad, 0.6) == pytest.approx(0.5 / 3.0)
    assert risk_at_coverage(scores, bad, 1.0) == pytest.approx(0.4)
    assert risk_at_coverage(scores, bad, 0.2) == 0.0


def test_stratified_folds_are_deterministic_disjoint_and_stratified() -> None:
    labels = [True] * 42 + [False] * 12
    folds = stratified_folds(labels, k=5, seed=0)
    again = stratified_folds(labels, k=5, seed=0)
    assert all(
        np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        for a, b in zip(folds, again, strict=True)
    )
    tests = np.concatenate([t for _, t in folds])
    assert sorted(tests.tolist()) == list(range(54))  # every question held out exactly once
    y = np.asarray(labels)
    for train, test in folds:
        assert not set(train.tolist()) & set(test.tolist())
        assert int((~y[test]).sum()) in (2, 3)
        assert int(y[test].sum()) in (8, 9)
    assert any(
        not np.array_equal(a[1], b[1])
        for a, b in zip(folds, stratified_folds(labels, k=5, seed=1), strict=True)
    )


def test_stratified_folds_validate_input() -> None:
    with pytest.raises(ValueError, match="at least k"):
        stratified_folds([True] * 10 + [False] * 3, k=5)
    with pytest.raises(ValueError, match="at least 2"):
        stratified_folds([True, False], k=1)


def test_best_threshold_hand_case_and_tie_break() -> None:
    # t = 0.45 answers all 3 answerable and abstains on 2 of 3 unanswerable (balanced 5/6);
    # t = 0.75 gives the same 5/6 the other way round. The lowest threshold wins ties.
    pos, neg = [0.6, 0.8, 0.9], [0.1, 0.3, 0.7]
    assert best_threshold(pos + neg, [True] * 3 + [False] * 3) == pytest.approx(0.45)
    assert best_threshold([0.5, 0.6, 0.1, 0.2], [True, True, False, False]) == pytest.approx(0.35)


def test_cross_validated_threshold_never_sees_its_held_out_fold() -> None:
    rng = np.random.default_rng(3)
    scores = np.concatenate([rng.normal(1.0, 1.0, 20), rng.normal(0.0, 1.0, 10)])
    labels = np.array([True] * 20 + [False] * 10)
    base = cross_validated_accept(scores, labels, k=5, seed=0)
    assert len(base.thresholds) == 5
    _, test0 = stratified_folds(labels, k=5, seed=0)[0]
    changed = scores.copy()
    changed[test0] = 100.0
    after = cross_validated_accept(changed, labels, k=5, seed=0)
    assert after.thresholds[0] == base.thresholds[0]


def test_cross_validated_threshold_on_separable_scores() -> None:
    labels = np.array([True] * 10 + [False] * 10)
    scores = np.where(labels, 1.0, 0.0) + np.linspace(0.0, 0.1, 20)
    cv = cross_validated_accept(scores, labels, k=5, seed=0)
    assert np.array_equal(cv.accepted, labels)
    assert accept_counts(cv.accepted, labels) == (10, 10, 10, 10)


def test_calibrate_threshold_is_the_answerable_quantile() -> None:
    assert calibrate_threshold([float(i) for i in range(21)], 0.05) == pytest.approx(1.0)


_DOCS = [
    ("a", "Congestion is relieved by redispatching generation units near the constraint."),
    ("b", "Imbalance settlement charges parties for deviations from their schedules."),
    ("c", "Capacity allocation uses flow based coordination between transmission operators."),
]


def test_compute_signals_matches_pipeline_support() -> None:
    pipeline = RAGPipeline()
    pipeline.index_documents(_DOCS)
    for q in ("How is congestion relieved?", "Who won the football final?", "zebra quantum"):
        row = compute_signals(pipeline, q)
        assert set(row.values) == set(SIGNALS)
        assert row.values["query_support"] == pipeline.answer(q).support
    on_topic = compute_signals(pipeline, "How is congestion relieved by redispatching?")
    off_topic = compute_signals(pipeline, "zebra quantum football")
    assert on_topic.top_doc_ids[0] == "a"
    assert on_topic.values["bm25_top"] > off_topic.values["bm25_top"]
    assert off_topic.values["bm25_norm"] == 0.0


def test_collect_signals_shapes_and_retrieval_miss() -> None:
    table = collect_signals(
        _DOCS,
        [
            {"q": "How is congestion relieved?", "relevant_doc_ids": ["a"]},
            {"q": "What does imbalance settlement charge?", "relevant_doc_ids": ["zzz"]},
        ],
        ["Who won the football final?"],
    )
    assert table.answerable.tolist() == [True, True, False]
    assert table.retrieval_miss.tolist() == [False, True, False]
    assert all(v.shape == (3,) for v in table.scores.values())


def _dataset(tmp_path: Path) -> list[str]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    topics = [
        ("congestion", "redispatching of generation units relieves congestion near constraints"),
        ("imbalance", "imbalance settlement charges parties for deviations from schedules"),
        ("capacity", "capacity allocation uses flow based coordination between operators"),
        ("balancing", "balancing energy platforms exchange frequency restoration reserves"),
        ("tariffs", "network tariffs recover the cost of grid infrastructure from users"),
        ("storage", "storage facilities shift energy from surplus hours to scarce hours"),
    ]
    for name, text in topics:
        (corpus / f"{name}.md").write_text(f"# {name}\n\n{text}.", encoding="utf-8")
    qa = tmp_path / "qa.jsonl"
    qa.write_text(
        "\n".join(
            json.dumps({"q": f"What about {t.split()[0]} {name}?", "relevant_doc_ids": [name]})
            for name, t in topics
        ),
        encoding="utf-8",
    )
    unans = tmp_path / "unans.jsonl"
    unans.write_text(
        "\n".join(
            json.dumps({"q": q, "kind": "off_domain"})
            for q in (
                "Who won the football final?",
                "How do I bake sourdough bread?",
                "What is the capital of Australia?",
                "Which language introduced the borrow checker?",
                "How many moons does Jupiter have?",
                "What is the boiling point of ethanol?",
            )
        ),
        encoding="utf-8",
    )
    return ["--corpus-dir", str(corpus), "--qa-file", str(qa), "--unanswerable-file", str(unans)]


def test_cli_abstention_signals_prints_all_tables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as ex:
        main(["abstention", "--signals", "all", *_dataset(tmp_path)])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "6 answerable vs 6 unanswerable" in out
    assert "| AUROC - baseline |" in out
    for name in SIGNALS:
        assert f"| `{name}` |" in out
    assert "unanswerable share of answered" in out
    assert "5-fold stratified CV" in out
    assert "Zero-shot transfer: threshold" in out


def test_cli_abstention_rejects_unknown_signal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as ex:
        main(["abstention", "--signals", "bm25_top,nope", *_dataset(tmp_path)])
    assert ex.value.code == 2
    assert "unknown signals ['nope']" in capsys.readouterr().err


def test_cli_eval_prints_paired_rerank_vs_hybrid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as ex:
        main(["eval", "--strategies", "hybrid,rerank", *_dataset(tmp_path)])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "Hybrid + Rerank minus Hybrid (RRF), rewrite None" in out
    assert "Hybrid + Rerank minus Hybrid (RRF), rewrite HyDE" in out
