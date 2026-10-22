"""Tests for abstention signals: AUROC, bootstrap, risk-coverage, CV thresholds, CLI."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ragkit.abstention import (
    SIGNALS,
    LogisticModel,
    accept_counts,
    auroc,
    auroc_ci,
    best_threshold,
    calibrate_threshold,
    collect_signals,
    compute_signals,
    cross_validated_accept,
    fit_logistic,
    nested_cv_logistic,
    paired_auroc_ci,
    risk_at_coverage,
    stratified_folds,
)
from ragkit.cli import main
from ragkit.modelsignals import MODEL_SIGNALS, CachedScorer, ModelScorers, ScoreCache
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


def _objective(
    x: np.ndarray, y: np.ndarray, model: LogisticModel, l2: float, w: np.ndarray, b: float
) -> float:
    z = ((x - model.mean) / model.scale) @ w + b
    t = y.astype(float)
    return float(np.sum(np.logaddexp(0.0, z) - t * z) + 0.5 * l2 * np.sum(w * w))


def test_fit_logistic_converges_on_a_separable_toy_case() -> None:
    x = np.array([[0.0, 5.0], [0.5, 4.0], [1.0, 6.0], [3.0, 5.5], [3.5, 4.5], [4.0, 5.0]])
    y = np.array([False, False, False, True, True, True])
    model = fit_logistic(x, y, l2=0.1)
    assert model.converged
    assert model.n_iter < 30
    scores = model.decision(x)
    assert np.array_equal(scores > 0, y)
    assert auroc(scores, y) == 1.0
    assert model.weights[0] > 0  # the informative feature points towards "answerable"
    # Strictly convex objective: no small perturbation of the solution may lower it.
    best = _objective(x, y, model, 0.1, model.weights, model.bias)
    rng = np.random.default_rng(0)
    for _ in range(50):
        dw, db = rng.normal(0.0, 0.05, size=2), float(rng.normal(0.0, 0.05))
        assert _objective(x, y, model, 0.1, model.weights + dw, model.bias + db) >= best
    # A stronger ridge shrinks the weights even though the classes are separable.
    assert np.linalg.norm(fit_logistic(x, y, l2=10.0).weights) < np.linalg.norm(model.weights)


def test_fit_logistic_matches_a_symmetric_hand_case() -> None:
    # Symmetric 1-D data: the optimum has zero intercept and a positive slope.
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([False, True, False, True])
    model = fit_logistic(x, y, l2=1.0)
    assert model.converged
    assert model.bias == pytest.approx(0.0, abs=1e-9)
    assert model.weights[0] > 0.0
    assert model.decision([[0.0]])[0] == pytest.approx(0.0, abs=1e-9)


def test_fit_logistic_validates_input() -> None:
    x = np.zeros((4, 2))
    with pytest.raises(ValueError, match="at least one"):
        fit_logistic(x, [True] * 4)
    with pytest.raises(ValueError, match="positive"):
        fit_logistic(x, [True, False, True, False], l2=0.0)
    with pytest.raises(ValueError, match="one row per label"):
        fit_logistic(x, [True, False])
    # A constant feature gets scale 1 instead of dividing by zero.
    model = fit_logistic(np.zeros((4, 1)), [True, False, True, False])
    assert np.all(np.isfinite(model.decision(x[:, :1])))


def _noisy_problem(
    seed: int, n_pos: int, n_neg: int, shift: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.array([True] * n_pos + [False] * n_neg)
    x = rng.normal(size=(n_pos + n_neg, 9))
    x[:n_pos, 0] += shift
    return x, labels


def test_nested_cv_never_fits_on_the_questions_it_scores() -> None:
    x, labels = _noisy_problem(4, 30, 20, shift=1.0)
    res = nested_cv_logistic(x, labels, k=5, seed=0)
    outer = [f for f in res.fits if f.stage == "outer"]
    assert len(outer) == 5
    # Every question is scored by exactly one outer fold.
    assert sorted(np.concatenate([f.scored for f in outer]).tolist()) == list(range(50))
    pending = []
    for record in res.fits:
        assert not set(record.fit.tolist()) & set(record.scored.tolist())
        if record.stage == "inner":
            pending.append(record)
            continue
        # The inner fits of this fold only touch the outer training questions.
        assert pending
        for inner in pending:
            assert set(inner.fit.tolist()) | set(inner.scored.tolist()) <= set(record.fit.tolist())
            assert not set(inner.fit.tolist()) & set(record.scored.tolist())
            assert not set(inner.scored.tolist()) & set(record.scored.tolist())
        pending = []
    assert not pending


def test_nested_cv_held_out_scores_come_from_models_fitted_without_them() -> None:
    x, labels = _noisy_problem(5, 30, 20, shift=1.0)
    res = nested_cv_logistic(x, labels, k=5, seed=0)
    train, test = stratified_folds(labels, k=5, seed=0)[0]
    refit = fit_logistic(x[train], labels[train], l2=res.l2[0])
    assert np.allclose(res.scores[test], refit.decision(x[test]))
    # Rewriting the held-out fold's features cannot change what was chosen inside its fold.
    changed = x.copy()
    changed[test] = 100.0
    after = nested_cv_logistic(changed, labels, k=5, seed=0)
    assert after.l2[0] == res.l2[0]
    assert after.thresholds[0] == res.thresholds[0]
    assert not np.allclose(after.scores[test], res.scores[test])
    assert nested_cv_logistic(x, labels, k=5, seed=0).thresholds == res.thresholds


def test_nested_cv_gives_chance_auroc_on_pure_noise_and_finds_real_signal() -> None:
    noise, labels = _noisy_problem(6, 42, 12, shift=0.0)
    assert auroc(nested_cv_logistic(noise, labels, k=5, seed=0).scores, labels) < 0.7
    signal, labels = _noisy_problem(6, 42, 12, shift=3.0)
    res = nested_cv_logistic(signal, labels, k=5, seed=0)
    assert auroc(res.scores, labels) > 0.9
    answered, _, abstained, _ = accept_counts(res.accepted, labels)
    assert answered >= 36 and abstained >= 9


def test_nested_cv_needs_two_questions_per_class_in_every_training_fold() -> None:
    x, labels = _noisy_problem(7, 4, 2, shift=1.0)
    with pytest.raises(ValueError, match="at least 2"):
        nested_cv_logistic(x, labels, k=2, seed=0)


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
    assert "| `combined` |" in out
    assert "nested CV by question" in out
    assert "unanswerable share of answered" in out
    assert "5-fold stratified CV" in out
    assert "Zero-shot transfer: threshold" in out


def test_cli_model_signals_with_fake_scorers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class Fake:
        scorer_id = "fake@0"

        def score(self, question: str, passages: Sequence[str]) -> list[float]:
            q = set(question.lower().split())
            return [float(len(q & set(p.lower().split()))) for p in passages]

    seen: dict[str, Any] = {}

    def fake_loader(cache_path: Path | None, device: str) -> ModelScorers:
        seen.update(cache_path=cache_path, device=device)
        cache = ScoreCache(cache_path)
        return ModelScorers(CachedScorer(Fake(), cache), CachedScorer(Fake(), cache), cache)

    monkeypatch.setattr("ragkit.cli.load_model_scorers", fake_loader)
    cache_path = tmp_path / "cache" / "scores.json"
    argv = ["abstention", "--signals", "all", "--model-signals", *_dataset(tmp_path)]
    argv += ["--model-cache", str(cache_path)]
    with pytest.raises(SystemExit) as ex:
        main(argv)
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert seen == {"cache_path": cache_path, "device": "cpu"}
    assert cache_path.exists()
    for name in [*SIGNALS, *MODEL_SIGNALS, "combined", "combined_model", "combined_all"]:
        assert f"| `{name}` |" in out
    assert "Model scoring (device cpu): ce: " in out
    assert "11 signals" in out
    assert "Zero-shot transfer: threshold" in out
    assert "| `ce_top` | " in out.split("Zero-shot transfer")[1]


def test_cli_without_model_flag_never_loads_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any, **__: Any) -> ModelScorers:
        raise AssertionError("models must not load without --model-signals")

    monkeypatch.setattr("ragkit.cli.load_model_scorers", boom)
    with pytest.raises(SystemExit) as ex:
        main(["abstention", "--signals", "all", *_dataset(tmp_path)])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "ce_top" not in out
    assert "combined_all" not in out


def test_cli_cohort_filters_unanswerable_questions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _dataset(tmp_path)
    unans = tmp_path / "unans.jsonl"
    records = [json.loads(line) for line in unans.read_text(encoding="utf-8").splitlines()]
    for i, record in enumerate(records):
        record["cohort"] = "original-12" if i < 2 else "expanded"
        record["miss_type"] = "national_rule"
    unans.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    with pytest.raises(SystemExit) as ex:
        main(["abstention", "--cohort", "expanded", *args])
    assert ex.value.code == 0
    out = capsys.readouterr().out
    assert "| unanswerable off_domain (abstained) |" in out
    assert "/4 |" in out  # 6 records, 2 tagged original-12
    assert "| unanswerable national_rule (abstained) |" in out
    with pytest.raises(SystemExit) as ex:
        main(["abstention", "--cohort", "original-12", *args])
    assert ex.value.code == 0
    assert "/2 |" in capsys.readouterr().out


def test_cli_cohort_without_tagged_records_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no unanswerable records with cohort 'expanded'"):
        main(["abstention", "--cohort", "expanded", *_dataset(tmp_path)])


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
