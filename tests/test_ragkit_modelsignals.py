from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from ragkit.abstention import SIGNALS, collect_signals
from ragkit.modelsignals import (
    MODEL_SIGNALS,
    CachedScorer,
    ModelScorers,
    ScoreCache,
    span_minus_null,
)


class FakeScorer:
    """Deterministic stand-in for a model: score = number of passage words found in the question."""

    def __init__(self, scorer_id: str = "fake@0", scale: float = 1.0) -> None:
        self.scorer_id = scorer_id
        self.scale = scale
        self.seen: list[tuple[str, int]] = []

    def score(self, question: str, passages: Sequence[str]) -> list[float]:
        self.seen.append((question, len(passages)))
        q = set(question.lower().split())
        return [self.scale * len(q & set(p.lower().split())) for p in passages]


_DOCS = [
    ("a", "# A\n\nCongestion is relieved by redispatching of generation units."),
    ("b", "# B\n\nImbalance settlement charges parties for deviations from schedules."),
]


def test_span_minus_null_prefers_best_context_span() -> None:
    start = np.array([2.0, 0.0, 5.0, 1.0, 0.0])
    end = np.array([1.0, 0.0, 1.0, 4.0, 0.0])
    context = np.array([False, True, True, True, False])
    # best span: start 2 (5.0) .. end 3 (4.0) = 9.0; null = 2.0 + 1.0
    assert span_minus_null(start, end, context) == pytest.approx(6.0)


def test_span_minus_null_respects_max_answer_length() -> None:
    start = np.array([0.0, 5.0, -9.0, -9.0, -9.0])
    end = np.array([0.0, 0.0, 0.0, 0.0, 9.0])
    context = np.array([False, True, True, True, True])
    assert span_minus_null(start, end, context, max_answer_tokens=3) == pytest.approx(5.0)
    assert span_minus_null(start, end, context, max_answer_tokens=4) == pytest.approx(14.0)


def test_span_minus_null_negative_when_null_wins() -> None:
    start = np.array([8.0, 0.0, 1.0])
    end = np.array([8.0, 1.0, 0.0])
    context = np.array([False, True, True])
    assert span_minus_null(start, end, context) < 0.0
    assert span_minus_null(start, end, np.zeros(3, dtype=np.bool_)) == 0.0


def test_cached_scorer_only_scores_uncached_pairs_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "scores.json"
    inner = FakeScorer()
    cached = CachedScorer(inner, ScoreCache(path))
    assert cached.score("redispatching units", ["redispatching units", "other"]) == [2.0, 0.0]
    assert cached.score("redispatching units", ["other", "units x"]) == [0.0, 1.0]
    assert inner.seen == [("redispatching units", 2), ("redispatching units", 1)]
    assert (cached.hits, cached.misses) == (1, 3)
    cached.cache.save()
    assert path.exists()

    reloaded_inner = FakeScorer()
    reloaded = CachedScorer(reloaded_inner, ScoreCache(path))
    assert reloaded.score("redispatching units", ["redispatching units", "units x"]) == [2.0, 1.0]
    assert reloaded_inner.seen == []
    assert (reloaded.hits, reloaded.misses) == (2, 0)


def test_cache_key_separates_scorers() -> None:
    cache = ScoreCache(None)
    a = CachedScorer(FakeScorer("a@1", 1.0), cache)
    b = CachedScorer(FakeScorer("b@1", 10.0), cache)
    assert a.score("units", ["units"]) == [1.0]
    assert b.score("units", ["units"]) == [10.0]


def test_scorers_aggregate_by_maximum() -> None:
    scorers = ModelScorers(FakeScorer(), FakeScorer(scale=2.0))
    values = scorers.signals("redispatching units", ["x", "redispatching", "redispatching units"])
    assert values == {"ce_top": 2.0, "qa_answerable": 4.0}
    assert scorers.signals("q", []) == {"ce_top": 0.0, "qa_answerable": 0.0}
    assert scorers.stats() == "no cache statistics"


def test_collect_signals_with_fake_scorers_adds_model_columns() -> None:
    qa = [
        {"q": "How is congestion relieved by redispatching?", "relevant_doc_ids": ["a"]},
        {"q": "What does imbalance settlement charge?", "relevant_doc_ids": ["b"]},
    ]
    unanswerable = ["Who won the football final?"]
    plain = collect_signals(_DOCS, qa, unanswerable)
    modelled = collect_signals(_DOCS, qa, unanswerable, ModelScorers(FakeScorer(), FakeScorer()))
    assert list(plain.scores) == list(SIGNALS)
    assert list(modelled.scores) == [*SIGNALS, *MODEL_SIGNALS]
    for name in SIGNALS:  # lexical signals are untouched by the model columns
        assert modelled.scores[name].tolist() == plain.scores[name].tolist()
    assert all(modelled.scores[name].shape == (3,) for name in MODEL_SIGNALS)
    assert modelled.scores["ce_top"][0] > modelled.scores["ce_top"][2]
    assert modelled.retrieval_miss.tolist() == plain.retrieval_miss.tolist()
