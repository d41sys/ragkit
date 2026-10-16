"""Tests for the bootstrap and Wilson interval helpers."""

from __future__ import annotations

import pytest

from ragkit.eval import bootstrap_ci, paired_bootstrap_ci, wilson_interval


def test_bootstrap_ci_of_constant_scores_is_degenerate() -> None:
    assert bootstrap_ci([1.0] * 20) == (1.0, 1.0, 1.0)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic() -> None:
    values = [0.0, 1.0] * 20
    mean, lo, hi = bootstrap_ci(values)
    assert mean == pytest.approx(0.5)
    assert lo < mean < hi
    assert lo > 0.3 and hi < 0.7
    assert bootstrap_ci(values) == (mean, lo, hi)


def test_bootstrap_ci_narrows_with_more_questions() -> None:
    small = bootstrap_ci([0.0, 1.0] * 5)
    large = bootstrap_ci([0.0, 1.0] * 100)
    assert (large[2] - large[1]) < (small[2] - small[1])


def test_bootstrap_ci_validates_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0], confidence=1.0)


def test_paired_bootstrap_separates_a_consistent_gap() -> None:
    a = [1.0, 0.8, 0.9, 1.0, 0.7, 0.9] * 5
    b = [x - 0.2 for x in a]
    mean, lo, hi = paired_bootstrap_ci(a, b)
    assert mean == pytest.approx(0.2)
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)
    assert paired_bootstrap_ci(a, a) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_ci([1.0], [1.0, 2.0])


def test_wilson_interval_matches_reference_values() -> None:
    lo, hi = wilson_interval(9, 10)
    assert lo == pytest.approx(0.5958, abs=1e-3)
    assert hi == pytest.approx(0.9821, abs=1e-3)
    lo, hi = wilson_interval(0, 12)
    assert lo == 0.0 and hi == pytest.approx(0.2425, abs=1e-3)


def test_wilson_interval_edge_cases() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(5, 5)
    assert hi == pytest.approx(1.0) and 0.5 < lo < 1.0
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(6, 5)
