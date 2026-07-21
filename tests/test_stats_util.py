import math

import pytest

from sonority_rsa.stats_util import mean_confidence_interval


def test_mean_confidence_interval_uses_student_t_interval():
    mean, lower, upper = mean_confidence_interval([1, 2, 3, 4])

    assert mean == pytest.approx(2.5)
    assert lower == pytest.approx(0.44574, abs=1e-5)
    assert upper == pytest.approx(4.55426, abs=1e-5)


def test_mean_confidence_interval_omits_non_finite_values():
    mean, lower, upper = mean_confidence_interval(
        [1, float('nan'), 1, float('inf')])

    assert (mean, lower, upper) == pytest.approx((1, 1, 1))


def test_mean_confidence_interval_needs_two_values_for_bounds():
    mean, lower, upper = mean_confidence_interval([0.25])

    assert mean == pytest.approx(0.25)
    assert math.isnan(lower)
    assert math.isnan(upper)


def test_mean_confidence_interval_returns_nan_without_finite_values():
    summary = mean_confidence_interval([float('nan')])

    assert all(math.isnan(value) for value in summary)


@pytest.mark.parametrize('confidence', [0, 1, 95])
def test_mean_confidence_interval_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match='greater than 0 and less than 1'):
        mean_confidence_interval([1, 2], confidence=confidence)
