import math

import pytest

from sonority_rsa.util_stats import score_statistics


def test_score_statistics_uses_sample_statistics_and_student_t_interval():
    statistics = score_statistics([1, 2, 3, 4])

    assert statistics == pytest.approx({
        'mean': 2.5,
        'sd': 1.2909944487,
        'sem': 0.6454972244,
        'mean_ci_lower': 0.44574,
        'mean_ci_upper': 4.55426,
        'n_total': 4,
        'n_valid': 4,
    }, abs=1e-5)


def test_score_statistics_omits_non_finite_values_but_counts_them():
    statistics = score_statistics(
        [1, float('nan'), 1, float('inf')])

    assert statistics == pytest.approx({
        'mean': 1,
        'sd': 0,
        'sem': 0,
        'mean_ci_lower': 1,
        'mean_ci_upper': 1,
        'n_total': 4,
        'n_valid': 2,
    })


def test_score_statistics_needs_two_values_for_dispersion_and_bounds():
    statistics = score_statistics([0.25])

    assert statistics['mean'] == pytest.approx(0.25)
    assert statistics['n_total'] == 1
    assert statistics['n_valid'] == 1
    for name in ['sd', 'sem', 'mean_ci_lower', 'mean_ci_upper']:
        assert math.isnan(statistics[name])


def test_score_statistics_returns_nan_without_finite_values():
    statistics = score_statistics([float('nan')])

    assert statistics['n_total'] == 1
    assert statistics['n_valid'] == 0
    for name in ['mean', 'sd', 'sem', 'mean_ci_lower', 'mean_ci_upper']:
        assert math.isnan(statistics[name])


@pytest.mark.parametrize('confidence', [0, 1, 95])
def test_score_statistics_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError, match='greater than 0 and less than 1'):
        score_statistics([1, 2], confidence=confidence)
