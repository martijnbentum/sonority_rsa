import warnings

import numpy as np
import pytest

from sonority_rsa.sampling import (compute_rsa_scores, replay_sampled_keys,
    sample_syllables, summarize_rsa_scores)
from sonority_rsa.fetch import SyllableData, SyllablePopulation


def make_population(n_syllables=90):
    rng = np.random.default_rng(0)
    syllables = [
        SyllableData(f's{i}', ['p', 'a'], [0, 5], rng.random((2, 4)))
        for i in range(n_syllables)
    ]
    return SyllablePopulation('wav2vec2', 0, 500, syllables, skipped={})


def test_sampling_draws_distinct_syllables():
    population = make_population()
    rng = FixedRng(np.arange(30))

    vectors, sonority, keys = sample_syllables(population, 30, rng)

    assert keys == [f's{i}' for i in range(30)]
    assert len(set(keys)) == len(keys)
    assert sonority.tolist() == [0, 5] * 30
    assert vectors.shape == (60, 4)


def test_sampling_rejects_more_syllables_than_population():
    population = make_population()
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match='exceeds the population size'):
        sample_syllables(population, 91, rng)


def test_compute_rsa_scores_returns_one_score_per_subset():
    population = make_population()

    scores = compute_rsa_scores(population, subset_size=30, n_subsets=5,
        random_state=1)

    assert len(scores) == 5
    assert all(-1 <= score <= 1 for score in scores)


def test_compute_rsa_scores_is_deterministic_for_a_seed():
    population = make_population()

    first = compute_rsa_scores(population, 30, 4, random_state=1)
    second = compute_rsa_scores(population, 30, 4, random_state=1)

    assert first == second


def test_compute_rsa_scores_warns_on_nan_scores():
    syllables = [
        SyllableData(f's{i}', ['p', 'a'], [0, 5],
            [[1.0, 1.0], [1.0, 1.0]])
        for i in range(90)
    ]
    population = SyllablePopulation('wav2vec2', 0, 500, syllables, skipped={})

    with pytest.warns(UserWarning, match='2 of 2 RSA scores'):
        scores = compute_rsa_scores(population, subset_size=30, n_subsets=2,
            random_state=1)

    assert all(np.isnan(score) for score in scores)


def test_compute_rsa_scores_does_not_warn_without_nan_scores():
    rng = np.random.default_rng(0)
    syllables = [
        SyllableData(f's{i}', ['p', 'a'], [i, i + 1], rng.random((2, 4)))
        for i in range(90)
    ]
    population = SyllablePopulation('wav2vec2', 0, 500, syllables, skipped={})

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compute_rsa_scores(make_population(90), subset_size=30, n_subsets=2,
            random_state=1)


def test_compute_rsa_scores_marks_single_sonority_subset_nan():
    rng = np.random.default_rng(0)
    syllables = [
        SyllableData(f's{i}', ['p', 'a'], [3, 3], rng.random((2, 4)))
        for i in range(30)
    ]
    population = SyllablePopulation('wav2vec2', 0, 500, syllables, skipped={})

    with pytest.warns(UserWarning, match='1 of 1 RSA scores are NaN'):
        scores = compute_rsa_scores(population, subset_size=30, n_subsets=1,
            random_state=1)

    assert np.isnan(scores[0])


def test_compute_rsa_scores_rejects_repeated_full_population_draws():
    population = make_population()

    with pytest.raises(ValueError, match='n_subsets must be 1'):
        compute_rsa_scores(population, subset_size=90, n_subsets=2,
            random_state=1)


def test_compute_rsa_scores_allows_one_full_population_draw():
    population = make_population()

    scores = compute_rsa_scores(population, subset_size=90, n_subsets=1,
        random_state=1)

    assert len(scores) == 1


def test_compute_rsa_scores_warns_for_large_sampling_fraction():
    population = make_population(60)

    with pytest.warns(UserWarning, match='at least 50%'):
        compute_rsa_scores(population, subset_size=30, n_subsets=1,
            random_state=1)


def test_compute_rsa_scores_reports_invalid_subset_reasons():
    rng = np.random.default_rng(0)
    syllables = [
        SyllableData(f's{i}', ['p', 'a'], [3, 3], rng.random((2, 4)))
        for i in range(30)
    ]
    population = SyllablePopulation('wav2vec2', 0, 500, syllables, skipped={})

    with pytest.warns(UserWarning, match='1 of 1 RSA scores are NaN'):
        scores, invalid_subsets = compute_rsa_scores(
            population, subset_size=30, n_subsets=1, random_state=1,
            return_diagnostics=True)

    assert np.isnan(scores[0])
    assert invalid_subsets == {
        'invalid_subsets': {
            'single_sonority_class': 1,
            'undefined_vector_distance': 0,
        },
        'invalid_reasons': ['single_sonority_class'],
    }


def test_compute_rsa_scores_rejects_small_subsets():
    population = make_population()

    with pytest.raises(ValueError, match='at least 30'):
        compute_rsa_scores(population, subset_size=29, n_subsets=1,
            random_state=1)


def test_summarize_rsa_scores_returns_one_row_per_layer():
    scores = {0: [0.1, 0.2, 0.3], 1: [0.4, 0.5, 0.6]}

    summary = summarize_rsa_scores(scores)

    assert [row['layer'] for row in summary] == [0, 1]
    assert summary[0]['mean_rsa'] == pytest.approx(0.2)
    assert summary[0]['n_subsets'] == 3
    assert summary[0]['n_subsets_valid'] == 3
    assert summary[0]['ci_lower'] <= summary[0]['ci_upper']


def test_summarize_rsa_scores_accepts_fractional_confidence_level():
    summary = summarize_rsa_scores({0: [0.0, 0.25, 0.5, 0.75, 1.0]}, ci=0.8)

    assert summary[0]['ci_lower'] == pytest.approx(0.1)
    assert summary[0]['ci_upper'] == pytest.approx(0.9)


def test_summarize_rsa_scores_warns_for_low_confidence_level():
    with pytest.warns(UserWarning, match='unexpected confidence interval'):
        summarize_rsa_scores({0: [0.1, 0.2, 0.3]}, ci=0.8)


@pytest.mark.parametrize('ci', [0, 1, 95])
def test_summarize_rsa_scores_rejects_invalid_confidence_level(ci):
    with pytest.raises(ValueError, match='greater than 0 and less than 1'):
        summarize_rsa_scores({0: [0.1, 0.2, 0.3]}, ci=ci)


def test_summarize_rsa_scores_ignores_nan_but_counts_all_subsets():
    scores = {0: [0.1, float('nan'), 0.3]}

    summary = summarize_rsa_scores(scores)

    assert summary[0]['n_subsets'] == 3
    assert summary[0]['n_subsets_valid'] == 2
    assert summary[0]['mean_rsa'] == pytest.approx(0.2)


def test_summarize_rsa_scores_handles_all_nan():
    scores = {0: [float('nan'), float('nan')]}

    summary = summarize_rsa_scores(scores)

    assert summary[0]['n_subsets'] == 2
    assert summary[0]['n_subsets_valid'] == 0
    assert np.isnan(summary[0]['mean_rsa'])
    assert np.isnan(summary[0]['ci_lower'])


def test_replay_reproduces_the_sampled_keys():
    population = make_population()
    seed = 7
    rng = np.random.default_rng(seed)
    drawn = [sample_syllables(population, 30, rng)[2] for _ in range(3)]

    replayed = replay_sampled_keys(population.keys, seed, subset_size=30,
        n_subsets=3)

    assert replayed == drawn


class FixedRng:

    def __init__(self, indices):
        self.indices = indices

    def choice(self, a, size, replace):
        assert replace is False
        return np.asarray(self.indices)
