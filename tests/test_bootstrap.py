import warnings

import numpy as np
import pytest

from sonority_rsa.bootstrap import (compute_bootstrap, replay_sampled_keys,
    sample_syllables, summarize_bootstrap)
from sonority_rsa.fetch import SyllableData, SyllablePopulation


def make_population():
    syllables = [
        SyllableData('s1', ['p', 'a'], [0, 5], [[1.0, 0.0], [0.0, 1.0]]),
        SyllableData('s2', ['s', 't'], [1, 0], [[0.8, 0.2], [0.9, 0.1]]),
        SyllableData('s3', ['m', 'l'], [2, 3], [[0.3, 0.7], [0.2, 0.8]]),
    ]
    return SyllablePopulation('wav2vec2', 0, 500, syllables, skipped={})


def test_sampling_draws_distinct_syllables():
    population = make_population()
    rng = FixedRng([0, 2])

    vectors, sonority, keys = sample_syllables(population, 2, rng)

    assert keys == ['s1', 's3']
    assert len(set(keys)) == len(keys)
    assert sonority.tolist() == [0, 5, 2, 3]
    assert vectors.shape == (4, 2)


def test_sampling_rejects_more_syllables_than_population():
    population = make_population()
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match='exceeds the population size'):
        sample_syllables(population, 4, rng)


def test_compute_bootstrap_returns_one_score_per_repetition():
    population = make_population()

    scores = compute_bootstrap(population, n_syllables=3, n_bootstraps=5,
        random_state=1)

    assert len(scores) == 5
    assert all(-1 <= score <= 1 for score in scores)


def test_compute_bootstrap_is_deterministic_for_a_seed():
    population = make_population()

    first = compute_bootstrap(population, 3, 4, random_state=1)
    second = compute_bootstrap(population, 3, 4, random_state=1)

    assert first == second


def test_compute_bootstrap_warns_on_nan_scores():
    syllables = [
        SyllableData('s1', ['p', 'a'], [0, 5], [[1.0, 1.0], [1.0, 1.0]]),
        SyllableData('s2', ['s', 't'], [1, 0], [[1.0, 1.0], [1.0, 1.0]]),
    ]
    population = SyllablePopulation('wav2vec2', 0, 500, syllables, skipped={})

    with pytest.warns(UserWarning, match='2 of 2 bootstrap RSA scores'):
        scores = compute_bootstrap(population, n_syllables=2, n_bootstraps=2,
            random_state=1)

    assert all(np.isnan(score) for score in scores)


def test_compute_bootstrap_does_not_warn_without_nan_scores():
    rng = np.random.default_rng(0)
    syllables = [
        SyllableData(f's{i}', ['p', 'a'], [i, i + 1], rng.random((2, 4)))
        for i in range(5)
    ]
    population = SyllablePopulation('wav2vec2', 0, 500, syllables, skipped={})

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compute_bootstrap(population, n_syllables=3, n_bootstraps=2,
            random_state=1)


def test_compute_bootstrap_warns_when_n_syllables_equals_population():
    population = make_population()

    with pytest.warns(UserWarning, match='equals the population size'):
        compute_bootstrap(population, n_syllables=3, n_bootstraps=2,
            random_state=1)


def test_summarize_bootstrap_returns_one_row_per_layer():
    scores = {0: [0.1, 0.2, 0.3], 1: [0.4, 0.5, 0.6]}

    summary = summarize_bootstrap(scores)

    assert [row['layer'] for row in summary] == [0, 1]
    assert summary[0]['mean_rsa'] == pytest.approx(0.2)
    assert summary[0]['n_bootstraps'] == 3
    assert summary[0]['ci_lower'] <= summary[0]['ci_upper']


def test_replay_reproduces_the_sampled_keys():
    population = make_population()
    seed = 7
    rng = np.random.default_rng(seed)
    drawn = [sample_syllables(population, 2, rng)[2] for _ in range(3)]

    replayed = replay_sampled_keys(population.keys, seed, n_syllables=2,
        n_bootstraps=3)

    assert replayed == drawn


class FixedRng:

    def __init__(self, indices):
        self.indices = indices

    def choice(self, a, size, replace):
        assert replace is False
        return np.asarray(self.indices)
