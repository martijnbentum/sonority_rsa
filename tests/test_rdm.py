import numpy as np
import pytest

from sonority_rsa.rdm import (compute_intensity_rsa,
    compute_sonority_random_baseline, compute_sonority_rsa, intensity_rdm,
    sonority_rdm, upper_triangle)


def test_upper_triangle_extraction():
    matrix = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0],
    ])

    result = upper_triangle(matrix)

    assert result.tolist() == [1, 2, 3]


def test_sonority_rdm_values():
    values = np.array([0, 2, 5])

    result = sonority_rdm(values)

    expected = np.array([
        [0, 2, 5],
        [2, 0, 3],
        [5, 3, 0],
    ])
    np.testing.assert_array_equal(result, expected)


def test_intensity_rdm_values():
    values = np.array([40, 45, 55])

    result = intensity_rdm(values)

    expected = np.array([
        [0, 5, 15],
        [5, 0, 10],
        [15, 10, 0],
    ])
    np.testing.assert_array_equal(result, expected)


def test_compute_intensity_rsa_returns_correlation():
    vectors = np.random.default_rng(0).normal(size=(12, 4))
    intensity = np.arange(12)

    result = compute_intensity_rsa(vectors, intensity)

    assert -1 <= result <= 1


def test_random_baseline_shuffles_sonority_once():
    vectors = np.array([
        [0.1, 0.3, 0.8],
        [0.7, 0.2, 0.4],
        [0.5, 0.9, 0.1],
        [0.2, 0.6, 0.7],
    ])
    sonority = np.array([0, 1, 3, 5])
    seed = 7
    shuffled = np.random.default_rng(seed).permutation(sonority)

    result = compute_sonority_random_baseline(vectors, sonority,
        random_state=seed)

    assert result == pytest.approx(compute_sonority_rsa(vectors, shuffled))
    np.testing.assert_array_equal(np.sort(shuffled), np.sort(sonority))


def test_random_baseline_defaults_to_reproducible_seed():
    vectors = np.random.default_rng(0).normal(size=(12, 4))
    sonority = np.arange(12)

    first = compute_sonority_random_baseline(vectors, sonority)
    second = compute_sonority_random_baseline(vectors, sonority)

    assert first == second


def test_random_baseline_accepts_generator():
    vectors = np.random.default_rng(0).normal(size=(12, 4))
    sonority = np.arange(12)
    actual_rng = np.random.default_rng(9)
    expected_rng = np.random.default_rng(9)

    result = compute_sonority_random_baseline(vectors, sonority,
        random_state=actual_rng)
    expected = compute_sonority_rsa(vectors,
        expected_rng.permutation(sonority))

    assert result == pytest.approx(expected)
