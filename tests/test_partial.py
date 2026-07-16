import numpy as np
import pytest

from sonority_rsa import partial_spearman_rsa
from sonority_rsa.rdm import spearman_rsa


def _rdm(upper):
    """Build a four-by-four symmetric RDM from six upper values."""
    matrix = np.zeros((4, 4), dtype=float)
    rows, cols = np.triu_indices(4, k=1)
    matrix[rows, cols] = upper
    matrix[cols, rows] = upper
    return matrix


def test_partial_spearman_matches_known_residual_correlation():
    model = _rdm([1, 5, 2, 6, 3, 4])
    target = _rdm([6, 1, 5, 2, 4, 3])
    control = _rdm([1, 2, 3, 4, 5, 6])

    result = partial_spearman_rsa(model, target, control)

    assert result == pytest.approx(-0.9443921798277298)


def test_partial_spearman_average_ranks_tied_distances():
    model = _rdm([0, 0, 1, 2, 2, 3])
    target = _rdm([1, 2, 1, 3, 2, 3])
    control = _rdm([3, 1, 1, 2, 2, 3])

    result = partial_spearman_rsa(model, target, control)

    assert result == pytest.approx(0.7181848464596079)


def test_partial_spearman_removes_shared_control_association():
    model = _rdm([1, 2, 3, 4, 6, 5])
    target = _rdm([1, 2, 4, 3, 5, 6])
    control = _rdm([1, 2, 3, 4, 5, 6])

    ordinary_rsa = spearman_rsa(model, target)
    partial_rsa = partial_spearman_rsa(model, target, control)

    assert ordinary_rsa == pytest.approx(0.8857142857142858)
    assert partial_rsa == pytest.approx(-0.02941176470588235)


def test_partial_spearman_rejects_unequal_triangle_lengths():
    with pytest.raises(ValueError,
            match='RDM upper triangles must have equal length'):
        partial_spearman_rsa(np.zeros((3, 3)), np.zeros((4, 4)),
            np.zeros((4, 4)))


def test_partial_spearman_uses_square_rdm_validation():
    with pytest.raises(ValueError, match='matrix must be square'):
        partial_spearman_rsa(np.zeros((3, 2)), np.zeros((3, 3)),
            np.zeros((3, 3)))


@pytest.mark.parametrize(('rdm_name', 'invalid_value'), [
    ('model', np.nan),
    ('target', np.inf),
    ('control', -np.inf),
])
def test_partial_spearman_returns_nan_for_nonfinite_values(rdm_name,
        invalid_value):
    rdms = {
        'model': _rdm([1, 5, 2, 6, 3, 4]),
        'target': _rdm([6, 1, 5, 2, 4, 3]),
        'control': _rdm([1, 2, 3, 4, 5, 6]),
    }
    rdms[rdm_name][0, 1] = invalid_value

    result = partial_spearman_rsa(rdms['model'], rdms['target'],
        rdms['control'])

    assert np.isnan(result)


@pytest.mark.parametrize(('model_upper', 'target_upper', 'control_upper'), [
    ([1, 5, 2, 6, 3, 4], [6, 1, 5, 2, 4, 3], [1, 1, 1, 1, 1, 1]),
    ([1, 2, 3, 4, 5, 6], [6, 1, 5, 2, 4, 3], [1, 2, 3, 4, 5, 6]),
    ([1, 5, 2, 6, 3, 4], [6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6]),
])
def test_partial_spearman_returns_nan_without_residual_variation(
        model_upper, target_upper, control_upper):
    result = partial_spearman_rsa(_rdm(model_upper), _rdm(target_upper),
        _rdm(control_upper))

    assert np.isnan(result)
