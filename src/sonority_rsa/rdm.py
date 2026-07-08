"""RDM construction and RSA scoring."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def upper_triangle(matrix):
    """
    Return the strict upper triangle of a square matrix.

    matrix: square numpy array
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('matrix must be square')

    rows, cols = np.triu_indices(matrix.shape[0], k=1)
    return matrix[rows, cols]


def correlation_rdm(vectors):
    """
    Compute a correlation-distance RDM for vectors.

    vectors: two-dimensional array of observations by features
    """
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise ValueError('vectors must be a two-dimensional array')
    if len(vectors) < 2:
        raise ValueError('at least two vectors are required')

    return squareform(pdist(vectors, metric='correlation'))


def sonority_rdm(values):
    """
    Compute an absolute-distance RDM for sonority values.

    values: one-dimensional sequence of numeric sonority values
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError('sonority values must be one-dimensional')
    if len(values) < 2:
        raise ValueError('at least two sonority values are required')

    return np.abs(values[:, None] - values[None, :])


def spearman_rsa(model_rdm, predictor_rdm):
    """
    Compare two RDM upper triangles with Spearman correlation.

    model_rdm: square model distance matrix
    predictor_rdm: square predictor distance matrix
    """
    model_upper = upper_triangle(model_rdm)
    predictor_upper = upper_triangle(predictor_rdm)

    if len(model_upper) != len(predictor_upper):
        raise ValueError('RDM upper triangles must have equal length')

    result = spearmanr(model_upper, predictor_upper)
    return result.statistic


def compute_sonority_rsa(vectors, sonority_values):
    """
    Compute RSA between vector geometry and sonority distances.

    vectors: two-dimensional array of hidden-state vectors
    sonority_values: one-dimensional sequence of sonority values
    """
    model_rdm = correlation_rdm(vectors)
    predictor_rdm = sonority_rdm(sonority_values)
    return spearman_rsa(model_rdm, predictor_rdm)
