import numpy as np

from sonority_rsa.rdm import sonority_rdm, upper_triangle


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
