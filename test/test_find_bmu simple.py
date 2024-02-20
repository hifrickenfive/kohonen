import numpy as np
from src.model import find_bmu_simple


def test_1x1x3_lower_than_values():
    grid_test = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    grid_test = grid_test.reshape((2, 2, 3))

    vector_test = np.array([-1, -1, -1])
    vector_test = vector_test.reshape((1, 1, 3))

    expected_idx = [0, 0]
    expected_d_squared = ((1 - -1) ** 2) * 3

    bmu, d_squared = find_bmu_simple(vector_test, grid_test)

    assert bmu == tuple(expected_idx)
    assert d_squared == expected_d_squared


def test_1x1x3_larger_than_values():
    grid_test = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    grid_test = grid_test.reshape((2, 2, 3))

    vector_test = np.array([5, 5, 5])
    vector_test = vector_test.reshape((1, 1, 3))

    expected_idx = [1, 1]
    expected_min_dist = 3

    bmu, d_squared = find_bmu_simple(vector_test, grid_test)

    assert bmu == tuple(expected_idx)
    assert d_squared == expected_min_dist


def test_3x2x1():
    grid_test = np.array([[1, 2], [3, 4], [5, 6]])
    grid_test = grid_test.reshape((3, 2, 1))

    vector_test = np.array([6])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [2, 1]
    expected_min_dist = 0

    bmu, d_squared = find_bmu_simple(vector_test, grid_test)

    assert bmu == tuple(expected_idx)
    assert d_squared == expected_min_dist**2


def test_larger_than_grid_vals():
    grid_test = np.array([[1, 2], [3, 4]])
    grid_test = grid_test.reshape((2, 2, 1))

    vector_test = np.array([6])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [1, 1]
    expected_min_dist = 2

    bmu, d_squared = find_bmu_simple(vector_test, grid_test)

    assert bmu == tuple(expected_idx)
    assert d_squared == expected_min_dist**2


def test_exact_match():
    grid_test = np.array([[1, 2], [3, 4]])
    grid_test = grid_test.reshape((2, 2, 1))

    vector_test = np.array([4])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [1, 1]
    expected_min_dist = 0

    bmu, d_squared = find_bmu_simple(vector_test, grid_test)

    assert bmu == tuple(expected_idx)
    assert d_squared == expected_min_dist**2


def test_smaller_than_grid_vals():
    grid_test = np.array([[1, 2], [3, 4]])
    grid_test = grid_test.reshape((2, 2, 1))

    vector_test = np.array([-1])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [0, 0]
    expected_min_dist = 2

    bmu, d_squared = find_bmu_simple(vector_test, grid_test)

    assert bmu == tuple(expected_idx)
    assert d_squared == expected_min_dist**2
