import numpy as np
from src.model import find_bmu_simple


def test_3x2():
    grid_test = np.array([[1, 2], [3, 4], [5, 6]])
    grid_test = grid_test.reshape((3, 2, 1))

    vector_test = np.array([6])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [2, 1]

    bmu = find_bmu_simple(vector_test, grid_test)

    assert bmu == tuple(expected_idx)


def test_larger_than_grid_vals():
    grid_test = np.array([[1, 2], [3, 4]])
    grid_test = grid_test.reshape((2, 2, 1))

    vector_test = np.array([6])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [1, 1]
    expected_min_dist = 2

    bmu = find_bmu_simple(vector_test, grid_test)

    assert bmu == tuple(expected_idx)


def test_exact_match():
    grid_test = np.array([[1, 2], [3, 4]])
    grid_test = grid_test.reshape((2, 2, 1))

    vector_test = np.array([4])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [1, 1]

    bmu = find_bmu_simple(vector_test, grid_test)

    assert bmu == tuple(expected_idx)


def test_smaller_than_grid_vals():
    grid_test = np.array([[1, 2], [3, 4]])
    grid_test = grid_test.reshape((2, 2, 1))

    vector_test = np.array([-1])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [0, 0]

    bmu = find_bmu_simple(vector_test, grid_test)

    assert bmu == tuple(expected_idx)
