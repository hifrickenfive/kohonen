import numpy as np
from model.model import find_bmu_vectorised


def test_3x2():
    grid_test = np.array([[1, 2], [3, 4], [5, 6]])
    grid_test = grid_test.reshape((3, 2, 1))

    vector_test = np.array([6])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [2, 1]
    expected_min_dist = 0

    idx, min_sum_squared_diff = find_bmu_vectorised(vector_test, grid_test)
    actual_idx = list(np.squeeze(idx))
    actual_min_dist = np.sqrt(np.squeeze(min_sum_squared_diff))

    assert actual_idx == expected_idx
    assert actual_min_dist == expected_min_dist


def test_larger_than_grid_vals():
    grid_test = np.array([[1, 2], [3, 4]])
    grid_test = grid_test.reshape((2, 2, 1))

    vector_test = np.array([6])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [1, 1]
    expected_min_dist = 2

    idx, min_sum_squared_diff = find_bmu_vectorised(vector_test, grid_test)
    actual_idx = list(np.squeeze(idx))
    actual_min_dist = np.sqrt(np.squeeze(min_sum_squared_diff))

    assert actual_idx == expected_idx
    assert actual_min_dist == expected_min_dist


def test_exact_match():
    grid_test = np.array([[1, 2], [3, 4]])
    grid_test = grid_test.reshape((2, 2, 1))

    vector_test = np.array([4])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [1, 1]
    expected_min_dist = 0

    idx, min_sum_squared_diff = find_bmu_vectorised(vector_test, grid_test)
    actual_idx = list(np.squeeze(idx))
    actual_min_dist = np.sqrt(np.squeeze(min_sum_squared_diff))

    assert actual_idx == expected_idx
    assert actual_min_dist == expected_min_dist


def test_smaller_than_grid_vals():
    grid_test = np.array([[1, 2], [3, 4]])
    grid_test = grid_test.reshape((2, 2, 1))

    vector_test = np.array([-1])
    vector_test = vector_test.reshape((1, 1))

    expected_idx = [0, 0]
    expected_min_dist = 2

    idx, min_sum_squared_diff = find_bmu_vectorised(vector_test, grid_test)
    actual_idx = list(np.squeeze(idx))
    actual_min_dist = np.sqrt(np.squeeze(min_sum_squared_diff))

    assert actual_idx == expected_idx
    assert actual_min_dist == expected_min_dist
