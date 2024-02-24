import matplotlib.pyplot as plt
import numpy as np
from src.model import update_weights, calc_influence, calc_d_squared


def test_bmu():
    node_weights = np.array([1, 1, 1])
    bmu_weight = np.array([1, 1, 1])
    lr = 1
    radius = 1
    current_vector = np.array([1, 1, 1])

    expected_weights = np.array([1, 1, 1])

    updated_weights = update_weights(
        node_weights, bmu_weight, lr, radius, current_vector, influence_tuning_factor=1
    )

    assert list(expected_weights) == list(updated_weights)


def test_basic1x1():
    node_weights = np.array([2, 2, 2])
    bmu_weight = np.array([1, 1, 1])
    lr = 1
    radius = 1
    current_vector = np.array([1, 1, 1])

    expected_d_squared = np.array([3])
    expected_influence = np.exp(-3 / (2 * radius**2))

    influence = calc_influence(3, radius)
    assert expected_influence == influence

    d_squared = calc_d_squared(node_weights, bmu_weight)
    assert expected_d_squared == d_squared

    expected_weights = node_weights + lr * expected_influence * (
        current_vector - node_weights
    )

    updated_weights = update_weights(
        node_weights, bmu_weight, lr, radius, current_vector, influence_tuning_factor=1
    )

    assert list(expected_weights) == list(updated_weights)
