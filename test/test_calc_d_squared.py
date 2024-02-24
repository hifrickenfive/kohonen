import numpy as np
from src.model import get_neighbourhood_nodes, calc_d_squared


def test_radius1():
    bmu = np.array([2, 2])
    radius = 1
    grid_width, grid_height = 5, 5
    expected_d_squared = np.array([1, 1, 0, 1, 1])
    neighbourhood_nodes = get_neighbourhood_nodes(bmu, radius, grid_width, grid_height)
    d_squared = calc_d_squared(neighbourhood_nodes, bmu)
    np.testing.assert_array_equal(d_squared.flatten(), expected_d_squared)


def test_radius2():
    bmu = np.array([2, 2])
    radius = 2
    grid_width, grid_height = 5, 5

    expected_sq_d = np.array(
        [4.0, 2.0, 1.0, 2.0, 4.0, 1.0, 0.0, 1.0, 4.0, 2.0, 1.0, 2.0, 4.0]
    )
    neighbourhood_nodes = get_neighbourhood_nodes(bmu, radius, grid_width, grid_height)
    actual_sq_d = calc_d_squared(neighbourhood_nodes, bmu)
    np.testing.assert_array_equal(actual_sq_d.flatten(), expected_sq_d)
