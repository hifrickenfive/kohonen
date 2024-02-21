import numpy as np
from src.model import get_neighbourhood_nodes
from typing import Tuple


def test_central_bmu_small_radius():
    bmu = np.array([5, 5])
    radius = 1
    grid_width = 10
    grid_height = 10
    expected_neighbours = np.array([[4, 5], [5, 4], [5, 6], [6, 5]])
    expected_neighbours = {(4, 5), (5, 4), (5, 6), (6, 5)}

    neighbourhood_nodes = get_neighbourhood_nodes(bmu, radius, grid_width, grid_height)
    assert set(tuple(node) for node in neighbourhood_nodes) == expected_neighbours


def test_large_radius_exceeding_grid():
    bmu = np.array([1, 1])
    radius = 10
    grid_width = 2
    grid_height = 2
    expected_neighbours = {(0, 0), (0, 1), (1, 0)}
    neighbourhood_nodes = get_neighbourhood_nodes(bmu, radius, grid_width, grid_height)
    assert set(tuple(node) for node in neighbourhood_nodes) == expected_neighbours


def test_radius_zero():
    bmu = np.array([3, 3])
    radius = 0
    grid_width = 10
    grid_height = 10
    expected_neighbours = set()
    neighbourhood_nodes = get_neighbourhood_nodes(bmu, radius, grid_width, grid_height)
    assert set(tuple(node) for node in neighbourhood_nodes) == expected_neighbours


def test_float_radius():
    bmu = np.array([5, 5])
    radius = 1.5
    grid_width = 10
    grid_height = 10
    # Expected neighbors might include those within a radius of 1.5 but not beyond
    expected_neighbours = {
        (4, 5),
        (5, 4),
        (6, 5),
        (5, 6),
        (4, 4),
        (6, 6),
        (4, 6),
        (6, 4),
    }
    neighbourhood_nodes = get_neighbourhood_nodes(bmu, radius, grid_width, grid_height)
    neighbourhood_nodes_set = set(tuple(node) for node in neighbourhood_nodes)

    assert neighbourhood_nodes_set == expected_neighbours


def test_corner():
    grid_width = 3
    grid_height = 3
    bmu = np.array([0, 2])
    radius = 1
    expected_neighbours = {(0, 1), (1, 2)}

    neighbourhood_nodes = get_neighbourhood_nodes(bmu, radius, grid_width, grid_height)
    neighbourhood_nodes_set = set(tuple(node) for node in neighbourhood_nodes)

    assert neighbourhood_nodes_set == expected_neighbours
