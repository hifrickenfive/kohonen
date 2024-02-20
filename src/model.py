import numpy as np
from typing import List


def update_weights(node_weights, bmu_weight, lr, radius, current_vector):
    d_squared = calc_d_squared(node_weights, bmu_weight)
    influence = calc_influence(d_squared, radius)  # (num nodes, 1)
    updated_weights = node_weights + lr * influence * (current_vector - node_weights)
    return updated_weights


def find_bmu_simple(current_vector: np.ndarray, grid: np.ndarray):
    d_squared = np.sum((grid - current_vector) ** 2, axis=2)  # sum colour pixel dim
    _bmu_idx = np.argmin(d_squared)
    bmu = np.unravel_index(_bmu_idx, d_squared.shape)  # tuple
    return bmu


def get_neighbourhood_nodes(
    bmu: np.ndarray, radius: float, grid_width: int, grid_height: int
) -> List[np.ndarray]:
    """
    Get the nodes in the neighbourhood of the BMU given a radius

    Args:
        bmu: coordinates of the BMU
        radius: the radius of the neighbourhood
        grid_width: the width of the grid
        grid_height: the height of the grid

    Returns:
        neighbourhood_nodes: list of nodes in the neighbourhood of the BMU
    """
    # Reduce search space to a square around the BMU
    radius_rounded = int(np.ceil(radius))  # int faster than float ops

    # Create 2D array of x and y deltas
    delta_x, delta_y = np.meshgrid(
        np.arange(-radius_rounded, radius_rounded + 1),
        np.arange(-radius_rounded, radius_rounded + 1),
    )

    # Flatten each 2d array and stack together to form 2 columns of x,y pairs
    delta_nodes = np.column_stack((delta_x.ravel(), delta_y.ravel()))

    # Remove bmu (0,0) by scanning across the rows, i.e. along columns
    delta_nodes = delta_nodes[~np.all(delta_nodes == 0, axis=1)]

    candidate_nodes = np.array(bmu) + delta_nodes

    # Prune nodes beyond grid limits (x,y) where x is height, y is width
    valid_nodes = (
        (candidate_nodes[:, 0] >= 0)
        & (candidate_nodes[:, 0] < grid_height)
        & (candidate_nodes[:, 1] >= 0)
        & (candidate_nodes[:, 1] < grid_width)
    )
    pruned_nodes = candidate_nodes[valid_nodes]
    distances_sq = np.sum((pruned_nodes - np.array(bmu)) ** 2, axis=1)
    within_radius = distances_sq <= radius**2

    return pruned_nodes[within_radius]


def calc_influence(d_squared, radius) -> float:
    """Calculate the influence of a node based on its distance from the BMU

    Args:
        d_squared: euclidean distance squared
        radius: radius of the neighbourhood

    Returns:
        influence: the influence of the node
    """
    return np.exp(-d_squared / (2 * radius**2))


def calc_d_squared(neighbourhood_nodes, bmu):
    """Calculate the squared euclidean distance between the BMU and the neighbourhood nodes

    Args:
        neighbourhood_nodes: the nodes in the neighbourhood of the BMU
        bmu: the best matching unit

    Returns:
        d_squared: the squared euclidean distance between the BMU and the neighbourhood nodes
    """
    d_squared = np.sum(
        (neighbourhood_nodes - bmu) ** 2,
        axis=-1,
        keepdims=True,
    )
    return d_squared
