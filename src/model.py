import numpy as np
from typing import List


def find_bmu_vectorised(input_vector: np.ndarray, grid: np.ndarray) -> list:
    """Find the best matching unit (BMU) in the grid for a given input vector
    Args:
        input_vector: the input vector
        grid: keys are the grid coordinates, values are weight vectors
    Assumptions:
        - the input vector has the same dimension as the weight vectors
        - the grid is a 2D grid
    Returns:
        bmu: the coordinates of the BMUs in the grid
        min_dist: the distance between the BMU and the input vectors
    """
    # Reshape input vector and grid for vectorised operations
    n_inputs, input_dim = input_vector.shape[0], input_vector.shape[1]
    grid_height, grid_width, grid_dim = grid.shape[0], grid.shape[1], grid.shape[2]

    reshaped_input_vector = input_vector.reshape((n_inputs, 1, 1, input_dim))
    tiled_vector = np.tile(reshaped_input_vector, (1, grid_height, grid_width, 1))

    reshaped_grid = grid.reshape((1, grid_height, grid_width, grid_dim))
    tiled_grid = np.tile(reshaped_grid, (n_inputs, 1, 1, 1))

    # Find min along the colour channels i.e. the last axis
    # sum_squared_diff shape is n_inputs x height x width x 1
    sum_squared_diff = np.sum((tiled_vector - tiled_grid) ** 2, axis=-1, keepdims=True)
    min_sum_squared_diff = np.min(sum_squared_diff, axis=(1, 2), keepdims=True)

    # Get indices of the minimum values
    # Previously used np.argwhere but there are corner cases where it doesn't return unique indices
    bmus = find_unique_bmu_indices(sum_squared_diff)
    return bmus, min_sum_squared_diff


def find_unique_bmu_indices(sum_squared_diff: np.ndarray) -> np.ndarray:
    unique_indices = []
    for i in range(sum_squared_diff.shape[0]):
        flat_diff = sum_squared_diff[i].flatten()
        min_index = np.argmin(flat_diff)  # returns first occurence only

        # Convert this flat index back to the original indices
        unique_index = np.unravel_index(min_index, sum_squared_diff[i].shape)
        unique_indices.append(unique_index[:-1])

    return np.array(unique_indices)


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
    radius_rounded = int(np.floor(radius))  # int faster than float ops

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
