import numpy as np
from typing import List, Dict, Tuple


def find_bmu_vectorised(
    input_vector: np.ndarray, grid: Dict[Tuple[int, int], np.ndarray]
) -> Tuple[int, int]:
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
    sum_squared_diff = np.sum(
        (tiled_vector - tiled_grid) ** 2, axis=-1, keepdims=True
    )  # n_inputs x width x height x 1
    min_sum_squared_diff = np.min(
        sum_squared_diff, axis=(1, 2), keepdims=True
    )  # n_inputs x 1 x 1 x 1

    # Get indices of the minimum values
    _indices_of_min = np.argwhere(
        sum_squared_diff == min_sum_squared_diff
    )  # n_inputs x 4 (col 0: vectorIdx, col1: h, col2:w, col3: zeros). inspo https://stackoverflow.com/questions/30180241/numpy-get-the-column-and-row-index-of-the-minimum-value-of-a-2d-arra
    indices_of_min = _indices_of_min[
        :, [1, 2]
    ]  # slice to get cols 1 and 2, which are height, width grid indices
    # bmu = [tuple(indices) for indices in indices_of_min]
    return indices_of_min, min_sum_squared_diff


def find_bmu(
    grid: Dict[Tuple[int, int], np.ndarray], input_vector: np.ndarray
) -> Tuple[int, int]:
    """Find the best matching unit (BMU) in the grid for a given input vector
    Args:
        grid: keys are the grid coordinates, values are weight vectors
        input_vector: the input vector

    Returns:
        bmu: coordinates of the BMU in the grid
        dist_to_bmu: distance to the BMU
    """
    weight_vectors = np.array(
        list(grid.values())
    )  # convert to np so we can use broadcasting but n, 1, dim
    weight_vectors = np.squeeze(weight_vectors)  # n, dim
    distances_squared = np.sum(
        (weight_vectors - input_vector) ** 2, axis=1
    )  # axis=1 along each row
    min_index = np.argmin(distances_squared)
    dist_to_bmu = distances_squared[min_index]
    bmu = list(grid.keys())[min_index]
    return bmu, dist_to_bmu


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
    delta_x, delta_y = np.meshgrid(
        np.arange(-radius_rounded, radius_rounded + 1),
        np.arange(-radius_rounded, radius_rounded + 1),
    )  # 2d array of x and y deltas
    delta_nodes = np.column_stack(
        (delta_x.ravel(), delta_y.ravel())
    )  # flatten each 2d array and stack together to form 2 columns of x,y pairs
    delta_nodes = delta_nodes[
        ~np.all(delta_nodes == 0, axis=1)
    ]  # remove bmu (0,0) by scanning across the rows, i.e. along columns
    candidate_nodes = np.array(bmu) + delta_nodes  # broadcast

    # Prune nodes beyond grid limits (x,y) where x is height, y is width
    valid_nodes = (
        (candidate_nodes[:, 0] >= 0)
        & (candidate_nodes[:, 0] < grid_height)
        & (candidate_nodes[:, 1] >= 0)
        & (candidate_nodes[:, 1] < grid_width)
    )
    candidate_nodes = candidate_nodes[valid_nodes]  # filter

    # Prune nodes outside the radius
    distances_sq = np.sum((candidate_nodes - np.array(bmu)) ** 2, axis=1)
    within_radius = distances_sq <= radius**2

    return candidate_nodes[within_radius]


def calc_influence(node: Tuple[int, int], bmu: Tuple[int, int], radius: float) -> float:
    """
    Calculate the influence of the BMU on a given node

    Args:
        node: the coordinates of the node
        bmu: the coordinates of the BMU
        radius: the radius of the neighbourhood from the BMU

    Returns:
        influence: the influence of the BMU on the node
    """
    dist = np.linalg.norm(np.array(bmu) - np.array(node))
    influence = np.exp(-(dist**2) / (2 * radius**2))
    return influence
