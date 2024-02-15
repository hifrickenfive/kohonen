import numpy as np
from data_preprocessing.grid_helper import pairwise_permutations_square
from typing import List, Dict, Tuple


def find_bmu(
    grid: Dict[Tuple[int, int], np.ndarray], input_vector: np.ndarray
) -> Tuple[int, int]:
    """Find the best matching unit (BMU) in the grid for a given input vector
    Args:
        grid: keys are the grid coordinates, values are weight vectors
        input_vector: the input vector

    Returns:
        bmu: coordinates of the BMU in the grid
    """
    weight_vectors = np.array(
        list(grid.values())
    )  # convert to np so we can use broadcasting but n, 1, dim
    weight_vectors = np.squeeze(weight_vectors)  # n, dim
    distances = np.linalg.norm(
        weight_vectors - input_vector, axis=1
    )  # axis=1 gives us the norm of each row
    min_index = np.argmin(distances)
    bmu = list(grid.keys())[min_index]
    return bmu


def calc_neighbourhood_radius(
    current_radius: float, max_iter: int, current_iter: int
) -> float:
    """
    Calculate the neighbourhood radius at a given iteration

    Args:
        current_radius: the current radius
        max_iter: the maximum number of iterations
        current_iter: the current iteration

    Returns:
        radius: the updated radius
    """
    updated_radius = current_radius * np.exp(-current_iter / max_iter)
    return updated_radius


def get_neighbourhood_nodes(
    bmu: Tuple[int, int], radius: float, grid_width: int, grid_height: int
) -> List[Tuple[int, int]]:
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
    # delta_nodes = pairwise_permutations_square(radius_rounded)
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
    # candidate_nodes = [tuple(np.array(bmu) + np.array(delta)) for delta in delta_nodes]
    candidate_nodes = (
        np.array(bmu) + delta_nodes
    )  # broadcast is faster than list comprehension

    # prune nodes beyond grid limits (x,y) where x is width, y is height
    valid_nodes = (candidate_nodes[:, 0] >= 0) & (candidate_nodes[:, 0] < grid_width) & \
                  (candidate_nodes[:, 1] >= 0) & (candidate_nodes[:, 1] < grid_height)

    # Prune nodes outside the radius
    distances = np.linalg.norm(candidate_nodes - np.array(bmu), axis=1)
    within_radius = distances <= radius

    # Bit logic the indices, then get values from candidate nodes
    _neighbourhood_nodes = candidate_nodes[valid_nodes & within_radius]
    neighbourhood_nodes = [tuple(node) for node in _neighbourhood_nodes]

    return neighbourhood_nodes


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
