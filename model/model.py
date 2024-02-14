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
    radius_rounded = np.floor(radius)
    delta_nodes = pairwise_permutations_square(radius_rounded)
    delta_nodes.remove((0, 0))  # remove the BMU
    candidate_nodes = [tuple(np.array(bmu) + np.array(delta)) for delta in delta_nodes]

    neighbourhood_nodes = []
    for node in candidate_nodes:
        # prune nodes beyond grid limits (x,y) where x is width, y is height
        if node[0] < 0 or node[0] > grid_width or node[1] < 0 or node[1] > grid_height:
            continue

        # prune nodes outside the radius
        dist = np.linalg.norm(np.array(bmu) - np.array(node))
        if dist <= radius_rounded:
            neighbourhood_nodes.append(node)
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
