import numpy as np
from typing import List, Dict, Tuple


def initialise_grid(
    grid_width: int, grid_height: int, dim: int
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Initialise the grid with random weight vectors

    Args:
        grid_width: The width of the grid.
        grid_height: The height of the grid.

    Returns:
        grid: grid coordinates mapped to weight vectors.
    """
    all_nodes = pairwise_permutations_grid(grid_width, grid_height)
    grid = dict()
    for node in all_nodes:
        grid[node] = np.random.random((1, dim))
    return grid


def pairwise_permutations_grid(x: int, y: int) -> List[Tuple[int, int]]:
    """Create a list of tuples of all permutations of values in the range [0, x] and [0, y]

    Args:
        x: the maximum value in the x-axis (columns).
        y: the maximum value in the y-axis (rows).

    Returns:
        pairwise_tuples
    """
    # Create an array of values in the range [-r, +r]
    x_values = np.arange(0, x + 1)
    y_values = np.arange(0, y + 1)

    X, Y = np.meshgrid(x_values, y_values)

    # Reshape the meshgrid to get pairs of values
    pairs = np.vstack([X.flatten(), Y.flatten()]).T

    # Convert pairs to tuples and return as a list
    pairwise_tuples = [tuple(pair) for pair in pairs]

    return pairwise_tuples
