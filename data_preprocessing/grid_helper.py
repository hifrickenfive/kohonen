import numpy as np
from typing import List, Dict, Tuple
import utils


def initialise_grid(
    grid_width: int, grid_height: int
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Initialise the grid with random weight vectors

    Args:
        grid_width: The width of the grid.
        grid_height: The height of the grid.

    Returns:
        grid (dict): keys are the coordinates (tuples) of the grid, values are weight vectors
    """
    all_nodes = pairwise_permutations_grid(grid_width, grid_height)
    grid = dict()
    for node in all_nodes:
        grid[node] = np.random.random((1, 3))
    return grid


def pairwise_permutations_grid(x, y):
    # Create an array of values in the range [-r, +r]
    x_values = np.arange(0, x + 1)
    y_values = np.arange(0, y + 1)

    # Create a meshgrid of all possible combinations of values
    X, Y = np.meshgrid(x_values, y_values)

    # Reshape the meshgrid to get pairs of values
    pairs = np.vstack([X.flatten(), Y.flatten()]).T

    # Convert pairs to tuples and return as a list
    pairwise_tuples = [tuple(pair) for pair in pairs]

    return pairwise_tuples


def pairwise_permutations_square(r):
    # Create an array of values in the range [-r, +r]
    values = np.arange(-r, r + 1)

    # Create a meshgrid of all possible combinations of values
    X, Y = np.meshgrid(values, values)

    # Reshape the meshgrid to get pairs of values
    pairs = np.vstack([X.flatten(), Y.flatten()]).T

    # Convert pairs to tuples and return as a list
    pairwise_tuples = [tuple(pair) for pair in pairs]

    return pairwise_tuples
