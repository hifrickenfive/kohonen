import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple
from model.model import (
    find_bmu_vectorised,
    calc_neighbourhood_radius,
    get_neighbourhood_nodes,
    calc_influence,
)


def update_lr(current_iter: int, initial_lr: int, time_constant: float) -> float:
    """
    Update the learning rate at a given iteration

    Args:
        current_iter: the current iteration
        initial_lr: the initial learning rate
        time_constant: the time constant

    Returns:
        updated_ lr: the updated learning rate
    """
    updated_lr = initial_lr * np.exp(-current_iter / time_constant)
    return updated_lr


def update_node_weights(
    node_weights: np.ndarray, lr: float, influence: float, input_vector: np.ndarray
) -> np.ndarray:
    """_summary_

    Args:
        node_weights: the weights of the node
        lr: the learning rate
        influence: the influence of the BMU on the node
        input_vector: the input vector

    Returns:
        updated_node_weights: the updated weights of the node
    """
    updated_node_weights = node_weights + lr * influence * (input_vector - node_weights)
    return updated_node_weights


def training_loop(
    grid: Dict[Tuple[int, int], np.ndarray],
    input_matrix: np.ndarray,
    max_iter: int,
    initial_lr: int,
    grid_width: int,
    grid_height: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Trains a Kohonen map

    Args:
        radius: The initial neighborhood radius.
        grid: A dict mapping grid coordinates to node weights.
        input_matrix: An array of input vectors for training.
        max_iter: The maximum number of iterations to perform.
        learning_rate: The initial learning rate.
        grid_width: The width of the grid.
        grid_height: The height of the grid.

    Returns:
        The trained grid with the same structure as the input grid.
    """
    # Initialise
    trained_grid = grid.copy()
    radius = max(grid_width, grid_height) / 2
    initial_radius = radius
    time_constant = max_iter / np.log(initial_radius / 2)

    for current_iter in tqdm(range(max_iter), "Training..."):
        bmus, min_sum_squared_diff = find_bmu_vectorised(input_matrix, trained_grid)
        radius = calc_neighbourhood_radius(radius, current_iter, time_constant)
        lr = update_lr(current_iter, initial_lr, time_constant)

        for idx_input_vector, bmu in enumerate(bmus):
            neighbourhood_nodes = get_neighbourhood_nodes(
                bmu, radius, grid_width, grid_height
            )
            for idx_node, node in enumerate(neighbourhood_nodes):
                node_height, node_width = node
                influence = calc_influence(node, bmu, radius)
                trained_grid[node_height, node_width, :] = update_node_weights(
                    trained_grid[node_height, node_width, :],
                    lr,
                    influence,
                    input_matrix[idx_input_vector, :],
                )
    return trained_grid, np.sqrt(np.mean(min_sum_squared_diff))
