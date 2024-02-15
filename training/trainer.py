import numpy as np
from typing import Dict, Tuple
from model.model import (
    find_bmu,
    calc_neighbourhood_radius,
    get_neighbourhood_nodes,
    calc_influence,
)


def update_lr(
    current_iter: int, current_lr: int, current_radius: float, max_iter: int
) -> float:
    """
    Update the learning rate at a given iteration

    Args:
        current_iter: the current iteration
        current_lr: the current learning rate
        current_radius: the current radius
        max_iter: the maximum number of iterations

    Returns:
        updated_ lr: the updated learning rate
    """
    eps = 1e-9  # prevent divide by zero error
    time_constant = max_iter / np.log(current_radius + eps)
    updated_lr = current_lr * np.exp(-current_iter / time_constant + eps)
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
    radius: float,
    grid: Dict[Tuple[int, int], np.ndarray],
    input_matrix: np.ndarray,
    max_iter: int,
    learning_rate: int,
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
    trained_grid = (
        grid.copy()
    )  # incur memory penalty to show before vs. after. Shallow ok

    all_av_dist_to_bmu = list()
    for current_iter in range(max_iter):

        # print(f"Training iteration {current_iter + 1}/{max_iter}")
        all_dist_to_bmu = list()
        for input_vector in input_matrix:
            bmu, dist_to_bmu = find_bmu(trained_grid, input_vector)
            all_dist_to_bmu.append(dist_to_bmu)
            radius = calc_neighbourhood_radius(radius, max_iter, current_iter)
            neighbourhood_nodes = get_neighbourhood_nodes(
                bmu, radius, grid_width, grid_height
            )
            lr = update_lr(current_iter, learning_rate, radius, max_iter)
            for node in neighbourhood_nodes:
                influence = calc_influence(node, bmu, radius)
                trained_grid[node] = update_node_weights(
                    trained_grid[node], lr, influence, input_vector
                )
        av_dist_to_bmu_iter = np.mean(all_dist_to_bmu)
        all_av_dist_to_bmu.append(av_dist_to_bmu_iter)

    final_av_dist_to_bmu = np.mean(all_dist_to_bmu)
    return trained_grid, final_av_dist_to_bmu
