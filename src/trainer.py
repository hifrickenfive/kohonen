import numpy as np
from tqdm import tqdm
from src.model import (
    find_bmu_simple,
    update_weights,
    get_neighbourhood_nodes,
)


def training_loop(
    grid: np.ndarray,
    input_matrix: np.ndarray,
    max_iter: int,
    lr: float,
    grid_width: int,
    grid_height: int,
) -> np.ndarray:
    """
    Trains a Kohonen map

    Args:
        grid: grid.
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
    initial_lr = lr
    time_constant = max_iter / np.log(initial_radius)

    all_d_squared = []
    for iter in tqdm(range(max_iter), "Training..."):

        # Enumerated input vector
        vector_idx = iter % input_matrix.shape[0]
        current_vector = input_matrix[vector_idx]

        # Find BMU based on pixel distance
        bmu, d_squared_to_bmu = find_bmu_simple(current_vector, trained_grid)
        all_d_squared.append(d_squared_to_bmu)

        # Find neighbourhood nodes based on spatial distance
        neighbourhood_nodes = get_neighbourhood_nodes(
            bmu, radius, grid_width, grid_height
        )

        # Get weights of nodes and bmu
        x_idx, y_idx = neighbourhood_nodes[:, 0], neighbourhood_nodes[:, 1]
        node_weights = trained_grid[x_idx, y_idx, :]  # (num nodes, dim)
        bmu_weight = trained_grid[bmu[0], bmu[1]]

        # Update neighbourhood weights
        trained_grid[x_idx, y_idx, :] = update_weights(
            node_weights, bmu_weight, lr, radius, current_vector
        )

        # Update learning rate and radius
        radius_tuning_factor = 0.8  # big number = faster decay
        radius = initial_radius * np.exp(-radius_tuning_factor * iter / time_constant)
        lr = initial_lr * np.exp(-iter / time_constant)

    return trained_grid, np.mean(all_d_squared)
