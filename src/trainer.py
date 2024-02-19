import numpy as np
from tqdm import tqdm
from src.model import (
    find_bmu_vectorised,
    get_neighbourhood_nodes,
    calc_influence,
    calc_d_squared,
)

import matplotlib.pyplot as plt


def training_loop(
    grid: np.ndarray,
    input_matrix: np.ndarray,
    max_iter: int,
    lr: int,
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
    inner_loop_iter = 0

    # "We enumerate through the training data for some number of iterations (repeating if necessary)..."
    # Implies max iter is function of number of input vectors
    # e.g. if max_iter = 100, and num_inputs = 20, then in a batch process we would have 5 iterations

    adj_max_iter_for_batch = round(max_iter / input_matrix.shape[0])

    lr_debug = []
    radius_debug = []
    inner_lr_debug = []
    inner_radius_debug = []
    for iter in tqdm(range(max_iter), "Training..."):
        # Find BMUs in batch. Return BMUs as height, row indices (num_input vectors, 2)
        vector_idx = iter % input_matrix.shape[0]
        current_vector = input_matrix[vector_idx]

        diff = np.abs(grid - current_vector)
        manhattan_dist = np.sum(diff, axis=2)  # (grid_height, grid_width)
        bmu_idx = np.argmin(manhattan_dist)  # (1,) pos in flat array
        bmu = np.unravel_index(bmu_idx, manhattan_dist.shape)  # (height idx, width idx)

        # Update previous radius and learning rate with each each batch iteration
        # radius = initial_radius * np.exp(-inner_loop_iter / time_constant)
        # lr = initial_lr * np.exp(-inner_loop_iter / time_constant)
        # inner_loop_iter += 1
        # lr_debug.append(lr)
        # radius_debug.append(radius)

        radius = initial_radius * np.exp(-iter / time_constant)
        lr = initial_lr * np.exp(-iter / time_constant)
        # inner_lr_debug.append(lr)
        # inner_radius_debug.append(radius)
        # inner_loop_iter += 1

        neighbourhood_nodes = get_neighbourhood_nodes(
            bmu, radius, grid_width, grid_height
        )
        d_squared = calc_d_squared(neighbourhood_nodes, bmu)  # (num nodes, 1)
        influence = calc_influence(d_squared, radius)  # (num nodes, 1)

        # Get variables for updating node weights
        x_idx, y_idx = neighbourhood_nodes[:, 0], neighbourhood_nodes[:, 1]
        node_weights = trained_grid[x_idx, y_idx, :]  # (num nodes, dim)

        # Update node weights in the batch of neighbourhood nodes
        # (num_nodes, dim) + scalar + (num_nodes, 1) * ((3,) - (num_nodes, dim))
        # (num_nodes, dim) + scalar + (num_nodes, 1) * (num_nodes, dim)
        # (num_nodes, dim) + (num_nodes, dim)
        # (num_nodes, dim)
        trained_grid[x_idx, y_idx, :] = node_weights + lr * influence * (
            current_vector - node_weights
        )

    # fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    # ax[0, 0].plot(lr_debug, "o-", color="red")
    # ax[0, 0].set_title("lr: batch")

    # ax[0, 1].plot(inner_lr_debug, "o-", color="darkred")
    # ax[0, 1].set_title("lr: inner loop")

    # ax[1, 0].plot(radius_debug, "o-", color="blue")
    # ax[1, 0].set_title("radius: batch")

    # ax[1, 1].plot(inner_radius_debug, "o-", color="darkblue")
    # ax[1, 1].set_title("radius: inner loop")

    # plt.savefig("debug\\initial_r_lr.png")
    # plt.close()

    return trained_grid, np.mean(manhattan_dist)
