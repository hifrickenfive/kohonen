import numpy as np
from tqdm import tqdm
from src.model import (
    find_bmu_vectorised,
    get_neighbourhood_nodes,
    calc_influence,
    calc_d_squared,
)


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
    time_constant = max_iter / np.log(initial_radius)
    inner_loop_iter = 0

    # "We enumerate through the training data for some number of iterations (repeating if necessary)..."
    # Implies max iter is function of number of input vectors
    # e.g. if max_iter = 100, and num_inputs = 20, then in a batch process we would have 5 iterations

    adj_max_iter_for_batch = round(max_iter / input_matrix.shape[0])

    lr_debug = []
    radius_debug = []
    for batch_iter in tqdm(range(adj_max_iter_for_batch), "Training..."):
        # Find BMUs in batch. Return BMUs as height, row indices (num_input vectors, 2)
        bmus, min_sum_squared_diff = find_bmu_vectorised(input_matrix, trained_grid)

        # Update previous radius and learning rate with each each batch iteration
        radius = radius * np.exp(-inner_loop_iter / time_constant)
        lr = lr * np.exp(-inner_loop_iter / time_constant)
        inner_loop_iter += 1
        lr_debug.append(lr)
        radius_debug.append(radius)

        for idx_input_vector, bmu in enumerate(bmus):
            neighbourhood_nodes = get_neighbourhood_nodes(
                bmu, radius, grid_width, grid_height
            )
            d_squared = calc_d_squared(neighbourhood_nodes, bmu)  # (num nodes, 1)
            influence = calc_influence(d_squared, radius)  # (num nodes, 1)

            # Get variables for updating node weights
            x_idx, y_idx = neighbourhood_nodes[:, 0], neighbourhood_nodes[:, 1]
            node_weights = trained_grid[x_idx, y_idx, :]  # (num nodes, dim)
            current_vector = input_matrix[idx_input_vector]  # (dim,)

            # Update node weights in the batch of neighbourhood nodes
            trained_grid[x_idx, y_idx, :] = node_weights + lr * influence * (
                current_vector - node_weights
            )

    return trained_grid, np.sqrt(np.mean(min_sum_squared_diff))
