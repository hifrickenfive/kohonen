import numpy as np
from tqdm import tqdm
from model.model import (
    find_bmu_vectorised,
    get_neighbourhood_nodes,
    calc_influence,
    calc_d_squared,
)


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
    initial_lr = lr
    initial_radius = radius
    time_constant = max_iter / np.log(initial_radius)

    # we enumerate through the training data for some number of iterations (repeating if necessary)...
    # implies max iter is function of number of input vectors
    # e.g. if max_iter = 100, and num_inputs = 20, then in a batch process we would have 5 iterations

    adj_max_iter_for_batch = round(max_iter / input_matrix.shape[0])

    for current_iter in tqdm(range(adj_max_iter_for_batch), "Training..."):
        bmus, min_sum_squared_diff = find_bmu_vectorised(input_matrix, trained_grid)

        radius = radius * np.exp(-current_iter / time_constant)
        lr = lr * np.exp(-current_iter / time_constant)

        for idx_input_vector, bmu in enumerate(bmus):
            neighbourhood_nodes = get_neighbourhood_nodes(
                bmu, radius, grid_width, grid_height
            )
            d_squared = calc_d_squared(neighbourhood_nodes, bmu.reshape(1, 2))
            for idx_node, node in enumerate(neighbourhood_nodes):
                node_height_idx, node_width_idx = node
                influence = calc_influence(d_squared[idx_node], radius)

                current_weight = trained_grid[node_height_idx, node_width_idx, :]
                trained_grid[node_height_idx, node_width_idx, :] = update_node_weights(
                    current_weight, lr, influence, input_matrix[idx_input_vector]
                )

    return trained_grid, np.sqrt(np.mean(min_sum_squared_diff))
