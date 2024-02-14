import numpy as np
from typing import List, Dict, Tuple
import utils
import yaml


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
    all_nodes = utils.pairwise_permutations_grid(grid_width, grid_height)
    grid = dict()
    for node in all_nodes:
        grid[node] = np.random.random((1, 3))
    return grid


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
    delta_nodes = utils.pairwise_permutations_square(radius_rounded)
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
    time_constant = max_iter / np.log(current_radius)
    updated_lr = current_lr * np.exp(-current_iter / time_constant)
    return updated_lr


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


def train(
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
        grid: A dictionary mapping grid coordinates (tuple of ints) to node weights (numpy arrays).
        input_matrix: An array of input vectors for training.
        max_iter: The maximum number of iterations to perform.
        learning_rate: The initial learning rate.
        grid_width: The width of the grid.
        grid_height: The height of the grid.

    Returns:
        The trained grid as a dictionary with the same structure as the input grid.
    """
    # Initialise
    trained_grid = grid

    for current_iter in range(max_iter):

        print(f"Training iteration {current_iter + 1}/{max_iter}")

        for input_vector in input_matrix:
            bmu = find_bmu(trained_grid, input_vector)
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

    return trained_grid


if __name__ == "__main__":
    # Set random seed
    np.random.seed(40)

    # Get params
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    grid_width = int(config["grid_width"])
    grid_height = int(config["grid_height"])
    num_input_vectors = int(config["num_input_vectors"])
    dim_of_input_vector = int(config["dim_of_input_vector"])
    max_iter = int(config["max_iter"])
    lr = float(config["learning_rate"])

    # Setup training inputs
    grid: Dict[Tuple[int, int], np.ndarray] = initialise_grid(grid_width, grid_height)
    initial_radius: float = max(grid_width, grid_height) / 2
    input_matrix: np.ndarray = np.random.rand(num_input_vectors, dim_of_input_vector)

    # Plot before
    utils.plot_pixel_grid(grid, "plot_of_initial_grid.png")

    # Train
    trained_grid = train(
        initial_radius,
        grid,
        input_matrix,
        max_iter,
        lr,
        grid_width,
        grid_height,
    )

    # Plot after
    utils.plot_pixel_grid(trained_grid, "plot_of_trained_grid.png")
