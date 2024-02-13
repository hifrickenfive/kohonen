import numpy as np
import utils


def initialise_grid(grid_width, grid_height, node_weight_range):
    """Initialise the grid with random weight vectors
    Args:
        grid_width (int)
        grid_height (int)
        node_weight_range (tuple of ints)

    Returns:
        grid (dict): keys are the coordinates (tuples) of the grid, values are weight vectors
    """
    # x = width, y = height
    all_nodes = utils.pairwise_permutations_grid(grid_width, grid_height)
    grid = dict()
    for node in all_nodes:
        grid[node] = np.random.random((1, 3))
        # grid[node] = np.random.randint(node_weight_range[0], node_weight_range[1], 3)
    return grid


def find_bmu(grid, input_vector, type="euclidean"):
    """Find the best matching unit (BMU) for a given input vector
    Args:
        grid (dict): keys are the coordinates (tuples) of the grid, values are weight vectors
        input_vector (np.array): input vector

    Returns:
        bmu (tuple): coordinates of the BMU
    """
    min_dist = np.inf
    if type == "euclidean":
        for node in grid.keys():
            dist = np.linalg.norm(grid[node] - input_vector)
            if dist < min_dist:
                min_dist = dist
                bmu = node
    return bmu


def calc_neighbourhood_radius(current_radius, max_iter, current_iter):
    """_summary_

    Args:
        current_radius (float)
        max_iter (int)
        current_iter (int)

    Returns:
        radius (float)
    """
    radius = current_radius * np.exp(-current_iter / max_iter)
    return radius


def get_neighbourhood_nodes(bmu, radius, grid_width, grid_height):
    """_summary_

    Args:
        bmu (tuple): _description_
        neighbourhood_params (_type_): _description_

    Returns:
        candidate_nodes (list): list of nodes (tuples) in the neighbourhood of the BMU
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


def update_lr(current_iter, current_lr, current_radius, max_iter):
    time_constant = max_iter / np.log(current_radius)
    return current_lr * np.exp(-current_iter / time_constant)


def calc_influence(node, bmu, radius):
    dist = np.linalg.norm(np.array(bmu) - np.array(node))
    influence = np.exp(-(dist**2) / (2 * radius**2))
    return influence


def update_node_weights(node_weights, lr, influence, input_vector):
    """_summary_

    Args:
        neighbourhood_nodes (_type_): _description_
        weight_update_params (_type_): _description_
    """
    return node_weights + lr * influence * (input_vector - node_weights)


def train(radius, grid, max_iter, input_matrix, learning_rate, grid_width, grid_height):
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
    np.random.seed(42)

    # Initialise training params
    grid_width = 9  # zero-based indexing
    grid_height = 9  # zero-based indexing
    node_weight_range = (0, 255)
    max_iter = 100  # risky, if 0 then runtime error because radius is 0, and log(0) in update_lr
    grid = initialise_grid(grid_width, grid_height, node_weight_range)
    input_matrix = np.random.rand(20, 3)
    initial_radius = max(grid_width, grid_height) / 2
    learning_rate = 0.1

    utils.plot_pixel_grid(grid, "initial_grid.png")

    # Train
    trained_grid = train(
        initial_radius,
        grid,
        max_iter,
        input_matrix,
        learning_rate,
        grid_width,
        grid_height,
    )

    utils.plot_pixel_grid(trained_grid, "trained_grid.png")
