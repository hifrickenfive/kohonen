import cv2
import numpy as np
from typing import List, Tuple


def update_weights(
    node_weights: np.ndarray,
    bmu_weights: np.ndarray,
    lr: float,
    radius: float,
    current_vector_weights: np.ndarray,
    influence_tuning_factor: float,
) -> np.ndarray:
    """
    Update the weights of the nodes in the neighbourhood of the BMU

    Args:
        node_weights: the weights of the nodes in the neighbourhood of the BMU
        bmu_weight: the weight of the BMU
        lr: learning rate
        radius: the radius of the neighbourhood
        current_vector: current input vector

    Returns:
        updated_node_weights: the updated weights of the nodes in the neighbourhood of the BMU
    """
    # Find spatial distance between between nodes position and bmu position
    d_squared = calc_d_squared(node_weights, bmu_weights)
    influence = calc_influence(
        d_squared, radius, influence_tuning_factor
    )  # (num nodes, 1)
    updated_weights = node_weights + lr * influence * (
        current_vector_weights - node_weights
    )
    return updated_weights


def find_bmu_simple(
    current_vector: np.ndarray, grid: np.ndarray
) -> Tuple[Tuple, float]:
    """Find BMU based on pixel distance

    Args:
        current_vector: current input vector.
        grid: pixel grid.

    Returns:
        bmu: the best matching unit.
        d_squared: euclidean distance squared.
    """
    d_squared = np.sum((grid - current_vector) ** 2, axis=2)  # sum in pixel dim
    _bmu_idx = np.argmin(d_squared)  # returns idx in flat array convention, row mjr
    bmu = np.unravel_index(_bmu_idx, d_squared.shape)  # tuple
    return bmu, d_squared[bmu]


def get_neighbourhood_nodes(
    bmu: np.ndarray, radius: float, grid_width: int, grid_height: int
) -> List[np.ndarray]:
    """
    Get the nodes in the neighbourhood of the BMU given a radius

    Args:
        bmu: coordinates of the BMU.
        radius: the radius of the neighbourhood.
        grid_width: the width of the grid.
        grid_height: the height of the grid.

    Returns:
        neighbourhood_nodes: list of nodes in the neighbourhood of the BMU
    """
    # Get all nodes within a square of side length 2*radius then prune nodes outside radius
    radius_rounded = int(np.floor(radius))  # int faster than float ops

    # 1. Create 2D array of x and y deltas in that square
    # delta_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    # delta_y = [[-1, -1, -1], [-0, 0, 0], [1, 1, 1]]
    delta_x, delta_y = np.meshgrid(
        np.arange(-radius_rounded, radius_rounded + 1),
        np.arange(-radius_rounded, radius_rounded + 1),
    )

    # 2. Flatten each 2d array with .ravel() and stack together to form 2 columns of x,y pairs
    # x -> [-1, 0, 1, -1, 0, 1, -1, 0, 1]
    # y -> [-1, -1, -1, 0, 0, 0, 1, 1, 1]
    delta_nodes = np.column_stack((delta_x.ravel(), delta_y.ravel()))

    # 3. Remove bmu (0,0) by scanning across the rows, i.e. along columns
    delta_nodes = delta_nodes[~np.all(delta_nodes == 0, axis=1)]

    candidate_nodes = np.array(bmu) + delta_nodes

    # 4. Prune nodes beyond grid limits (x,y) where x is height, y is width
    valid_nodes = (
        (candidate_nodes[:, 0] >= 0)
        & (candidate_nodes[:, 0] < grid_height)  # check column 0 i.e. (height)
        & (candidate_nodes[:, 1] >= 0)
        & (candidate_nodes[:, 1] < grid_width)  # check column 1 i.e. y (width)
    )
    pruned_nodes = candidate_nodes[valid_nodes]
    distances_sq = np.sum((pruned_nodes - np.array(bmu)) ** 2, axis=1)
    within_radius = distances_sq <= radius**2

    return pruned_nodes[within_radius]


def calc_influence(d_squared: float, radius: float, influence_tuning_factor=1) -> float:
    """Calculate the influence of a node based on its distance from the BMU

    Args:
        d_squared: euclidean distance squared
        radius: radius of the neighbourhood
        influence_tuning_factor: larger value = faster decay

    Returns:
        influence: the influence of the node
    """
    return np.exp(-influence_tuning_factor * d_squared / (2 * radius**2))


def calc_d_squared(neighbourhood_nodes: np.ndarray, bmu: tuple):
    """Calculate the squared euclidean distance between the BMU and the neighbourhood nodes

    Args:
        neighbourhood_nodes: the nodes in the neighbourhood of the BMU
        bmu: the best matching unit

    Returns:
        d_squared: the squared euclidean distance between the BMU and the neighbourhood nodes
    """
    d_squared = np.sum(
        (neighbourhood_nodes - bmu) ** 2,
        axis=-1,
        keepdims=True,
    )  # -1 to retain (n,1) shape else (n,)
    return d_squared


def calc_metric_av_gradient_mag(image_path: str) -> float:
    """
    Create a metric to evaluate the average gradient magnitude of an image

    credit: https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/

    Args:
        image_path: str, path to image
    """
    image = cv2.imread(image_path)

    # Find image gradient in x and y direction
    gX = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=3)  # img shape
    gY = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=3)

    # Eval gradient magnitude
    gradient_magnitude = np.sqrt(gX**2 + gY**2)
    avg_gradient_magnitude = np.mean(gradient_magnitude)

    return avg_gradient_magnitude
