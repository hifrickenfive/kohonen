import numpy as np
from tqdm import tqdm
from src.model import (
    find_bmu_vectorised,
    get_neighbourhood_nodes,
    calc_influence,
    calc_d_squared,
)

# from utils.plot_utils import plot_kohonen_iteration
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches


def plot_kohonen_iteration(
    trained_grid,
    input_matrix,
    bmu,
    radius,
    neighbourhood_nodes,
    influence,
    vector_idx,
    iter,
    learning_rate,
    current_weights,
    updated_weights,
):
    """
    Plot the current state of the Kohonen map training iteration with various overlays and annotations.
    """
    figsize = (8, 8)

    fig, ax = plt.subplots(figsize=figsize)

    padding = 5
    ax.set_xlim([-padding, trained_grid.shape[1] + padding])
    ax.set_ylim([trained_grid.shape[0] + padding, -padding])

    ax.imshow(trained_grid, aspect="auto")

    # Overlay BMU
    ax.scatter(
        bmu[1], bmu[0], s=100, c="white", marker="o", edgecolors="black", linewidths=1
    )

    # Draw radius circle around BMU
    radius_circle = Circle(
        (bmu[1], bmu[0]), radius, color="red", fill=False, clip_on=True
    )
    ax.add_patch(radius_circle)

    # # Overlay and annotate each neighbor node with influence value
    # for (x, y), infl in zip(neighbourhood_nodes, influence):
    #     node_influence = infl * 100  # scale for size
    #     ax.scatter(y, x, s=node_influence, c="orange", alpha=0.6)
    #     infl_value = np.squeeze(infl)
    #     ax.annotate(
    #         f"{infl_value:.2f}",
    #         (y, x),
    #         textcoords="offset points",
    #         xytext=(5, 5),
    #         ha="center",
    #         color="black",
    #     )

    weight_change = np.linalg.norm(updated_weights - current_weights, axis=1)

    # Calculate weight change for visualization
    weight_change = np.linalg.norm(updated_weights - current_weights, axis=1)

    # Overlay and annotate each neighbor node with its weight change value
    for idx, (x, y) in enumerate(neighbourhood_nodes):
        # Annotate with the magnitude of weight change
        ax.text(
            y,
            x,
            f"{weight_change[idx]:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color="black",
        )

    # Show input matrix as vertical bar of pixels
    input_bar_height = min(len(input_matrix), trained_grid.shape[0])
    for idx, vec in enumerate(input_matrix[:input_bar_height]):
        ax.add_patch(
            plt.Rectangle(
                (trained_grid.shape[1], idx),
                1,
                1,
                color=vec,
                transform=ax.transData,
                clip_on=True,
            )
        )

    # Annotate the overall plot with the current radius, learning rate, and the current input vector's weight
    plot_info = f"Radius: {radius:.2f}, LR: {learning_rate:.4f}, Input Vec: {np.array2string(input_matrix[vector_idx])}"
    ax.text(
        0.5,
        -0.1,
        plot_info,
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Mark current input pixel so I can check BMU
    ax.add_patch(
        plt.Rectangle(
            (trained_grid.shape[1], vector_idx),
            1,
            1,
            color=input_matrix[vector_idx],
            ec="black",
            lw=2,
            transform=ax.transData,
            clip_on=True,
        )
    )

    plt.tight_layout()  # Adjust layout to avoid clipping of annotation

    ax.set_xlim([0, trained_grid.shape[1] + 1])
    ax.set_ylim([trained_grid.shape[0], 0])
    ax.axis("off")

    filename = f"debug/iter_{iter}_vectorIdx_{vector_idx}.png"
    plt.savefig(filename, dpi=96)
    plt.close(fig)

    return filename


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

    for iter in tqdm(range(max_iter), "Training..."):
        vector_idx = iter % input_matrix.shape[0]
        current_vector = input_matrix[vector_idx]

        d_squared = np.sum((grid - current_vector) ** 2, axis=2)  # sum colour pixel dim
        _bmu_idx = np.argmin(d_squared)
        bmu = np.unravel_index(_bmu_idx, d_squared.shape)

        radius = initial_radius * np.exp(-iter / time_constant)
        lr = initial_lr * np.exp(-iter / time_constant)

        neighbourhood_nodes = get_neighbourhood_nodes(
            bmu, radius, grid_width, grid_height
        )

        # Update neighbourhood node weights
        x_idx, y_idx = neighbourhood_nodes[:, 0], neighbourhood_nodes[:, 1]
        node_weights = trained_grid[x_idx, y_idx, :]  # (num nodes, dim)
        bmu_weight = trained_grid[bmu[0], bmu[1]]

        d_squared = np.sum((node_weights - bmu_weight) ** 2, axis=-1)
        # influence = calc_influence(d_squared, radius)  # (num nodes, 1)
        influence = np.exp(-d_squared / (2 * radius**2))  # (num nodes,)
        influence = influence.reshape(-1, 1)  # (num nodes,1)

        # Update node weights in the batch of neighbourhood nodes
        trained_grid[x_idx, y_idx, :] = node_weights + lr * influence * (
            current_vector - node_weights
        )

        # plot_kohonen_iteration(
        #     trained_grid,
        #     input_matrix,
        #     bmu,
        #     radius,
        #     neighbourhood_nodes,
        #     influence,
        #     vector_idx,
        #     iter,
        #     lr,
        #     node_weights,
        #     trained_grid[x_idx, y_idx, :],
        # )

    return trained_grid, np.mean(np.sqrt(d_squared))
