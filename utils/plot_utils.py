import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle


def plot_kohonen_iteration(
    trained_grid,
    input_matrix,
    bmu,
    radius,
    neighbourhood_nodes,
    influence,
    vector_idx,
    iter,
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

    # Overlay and annotate each neighbor node with influence value
    for (x, y), infl in zip(neighbourhood_nodes, influence):
        node_influence = infl * 100  # scale for size
        ax.scatter(y, x, s=node_influence, c="orange", alpha=0.6)
        infl_value = np.squeeze(infl)
        ax.annotate(
            f"{infl_value:.2f}",
            (y, x),
            textcoords="offset points",
            xytext=(5, 5),
            ha="center",
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

    ax.set_xlim([0, trained_grid.shape[1] + 1])
    ax.set_ylim([trained_grid.shape[0], 0])
    ax.axis("off")

    filename = f"debug/iter_{iter}_vectorIdx_{vector_idx}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)

    return filename


def plot_pixel_inputs(input_vectors, input_filename):
    num_pixels = input_vectors.shape[0]
    num_channels = input_vectors.shape[1]

    # Reshape so matplotlib treats the input vector as pixel colours
    input_vector_reshaped = input_vectors.reshape(1, num_pixels, num_channels)

    fig, ax = plt.subplots(figsize=(num_pixels, 1))
    ax.imshow(input_vector_reshaped, aspect="auto")

    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(input_filename)
    plt.close(fig)

    return fig


def plot_pixel_grid(
    pixel_grid: np.ndarray,
    filename: str,
    config: dict,
    tick_step=2,
):
    """
    Plot a grid of pixels
    Assumes the values in pixel dict are 3 dimensional (RGB)

    Args:
        pixel_dict: a dictionary of pixel positions and colours
        filename: the filename of the plot to be saved
    """
    height, width = pixel_grid.shape[0], pixel_grid.shape[1]

    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(0, width + 1, tick_step))
    ax.set_yticks(np.arange(0, height + 1, tick_step))

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))

    # Add annotation in bottom left corner of config values
    params_text = "\n".join(f"{key}: {value}" for key, value in config.items())
    ax.text(
        0.05,
        0.05,
        params_text,
        transform=ax.transAxes,  # set position in axis coordinates i.e. (0,0) bottom left
        fontsize=8,
        verticalalignment="bottom",
    )

    ax.imshow(pixel_grid)
    fig.savefig(filename)
    plt.close(fig)  # Close the specific figure

    return fig


def test_plot(pixel_grid):
    plt.imshow(pixel_grid)
