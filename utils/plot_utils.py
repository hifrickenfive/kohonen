import imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np
import os
from typing import List, Tuple


def plot_pixel_inputs(input_vectors: np.ndarray, input_filename: str) -> plt.Figure:
    """
    Plots the vector of pixels that serve as the input to the kohonen map
    Assumes the values in pixel dict are 3 dimensional (RGB)

    Args:
        input_vectors: the input vectors
        input_filename: the filename to save the plot of the input vector

    Returns: plot saved in filename
    """
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
) -> plt.Figure:
    """
    Plot a grid of pixels
    Assumes the values in pixel dict are 3 dimensional (RGB)

    Args:
        pixel_dict: a dictionary of pixel positions and colours
        filename: the filename of the plot to be saved

    Returns: plot saved in filename
    """

    fig, ax = plt.subplots()
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
    plt.close(fig)
    return fig


def plot_bmu_and_neighbours(
    grid: np.ndarray,
    bmu: Tuple[int, int],
    neighbourhood_nodes: np.ndarray,
    influences: np.ndarray,
    d_squared: np.ndarray,
    radius: float,
    iter_num: int,
    bmu_idx: int,
    input_vector: np.ndarray,
    folder: str = "debug",
):
    """Plot the grid at each iteration, show BMU, its neighbours and their
    influence and d_squared values. Show input pixel for context.

    Args:
        grid: the trained grid at the current iteration
        bmu: the best matching unit
        neighbourhood_nodes: the nodes in the neighbourhood of the BMU
        influences: the influence of each node in the neighbourhood
        d_squared: the euclidean distance squared of each node in the neighbourhood
        radius: the radius of the neighbourhood
        iter_num: the current iteration number
        bmu_idx: the index of the BMU in the input vector
        input_vector: the input vector

    Returns: plot saved in folder
    """
    fig, ax = plt.subplots()
    ax.imshow(grid)

    # Add inset input vector
    left_inset = plt.axes([0.01, 0.5, 0.05, 0.05])
    left_inset.imshow(input_vector.reshape((1, 1, 3)))
    left_inset.axis("off")  # Turn off axis for inset
    left_inset.set_title("Input \n pixel", fontsize=8, pad=5)

    bmu_x, bmu_y = bmu[1], bmu[0]

    # Plot the radius around the bmu
    bmu_circle = plt.Circle((bmu_x, bmu_y), radius, color="red", fill=False)
    ax.add_artist(bmu_circle)

    # Mark the BMU node with a thick border rectangle
    rect = patches.Rectangle(
        (bmu_x - 0.5, bmu_y - 0.5), 1, 1, linewidth=2, edgecolor="red", facecolor="none"
    )
    ax.add_patch(rect)

    # Plot neighbour nodes and annotate their influence and d_squared values
    for i, (node_x, node_y) in enumerate(neighbourhood_nodes):
        influence = influences[i][0]
        d_sq = int(d_squared[i][0])
        alpha = influence  # Transparency as a function of influence
        ax.scatter(
            node_y, node_x, color="blue", alpha=alpha, s=10
        )  # s is the size of the marker
        ax.annotate(
            f"{influence:.2f}\n{d_sq}",
            (node_y, node_x),  # (column, row) convention for imshow
            textcoords="offset points",
            xytext=(5, -5),
            ha="center",
            fontsize=8,
        )

    ax.scatter([], [], color="blue", label="Neighbour node", s=10)
    ax.legend()
    ax.set_title(f"BMU: ({bmu_y}, {bmu_x}), Iter: {iter_num}")

    # Save
    os.makedirs(folder, exist_ok=True)
    plt.savefig(
        os.path.join(
            folder,
            f"iter_{iter_num}_bmu_{bmu_idx}_{bmu_y}_{bmu_x}.png",
        )
    )
    plt.close(fig)


def animate_plots(folder_path: str = "debug"):
    """
    Create an mp4 animation given the plots of each iteration in the training run

    Assumptions: the plots are prepended with "iter_"

    Args:
        folder_path: the folder containing the plots

    Returns: .avi saved in folder
    """
    file_names = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.startswith("iter_")
        ],
        key=lambda x: int(
            x.split("_")[1]
        ),  # Sort based on the number following 'iter_'
    )

    with imageio.get_writer(folder_path + "//animation.mp4", fps=0.5) as writer:
        for filename in file_names:
            image = imageio.imread(filename)
            writer.append_data(image)
