import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle


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
