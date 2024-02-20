import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_pixel_inputs(input_vectors: np.ndarray, input_filename: str) -> plt.Figure:
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
