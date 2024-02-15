import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def plot_pixel_inputs(input_vector, input_filename):
    # Assuming input_vector is a (20, 3) array for 20 pixels with RGB values
    num_pixels = input_vector.shape[0]  # 20 pixels
    num_channels = input_vector.shape[1]  # 3 channels (RGB)

    # Reshape input_vector to a (1, 20, 3) array for plotting
    # This arranges the pixels in a single row with 20 columns
    input_vector_reshaped = input_vector.reshape(1, num_pixels, num_channels)

    # Create a figure with adjusted dimensions to ensure each pixel is displayed as a square
    # The width of the figure is adjusted to ensure each of the 20 pixels can be displayed as a square
    # A higher width-to-height ratio is used to accommodate all pixels horizontally
    fig, ax = plt.subplots(
        figsize=(20, 1)
    )  # Adjusted figure size for horizontal layout
    ax.imshow(
        input_vector_reshaped, aspect="auto"
    )  # 'auto' lets matplotlib adjust pixels to fill the space

    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks

    plt.savefig(input_filename)
    plt.close(fig)

    return fig


def plot_pixel_grid(
    pixel_dict: Dict[Tuple[int, int], np.ndarray],
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
    # Extract grid size
    max_x = max(key[0] for key in pixel_dict.keys())
    max_y = max(key[1] for key in pixel_dict.keys())
    width = max_x + 1
    height = max_y + 1

    # Create an empty array to store pixel colors
    pixel_grid = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill in pixel colors
    for position, color in pixel_dict.items():
        x, y = position
        pixel_grid[y, x] = color * 255  # Set colour, y is rows, x is column convention

    # Set ticks as integers
    min_x = min(key[0] for key in pixel_dict.keys())
    min_y = min(key[1] for key in pixel_dict.keys())

    fig, ax = plt.subplots()

    # Set the ticks using the axis object
    ax.set_xticks(np.arange(min_x, max_x + 1, tick_step))
    ax.set_yticks(np.arange(min_y, max_y + 1, tick_step))

    # Add annotation in bottom left corner of config values
    params_text = "\n".join(f"{key}: {value}" for key, value in config.items())
    ax.text(
        0.05,
        0.05,
        params_text,
        transform=ax.transAxes,  # set position in axis coordinates i.e. (0,0) bottom left
        fontsize=8,
        verticalalignment="bottom",
        # bbox=dict(facecolor="white", alpha=0.2),  # lower alpha = more transparency
    )

    ax.imshow(pixel_grid)
    fig.savefig(filename)
    plt.close(fig)  # Close the specific figure

    return fig
