import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple


def plot_pixel_grid(
    pixel_dict: Dict[Tuple[int, int], np.ndarray], filename: str, tick_step=2
):
    """
    Plot a grid of pixels

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

    plt.xticks(np.arange(min_x, max_x + 1, tick_step))
    plt.yticks(np.arange(min_y, max_y + 1, tick_step))

    fig, ax = plt.subplots()
    ax.imshow(pixel_grid)
    fig.savefig(filename)
    plt.close(fig)  # Close the specific figure
