import numpy as np
import matplotlib.pyplot as plt


def pairwise_permutations_grid(x, y):
    # Create an array of values in the range [-r, +r]
    x_values = np.arange(0, x + 1)
    y_values = np.arange(0, y + 1)

    # Create a meshgrid of all possible combinations of values
    X, Y = np.meshgrid(x_values, y_values)

    # Reshape the meshgrid to get pairs of values
    pairs = np.vstack([X.flatten(), Y.flatten()]).T

    # Convert pairs to tuples and return as a list
    pairwise_tuples = [tuple(pair) for pair in pairs]

    return pairwise_tuples


def pairwise_permutations_square(r):
    # Create an array of values in the range [-r, +r]
    values = np.arange(-r, r + 1)

    # Create a meshgrid of all possible combinations of values
    X, Y = np.meshgrid(values, values)

    # Reshape the meshgrid to get pairs of values
    pairs = np.vstack([X.flatten(), Y.flatten()]).T

    # Convert pairs to tuples and return as a list
    pairwise_tuples = [tuple(pair) for pair in pairs]

    return pairwise_tuples


def plot_pixel_grid(pixel_dict, filename):
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
        pixel_grid[y, x] = color * 255  # Set colour

    # Set ticks as integers
    min_x = min(key[0] for key in pixel_dict.keys())
    min_y = min(key[1] for key in pixel_dict.keys())

    tick_step = 2
    plt.xticks(np.arange(min_x, max_x + 1, tick_step))
    plt.yticks(np.arange(min_y, max_y + 1, tick_step))

    # Display the pixel grid
    plt.imshow(pixel_grid)
    plt.savefig(
        filename
    )  # https://stackoverflow.com/questions/9012487/savefig-outputs-blank-image
    plt.show()
