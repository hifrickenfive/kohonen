import imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np
import os


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


def plot_bmu_and_neighbours(
    grid,
    bmu,
    neighbourhood_nodes,
    influences,
    d_squared,
    radius,
    iter_num,
    bmu_idx,
    input_vector,
):
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
    folder = "debug"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(
        os.path.join(
            folder,
            f"iter_{iter_num}_bmu_{bmu_idx}_{bmu_y}_{bmu_x}.png",
        )
    )
    plt.close(fig)

    title = f"BMU: ({bmu_y}, {bmu_x}), Iter: {iter_num}"
    return fig, title


def animate_plots(folder_path="debug"):
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
