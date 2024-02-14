import argparse
import numpy as np
import os
from typing import Dict, Tuple
from data_preprocessing.grid_helper import initialise_grid
from training.trainer import training_loop
from utils.plot_utils import plot_pixel_grid
from config.config_helper import load_YAML_config, check_config


def main(config_file: str, plot: bool = False):

    # Get params
    config = load_YAML_config(config_file)
    check_config(config)

    np.random.seed(int(config["random_seed"]))
    grid_width = int(config["grid_width"])
    grid_height = int(config["grid_height"])
    num_input_vectors = int(config["num_input_vectors"])
    dim_of_input_vector = int(config["dim_of_input_vector"])
    max_iter = int(config["max_iter"])
    lr = float(config["learning_rate"])

    # Setup training inputs
    grid: Dict[Tuple[int, int], np.ndarray] = initialise_grid(grid_width, grid_height)
    initial_radius: float = max(grid_width, grid_height) / 2
    input_matrix: np.ndarray = np.random.rand(num_input_vectors, dim_of_input_vector)

    # Train
    trained_grid = training_loop(
        initial_radius,
        grid,
        input_matrix,
        max_iter,
        lr,
        grid_width,
        grid_height,
    )

    if plot:
        config_filename = os.path.splitext(os.path.basename(config_file))[0]
        filename_initial_grid = f"exp/plot_of_initial_grid_{config_filename}.png"
        filename_trained_grid = f"exp/plot_of_trained_grid_{config_filename}.png"
        plot_pixel_grid(grid, filename_initial_grid)
        plot_pixel_grid(trained_grid, filename_trained_grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Kohonen map given a config file and plot before vs. after maps"
    )
    parser.add_argument(
        "config_file",
        type=str,
        nargs="?",
        default="config\\default_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--plot",
        "-p",
        action="store_true",
        help="Plot initialised map vs. trained pixel map",
    )

    args = parser.parse_args()
    main(args.config_file, args.plot)
