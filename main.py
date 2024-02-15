import argparse
from datetime import datetime
import numpy as np
import time
from typing import Dict, Tuple
from config.config_helper import load_and_check_config
from data_preprocessing.grid_helper import initialise_grid
from training.trainer import training_loop
from utils.plot_utils import plot_pixel_grid
from utils.log_utils import create_log


def run_main_function(config: dict):
    start_time = time.time()

    # Set random seed
    np.random.seed(config["random_seed"])

    # Set inputs
    grid: Dict[Tuple[int, int], np.ndarray] = initialise_grid(
        config["grid_width"], config["grid_height"]
    )
    initial_radius: float = max(config["grid_width"], config["grid_height"]) / 2
    input_matrix: np.ndarray = np.random.rand(
        config["num_input_vectors"], config["dim_of_input_vector"]
    )

    # Train
    trained_grid, final_av_dist_to_bmu = training_loop(
        initial_radius,
        grid,
        input_matrix,
        config["max_iter"],
        config["learning_rate"],
        config["grid_width"],
        config["grid_height"],
    )

    # Plot results
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename_initial_grid = f"exp/plot_of_initial_grid_{date_time}.png"
    filename_trained_grid = f"exp/plot_of_trained_grid_{date_time}.png"
    fig_initial_grid = plot_pixel_grid(grid, filename_initial_grid, config)
    fig_trained_grid = plot_pixel_grid(trained_grid, filename_trained_grid, config)

    # Log
    end_time = time.time()
    log_message = (
        f"Datetime: {datetime.now()}\n"
        f"Config: {config} \n"
        f"Elapsed time: {(end_time - start_time):.2f} seconds\n"
        f"Final score: {final_av_dist_to_bmu:.3f}. final_av_dist_to_bmu. Lower is better."
    )
    create_log(log_message, "logs\\log.txt")
    return fig_initial_grid, fig_trained_grid, log_message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Kohonen map given a config file and plot before vs. after maps"
    )
    parser.add_argument(
        "config_file",
        type=str,
        nargs="?",
        default="default_config.yaml",
        help="Name of config file in config directory. Default: default_config.yaml",
    )
    args = parser.parse_args()
    config = load_and_check_config(args.config_file)
    __, __, log_message = run_main_function(config)
    print(log_message)
