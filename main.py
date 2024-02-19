import argparse
from datetime import datetime
import numpy as np
import time
from utils.config_utils import load_and_check_config
from src.trainer import training_loop
from utils.plot_utils import plot_pixel_grid, plot_pixel_inputs
from utils.log_utils import append_to_log_file, create_log_message


def run_main_function(config: dict, input_matrix=None):
    start_time = time.time()

    # Set random seed
    np.random.seed(config["random_seed"])

    # Set inputs
    grid = np.random.rand(
        config["grid_height"], config["grid_width"], config["dim_of_input_vector"]
    )
    if input_matrix is None:
        input_matrix = np.random.rand(
            config["num_input_vectors"], config["dim_of_input_vector"]
        )

    # Train
    trained_grid, final_av_dist_to_bmu = training_loop(
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
    filename_input = f"exp/plot_of_input_{date_time}.png"
    filename_initial_grid = f"exp/plot_of_initial_grid_{date_time}.png"
    filename_trained_grid = f"exp/plot_of_trained_grid_{date_time}.png"
    # fig_input = plot_pixel_inputs(input_matrix, filename_input)
    # fig_initial_grid = plot_pixel_grid(grid, filename_initial_grid, config)
    fig_trained_grid = plot_pixel_grid(trained_grid, filename_trained_grid, config)

    # Log
    elapsed_time = time.time() - start_time
    log = {
        "Datetime": now,
        "Config": config,
        "Elapsed time": elapsed_time,
        "final_av_dist_to_bmu": final_av_dist_to_bmu,
    }
    log_message = create_log_message(log)
    append_to_log_file(log_message, "logs\\log.txt")
    # return fig_input, fig_initial_grid, fig_trained_grid, log
    return None, None, fig_trained_grid, log


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
    __, __, __, log = run_main_function(config)
    print(create_log_message(log))
