import argparse
from datetime import datetime
import mlflow
import numpy as np
import time

from src.trainer import training_loop
from utils.config_utils import load_and_check_config
from utils.log_utils import append_to_log_file, create_log_message
from utils.plot_utils import plot_pixel_grid, plot_pixel_inputs

mlflow.set_experiment("Setup logging of an array of metrics")


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
    trained_grid, all_d_squared_to_bmu = training_loop(
        grid,
        input_matrix,
        config["max_iter"],
        config["learning_rate"],
        config["grid_width"],
        config["grid_height"],
        config["radius_tuning_factor"],
        config["influence_tuning_factor"],
    )

    # Plot results
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename_input = f"exp/plot_of_input_{date_time}.png"
    filename_initial_grid = f"exp/plot_of_initial_grid_{date_time}.png"
    filename_trained_grid = f"exp/plot_of_trained_grid_{date_time}.png"
    fig_input = plot_pixel_inputs(input_matrix, filename_input)
    fig_initial_grid = plot_pixel_grid(grid, filename_initial_grid, config)
    fig_trained_grid = plot_pixel_grid(trained_grid, filename_trained_grid, config)

    # Log to txt
    elapsed_time = time.time() - start_time
    log = {
        "Datetime": now,
        "Config": config,
        "Elapsed time": elapsed_time,
        "final_av_dist_to_bmu": np.mean(all_d_squared_to_bmu[-20:]),
    }
    log_message = create_log_message(log)
    append_to_log_file(log_message, "logs\\log.txt")

    # Log to mlflow
    batch_size = config["num_input_vectors"]
    means = [
        np.mean(all_d_squared_to_bmu[i : i + batch_size])
        for i in range(0, len(all_d_squared_to_bmu), batch_size)
    ]
    with mlflow.start_run():
        mlflow.log_params(config)
        for idx, mean_dsq2bmu in enumerate(means):
            mlflow.log_metric("mean_dsq2bmu", mean_dsq2bmu, step=idx + 1)
        mlflow.log_artifact("src\\trainer.py")
        mlflow.log_artifact("src\\model.py")
        mlflow.log_artifact(filename_input)
        mlflow.log_artifact(filename_initial_grid)
        mlflow.log_artifact(filename_trained_grid)

    return fig_input, fig_initial_grid, fig_trained_grid, log


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
