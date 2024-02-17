import jsonschema
import os
import sys
import yaml


def load_and_check_config(config_file: str):
    """
    Load a YAML config file
    Args:
        config_file: config file name with with extension
    Returns:
        config: the loaded config file else exit with error
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_project_dir = os.path.normpath(os.path.join(script_dir, ".."))

    filename_with_ext = os.path.basename(config_file)

    full_config_file_path = os.path.join(
        root_project_dir, "config", filename_with_ext
    )  # to resolve compatibility linux vs. windows

    try:
        with open(full_config_file_path, "r") as f:
            config = yaml.safe_load(f)
        check_config(config)
        return config
    except FileNotFoundError:
        print(f"Error: YAML config file '{full_config_file_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(
            f"Error YAML format incorrect in config file '{full_config_file_path}': {e}"
        )
        sys.exit(1)


def check_config(config):
    schema = {
        "type": "object",
        "properties": {
            "grid_width": {
                "type": "integer",
                "minimum": 2,
                "maximum": 100,
            },
            "grid_height": {
                "type": "integer",
                "minimum": 2,
                "maximum": 100,
            },
            "max_iter": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
            },
            "learning_rate": {
                "type": "number",
                "minimum": 0,
                "maximum": 1.0,
            },
            "num_input_vectors": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
            },
            "dim_of_input_vector": {
                "type": "integer",
                "minimum": 3,
                "maximum": 3,  # 3 dim required to produce pixel maps
            },
            "random_seed": {
                "type": "integer",
                "minimum": 0,
                "maximum": sys.maxsize,
            },
        },
        "required": [
            "grid_width",
            "grid_height",
            "max_iter",
            "learning_rate",
            "num_input_vectors",
            "dim_of_input_vector",
            "random_seed",
        ],
    }
    jsonschema.validate(instance=config, schema=schema)
