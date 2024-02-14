import yaml
import sys
import jsonschema


def load_YAML_config(path_to_config_file: str):
    """
    Load a YAML config file
    Args:
        config_file: path to the config file
    Returns:
        config: the loaded config file else exit with error
    """
    try:
        with open(path_to_config_file, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: YAML config file '{path_to_config_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(
            f"Error YAML format incorrect in config file '{path_to_config_file}': {e}"
        )
        sys.exit(1)


def check_expected_config_fields(config):
    """
    Check if expected parameters are in the YAML config file

    Args:
        config (loaded YAML file)
    """
    expected_params = [
        "grid_width",
        "grid_height",
        "max_iter",
        "learning_rate",
        "num_input_vectors",
        "dim_of_input_vector",
        "random_seed",
    ]
    for param in expected_params:
        if param not in config:
            print(f"Error: '{param}' missing from config file.")
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
                "maximum": 1,
            },
            "num_input_vectors": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
            },
            "dim_of_input_vector": {
                "type": "integer",
                "minimum": 1,
                "maximum": 3,
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