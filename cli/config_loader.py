import json

def load_config(config_path):
    """
    Load configuration JSON from the given path.
    """
    with open(config_path, "r") as f:
        return json.load(f)

def load_instructions(instruction_json):
    """
    Load instruction sets JSON from the given path.
    """
    with open(instruction_json, "r") as f:
        return json.load(f)