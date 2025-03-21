import yaml
import os

def load_protein_selection_params(path: str = "config/config.yaml"):
    """
    Loads num_targets and num_antitargets from a YAML file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find config file at '{path}'")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    num_targets = data["protein_selection"]["num_targets"]
    num_antitargets = data["protein_selection"]["num_antitargets"]

    return num_targets, num_antitargets