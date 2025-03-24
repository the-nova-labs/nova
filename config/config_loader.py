import yaml
import os

def load_config(path: str = "config/config.yaml"):
    """
    Loads configuration from a YAML file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find config file at '{path}'")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load configuration options
    num_targets = config["protein_selection"]["num_targets"]
    num_antitargets = config["protein_selection"]["num_antitargets"]
    
    validation_config = config["molecule_validation"]
    target_weight = validation_config["target_weight"]
    antitarget_weight = validation_config["antitarget_weight"]
    min_heavy_atoms = validation_config["min_heavy_atoms"]

    return {
        'num_targets': num_targets,
        'num_antitargets': num_antitargets,
        'target_weight': target_weight,
        'antitarget_weight': antitarget_weight,
        'min_heavy_atoms': min_heavy_atoms,
    }