import os
import argparse
import yaml
from pathlib import Path
import itertools
from copy import deepcopy
import subprocess
import time

from barygnn.config import Config


def generate_configs(base_config_path, param_grid, output_dir):
    """
    Generate configuration files for grid search.
    
    Args:
        base_config_path: Path to base configuration file
        param_grid: Dictionary of parameter grids
        output_dir: Output directory for configuration files
        
    Returns:
        config_paths: List of paths to generated configuration files
    """
    # Load base configuration
    base_config = Config.from_yaml(base_config_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    # Generate configuration files
    config_paths = []
    
    for i, params in enumerate(param_combinations):
        # Create a copy of the base configuration
        config = deepcopy(base_config)
        
        # Set parameters
        for name, value in zip(param_names, params):
            # Handle nested parameters
            if "." in name:
                parts = name.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(config, name, value)
        
        # Set experiment name
        param_str = "_".join([f"{name.split('.')[-1]}={value}" for name, value in zip(param_names, params)])
        config.experiment_name = f"{base_config.experiment_name}_{param_str}"
        
        # Save configuration
        config_path = output_dir / f"{config.experiment_name}.yaml"
        config.to_yaml(config_path)
        config_paths.append(config_path)
    
    return config_paths


def main():
    """
    Main function.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run experiments")
    parser.add_argument("--base_config", type=str, required=True, help="Path to base configuration file")
    parser.add_argument("--output_dir", type=str, default="configs", help="Output directory for configuration files")
    parser.add_argument("--train_script", type=str, default="barygnn/scripts/train.py", help="Path to training script")
    args = parser.parse_args()
    
    # Define parameter grid
    param_grid = {
        "model.readout_type": ["weighted_mean", "concat"],
        "model.pooling.codebook_size": [8, 16, 32, 64],
        "model.pooling.epsilon": [0.01, 0.05, 0.1],
        "model.encoder.type": ["GIN", "GraphSAGE"],
    }
    
    # Generate configuration files
    config_paths = generate_configs(args.base_config, param_grid, args.output_dir)
    
    # Run experiments
    for config_path in config_paths:
        print(f"Running experiment with config: {config_path}")
        
        # Run training script
        subprocess.run(["python", args.train_script, "--config", str(config_path)])
        
        # Wait a bit to avoid potential issues
        time.sleep(1)


if __name__ == "__main__":
    main() 