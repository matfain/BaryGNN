import os
import argparse
import yaml
from pathlib import Path
import itertools
import subprocess
import time
from copy import deepcopy

from barygnn.config import Config
from barygnn.scripts.run_experiments import generate_configs


def create_slurm_script(config_path, train_script, output_dir, partition="studentrun", time_limit="3:00:00", gpu=True):
    """
    Create a SLURM script for running an experiment.
    
    Args:
        config_path: Path to configuration file
        train_script: Path to training script
        output_dir: Output directory for SLURM scripts
        partition: SLURM partition
        time_limit: Time limit
        gpu: Whether to use GPU
        
    Returns:
        slurm_script_path: Path to generated SLURM script
    """
    # Load configuration
    config = Config.from_yaml(config_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create SLURM script
    slurm_script_path = output_dir / f"{config.experiment_name}.sh"
    
    with open(slurm_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={config.experiment_name}\n")
        f.write(f"#SBATCH --output=logs/{config.experiment_name}.out\n")
        f.write(f"#SBATCH --error=logs/{config.experiment_name}.err\n")
        f.write(f"#SBATCH --time={time_limit}\n")
        f.write(f"#SBATCH --partition={partition}\n")
        f.write("#SBATCH --ntasks=1\n")
        
        if gpu:
            f.write("#SBATCH --gres=gpu:1\n")
        
        f.write("\n")
        f.write("# Activate conda environment\n")
        f.write("source ~/anaconda3/bin/activate barygnn\n")
        f.write("\n")
        f.write("# Run experiment\n")
        f.write(f"python {train_script} --config {config_path}\n")
    
    # Make script executable
    slurm_script_path.chmod(0o755)
    
    return slurm_script_path


def main():
    """
    Main function.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for experiments")
    parser.add_argument("--base_config", type=str, required=True, help="Path to base configuration file")
    parser.add_argument("--config_dir", type=str, default="configs", help="Output directory for configuration files")
    parser.add_argument("--slurm_dir", type=str, default="slurm", help="Output directory for SLURM scripts")
    parser.add_argument("--train_script", type=str, default="barygnn/scripts/train.py", help="Path to training script")
    parser.add_argument("--partition", type=str, default="studentrun", help="SLURM partition")
    parser.add_argument("--time_limit", type=str, default="3:00:00", help="Time limit")
    parser.add_argument("--no_gpu", action="store_true", help="Do not use GPU")
    args = parser.parse_args()
    
    # Define parameter grid
    param_grid = {
        "model.readout_type": ["weighted_mean", "concat"],
        "model.pooling.codebook_size": [8, 16, 32, 64],
        "model.pooling.epsilon": [0.01, 0.05, 0.1],
        "model.encoder.type": ["GIN", "GraphSAGE"],
    }
    
    # Generate configuration files
    config_paths = generate_configs(args.base_config, param_grid, args.config_dir)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Create SLURM scripts and submit jobs
    for config_path in config_paths:
        # Create SLURM script
        slurm_script_path = create_slurm_script(
            config_path,
            args.train_script,
            args.slurm_dir,
            args.partition,
            args.time_limit,
            not args.no_gpu,
        )
        
        # Submit job
        print(f"Submitting job for config: {config_path}")
        subprocess.run(["sbatch", str(slurm_script_path)])
        
        # Wait a bit to avoid potential issues
        time.sleep(1)


if __name__ == "__main__":
    main() 