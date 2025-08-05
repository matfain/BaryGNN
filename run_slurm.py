import os
import argparse
from pathlib import Path
import subprocess
import time
import logging

from barygnn.config import Config, generate_configs



def create_slurm_script(config_path, output_dir):
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
    job_name = f"{config.experiment_type}_{config.data.name}_{config.model.pooling.backend}"
    slurm_script_path = output_dir / f"{job_name}.sh"
    with open(slurm_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={job_name}\n")
        f.write(f"#SBATCH --output=barygnn/logs/{job_name}/logger.out\n")
        f.write(f"#SBATCH --error=barygnn/logs/{job_name}/logger.err\n")
        f.write(f"#SBATCH --time={config.slurm.timelimit}\n")
        f.write(f"#SBATCH --partition={config.slurm.partition}\n")
        f.write(f"#SBATCH --mem={config.slurm.mem}\n")
        f.write(f"#SBATCH --gres=gpu:{config.slurm.gpu}\n")
        f.write("\n")
        f.write("# Activate conda environment\n")
        f.write(f"source /home/yandex/MLWG2025/amitr5/BaryGNN/anaconda3/bin/activate barygnn\n")
        f.write("\n")
        f.write("# Run experiment\n")
        f.write(f"python train.py --config {config_path} --log_dir barygnn/logs/{job_name}\n")
        f.write(f"chmod -R g+rwx /home/yandex/MLWG2025/amitr5/BaryGNN/barygnn/logs/{job_name}\n")
 
    return slurm_script_path


def submit_job(config_path, slurm_script_path):
        print(f"Submitting job for config: {config_path}")
        subprocess.run(["sbatch", str(slurm_script_path)])
        
        # Wait a bit to avoid potential issues
        time.sleep(1)
        
        
def main():
    """
    Main function.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for experiments")
    parser.add_argument("--base_config", type=str, default="barygnn/config/default_config.yaml", help="Path to base configuration file")
    parser.add_argument("--config_dir", type=str, default="barygnn/config/test_configs", help="Output directory for configuration files")
    parser.add_argument("--sh_dir", type=str, default="barygnn/config/test_shell_scripts", help="Output directory for shell script files")
    parser.add_argument("--test", type=str, default="", help="Run single preprepared test")
    args = parser.parse_args()

    if args.test != "":
        # Create SLURM script
        slurm_script_path = create_slurm_script(
            args.test,
            args.sh_dir
        )
        
        submit_job(args.test, slurm_script_path)
        return
    
    # Define parameter grid
    param_grid = {
        "model.pooling.readout_type": ["weighted_mean", "concat"],
        "model.pooling.codebook_size": [32, 64],
        "model.pooling.epsilon": [0.01, 0.05, 0.1],
        "model.encoder.type": ["GIN", "GraphSAGE"],
    }

    # Simplified parameter grid for testing
    param_grid = {
        "model.pooling.readout_type": ["weighted_mean"],
        "model.pooling.codebook_size": [64],
        "model.pooling.epsilon": [0.31],
        "model.encoder.type": ["GIN"],
    }
    
    # Generate configuration files
    config_paths = generate_configs(args.base_config, param_grid, args.config_dir)
    
    # Create SLURM scripts and submit jobs
    for config_path in config_paths:
        # Create SLURM script
        slurm_script_path = create_slurm_script(
            config_path,
            args.sh_dir
        )
        
        submit_job(config_path, slurm_script_path)


if __name__ == "__main__":
    main() 