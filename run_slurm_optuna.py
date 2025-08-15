import os
import argparse
from pathlib import Path
import subprocess
import time
import logging
from datetime import datetime

from barygnn.config import OptunaConfig


def create_optuna_slurm_script(optuna_config_path, output_dir, timestamp=None):
    """
    Create a SLURM script for running an Optuna study.
    
    Args:
        optuna_config_path: Path to Optuna configuration file
        output_dir: Output directory for SLURM scripts
        timestamp: Timestamp string for organizing outputs
        
    Returns:
        slurm_script_path: Path to generated SLURM script
    """
    # Load configuration
    optuna_config = OptunaConfig.from_yaml(optuna_config_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%d-%m_%H-%M")  # DD-MM_HH-MM format
    
    # Create SLURM script
    job_name = f"optuna_{optuna_config.study.name}"
    slurm_script_path = output_dir / f"{job_name}.sh"
    
    # Set up study directory path
    study_dir = f"barygnn/optuna_outputs/{optuna_config.study.name}_{timestamp}"
    log_dir = f"{study_dir}/logs"
    
    with open(slurm_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={job_name}\n")
        f.write(f"#SBATCH --output={log_dir}/slurm.out\n")
        f.write(f"#SBATCH --error={log_dir}/slurm.err\n")
        f.write(f"#SBATCH --time=24:00:00\n")  # 24 hours time limit
        f.write(f"#SBATCH --partition=studentkillable\n")
        f.write(f"#SBATCH --mem=16G\n")
        f.write(f"#SBATCH --gres=gpu:1\n")
        f.write("\n")
        f.write("export TMPDIR=/home/yandex/MLWG2025/amitr5/tmp\n")
        f.write("export PYTHONDONTWRITEBYTECODE=1\n")
        f.write("export PYTHONWARNINGS='ignore::UserWarning'\n")
        f.write("export PYTHONPATH=/home/yandex/MLWG2025/amitr5/BaryGNN:$PYTHONPATH\n")
        f.write("# Activate conda environment\n")
        f.write(f"source /home/yandex/MLWG2025/amitr5/BaryGNN/anaconda3/bin/activate barygnn\n")
        f.write("\n")
        f.write("# Create directory structure\n")
        f.write(f"mkdir -p {study_dir}/{{config,checkpoints,trials,visualizations,logs}}\n")
        f.write("\n")
        f.write("# Run Optuna study\n")
        f.write(f"python run_optuna.py --config {optuna_config_path} --timestamp {timestamp}\n")
        f.write(f"chmod -R g+rwx {study_dir}\n")
 
    return slurm_script_path, timestamp


def submit_optuna_job(optuna_config_path, slurm_script_path):
    """
    Submit an Optuna job to SLURM.
    
    Args:
        optuna_config_path: Path to Optuna configuration file
        slurm_script_path: Path to SLURM script
    """
    print(f"Submitting Optuna job for config: {optuna_config_path}")
    subprocess.run(["sbatch", str(slurm_script_path)])
    time.sleep(1)


def main():
    """
    Main function.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Submit Optuna SLURM jobs for hyperparameter optimization")
    parser.add_argument("--optuna_config", type=str, required=True, help="Path to Optuna configuration file")
    parser.add_argument("--sh_dir", type=str, default="barygnn/config/optuna_shell_scripts", help="Output directory for shell script files")
    parser.add_argument("--visualize", action="store_true", help="Visualize Optuna study results")
    args = parser.parse_args()
    
    # Create output directory for shell scripts
    os.makedirs(args.sh_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%d-%m_%H-%M")  # DD-MM_HH-MM format
    
    # Create and submit SLURM script
    slurm_script_path, timestamp = create_optuna_slurm_script(
        args.optuna_config,
        args.sh_dir,
        timestamp=timestamp
    )
    
    submit_optuna_job(args.optuna_config, slurm_script_path)


if __name__ == "__main__":
    main()