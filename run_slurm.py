import os
import argparse
from pathlib import Path
import subprocess
import time
import logging
from datetime import datetime
from barygnn.config import Config, generate_configs



def create_slurm_script(config_path, output_dir, timestamp=None, is_grid_search=False):
    """
    Create a SLURM script for running an experiment.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for SLURM scripts
        timestamp: Timestamp string for organizing outputs
        is_grid_search: Whether this is part of a grid search
        
    Returns:
        slurm_script_path: Path to generated SLURM script
    """
    # Load configuration
    config = Config.from_yaml(config_path)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create SLURM script
    if is_grid_search:
        # For grid search, use the experiment_type directly (which now contains the job index)
        job_name = f"{config.experiment_type}"
    else:
        # For individual tests, use the old naming convention
        job_name = f"{config.experiment_type}_{config.data.name}_{config.model.pooling.backend}"
    
    slurm_script_path = output_dir / f"{job_name}.sh"
    
    # Set up log directory path
    if is_grid_search and timestamp:
        # For grid search, use grid_search_logs/timestamp/job_name
        log_dir = f"grid_search_logs/{timestamp}/{job_name}"
    elif timestamp:
        # For individual runs with timestamp
        log_dir = f"{timestamp}/{job_name}"
    else:
        # For individual runs without timestamp
        log_dir = job_name
    with open(slurm_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={job_name}\n")
        f.write(f"#SBATCH --output=barygnn/logs/{log_dir}/logger.out\n")
        f.write(f"#SBATCH --error=barygnn/logs/{log_dir}/logger.err\n")
        f.write(f"#SBATCH --time={config.slurm.timelimit}\n")
        f.write(f"#SBATCH --partition={config.slurm.partition}\n")
        f.write(f"#SBATCH --mem={config.slurm.mem}\n")
        f.write(f"#SBATCH --gres=gpu:{config.slurm.gpu}\n")
        f.write("\n")
        f.write("# Activate conda environment\n")
        f.write(f"source /home/yandex/MLWG2025/amitr5/BaryGNN/anaconda3/bin/activate barygnn\n")
        f.write("\n")
        f.write("# Run experiment\n")
        f.write(f"python train.py --config {config_path} --log_dir barygnn/logs/{log_dir}\n")
        f.write(f"chmod -R g+rwx /home/yandex/MLWG2025/amitr5/BaryGNN/barygnn/logs/{log_dir}\n")
 
    return slurm_script_path


def submit_job(config_path, slurm_script_path, job_limit=10):
        print(f"Submitting job for config: {config_path}")
        subprocess.run(["sbatch", str(slurm_script_path)])

        # Wait a bit to avoid potential issues
        time.sleep(1)

        while count_my_slurm_jobs() > job_limit:
            print(f"Too many jobs in the queue ({count_my_slurm_jobs()}), waiting...")
            time.sleep(60)
        

def count_my_slurm_jobs():
    """
    Count the number of jobs in the SLURM queue for the current user.
    
    Returns:
        int: The number of jobs in the queue (excluding the header line)
    """
    # Run squeue --me and capture the output
    result = subprocess.run(["squeue", "--me"], capture_output=True, text=True)
    
    # Split the output by lines and count them
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        job_count = len(lines) - 1  # Subtract 1 to exclude the header line
        return job_count
    else:
        print("Error running squeue command:", result.stderr)
        return -1


def auto_handle_slurm(config_dir):
    """
    Automatically handle SLURM bullshit.
    
    This function is a placeholder for any automatic handling of SLURM-related issues.
    """
    script_path = '/home/yandex/MLWG2025/amitr5/BaryGNN/barygnn/utils/auto_slurm.sh'
    job_name = "auto_slurm"
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name=Slurm_Blushit_Handler\n")
        f.write(f"#SBATCH --output=barygnn/logs/{job_name}/logger.out\n")
        f.write(f"#SBATCH --error=barygnn/logs/{job_name}/logger.err\n")
        f.write(f"#SBATCH --time=12:00:00\n")
        f.write(f"#SBATCH --partition=studentkillable\n")
        # f.write(f"#SBATCH --mem={config.slurm.mem}\n")
        # f.write(f"#SBATCH --gres=gpu:{config.slurm.gpu}\n")
        f.write("\n")
        
        f.write("# Activate conda environment\n")
        f.write(f"source /home/yandex/MLWG2025/amitr5/BaryGNN/anaconda3/bin/activate barygnn\n")
        f.write("\n")
        f.write("# Run experiment\n")
        f.write(f"python run_slurm.py --base_config {config_dir}\n")

    print(f"Running automatic SLURM bullshit handler script")
    subprocess.run(["sbatch", str(script_path)])
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

    parser.add_argument("--auto", action="store_true", help="Automatically handle SLURM bullshit")
    args = parser.parse_args()

    if args.auto:
        # Automatically handle SLURM bullshit
        print("Automatically handling SLURM bullshit...")
        auto_handle_slurm(args.base_config)

        return

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
        "model.hidden_dim": [32, 64],
        "model.encoder.num_layers": [3, 5, 8, 12],
        "model.pooling.backend": ["barycenter"],
        "model.pooling.readout_type": ["weighted_mean"],
        "model.pooling.standard_pooling_method": ["global_mean_pool", "global_max_pool", "global_add_pool"],
        "model.pooling.codebook_size": [32, 64],
        "model.pooling.epsilon": [0.05, 0.1, 0.2],
        "training.patience": [20, 40, 60],
    }
    
    timestamp = datetime.now().strftime("%d-%m_%H-%M")  # DD-MM_HH-MM format

    # Create grid_search_configs directory inside config dir with timestamp subdirectory
    grid_search_dir = Path("barygnn/config/grid_search_configs")
    grid_search_dir.mkdir(parents=True, exist_ok=True)
    
    timestamped_dir = grid_search_dir / timestamp
    timestamped_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate configuration files in the timestamped directory
    config_paths = generate_configs(args.base_config, param_grid, timestamped_dir)
    
    # Create SLURM scripts in the same timestamped directory
    grid_sh_dir = timestamped_dir
    for config_path in config_paths:
        # Create SLURM script
        slurm_script_path = create_slurm_script(
            config_path,
            grid_sh_dir,
            timestamp=timestamp,
            is_grid_search=True
        )
        
        submit_job(config_path, slurm_script_path)


if __name__ == "__main__":
    main() 