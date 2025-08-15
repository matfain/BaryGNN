#!/usr/bin/env python3
import os
import argparse
import logging
import json
import yaml
import numpy as np
import torch
import train
import scipy.stats
from pathlib import Path
import subprocess
from datetime import datetime
import time
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, List, Tuple, Optional, Union, Any
import statistics
from copy import deepcopy
from torch_geometric.datasets import TUDataset
from barygnn import Config
from barygnn.datasets.dataset import load_dataset
from train import run_training


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run k-fold cross-validation for BaryGNN')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='barygnn/k_fold_outputs', help='Directory to save results')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--metrics', type=str, nargs='+', default=['accuracy', 'macro_f1', 'roc_auc'], 
                        help='Metrics to report')
    parser.add_argument('--stratified', action='store_true', help='Use stratified k-fold')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # SLURM parameters
    parser.add_argument('--slurm', action='store_true', help='Run on SLURM')
    parser.add_argument('--partition', type=str, default=None, help='SLURM partition')
    parser.add_argument('--mem', type=str, default=None, help='SLURM memory allocation')
    parser.add_argument('--time', type=str, default=None, help='SLURM time limit')
    parser.add_argument('--gpu', type=int, default=None, help='Number of GPUs')
    
    return parser.parse_args()


def create_slurm_script(config_path: str, output_dir: str, job_name: str, timestamp: str) -> str:
    """
    Create a SLURM script for running k-fold cross-validation.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
        job_name: Name for the SLURM job
        timestamp: Timestamp string for organizing outputs
        
    Returns:
        slurm_script_path: Path to generated SLURM script
    """
    # Load configuration
    config = Config.from_yaml(config_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create SLURM script
    slurm_script_path = output_dir / f"{job_name}.sh"
    
    # Set up log directory path
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Use SLURM parameters from args if provided, otherwise use config
    partition = args.partition or config.slurm.partition
    mem = args.mem or config.slurm.mem
    timelimit = args.time or config.slurm.timelimit
    gpu = args.gpu or config.slurm.gpu
    
    with open(slurm_script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"#SBATCH --job-name={job_name}\n")
        f.write(f"#SBATCH --output={log_dir}/slurm.out\n")
        f.write(f"#SBATCH --error={log_dir}/slurm.err\n")
        f.write(f"#SBATCH --time={timelimit}\n")
        f.write(f"#SBATCH --partition={partition}\n")
        f.write(f"#SBATCH --mem={mem}\n")
        f.write(f"#SBATCH --gres=gpu:{gpu}\n")
        f.write("\n")
        f.write("export PYTHONDONTWRITEBYTECODE=1\n")
        f.write("export PYTHONWARNINGS='ignore::UserWarning'\n")
        f.write("export PYTHONPATH=/home/yandex/MLWG2025/amitr5/BaryGNN:$PYTHONPATH\n")
        f.write("# Activate conda environment\n")
        f.write("source /home/yandex/MLWG2025/amitr5/BaryGNN/anaconda3/bin/activate barygnn\n")
        f.write("\n")
        f.write("# Run cross-validation\n")
        f.write(f"python run_cross_validation.py --config {config_path} --output_dir {output_dir} --n_folds {args.n_folds}")
        
        # Add additional arguments
        if args.stratified:
            f.write(" --stratified")
        if args.metrics:
            metrics_str = ' '.join(args.metrics)
            f.write(f" --metrics {metrics_str}")
        f.write(f" --seed {args.seed}")
        f.write("\n")
        
    return slurm_script_path


def prepare_k_fold_indices(dataset_name: str, n_folds: int, stratified: bool, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare indices for k-fold cross-validation.
    
    Args:
        dataset_name: Name of the dataset
        n_folds: Number of folds
        stratified: Whether to use stratified k-fold
        seed: Random seed
        
    Returns:
        fold_indices: List of (train_indices, test_indices) tuples for each fold
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For OGB datasets, we don't support cross-validation
    if dataset_name.startswith("ogbg"):
        raise ValueError(
            f"Cross-validation is not supported for OGB datasets ({dataset_name}). "
            f"OGB datasets have predefined splits that should be used instead. "
            f"Please use a TU dataset for cross-validation."
        )
    
    # For TU datasets - load the dataset WITHOUT transforms, just to get indices and labels
    # Load the dataset without transforms - we only need the structure and labels
    dataset = TUDataset(root="data", name=dataset_name, transform=None)
    
    # Get all indices
    all_indices = np.arange(len(dataset))
    
    # Get labels for stratification
    labels = []
    for data in dataset:
        if hasattr(data, 'y') and data.y is not None:
            if data.y.dim() > 1 and data.y.size(-1) == 1:
                labels.append(data.y.squeeze().item())
            else:
                labels.append(data.y.item())
        else:
            # If no labels, just use dummy labels
            labels.append(0)
    labels = np.array(labels)
    
    # Create k-fold splitter
    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_indices = [(train_idx, test_idx) for train_idx, test_idx in kf.split(all_indices, labels)]
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_indices = [(train_idx, test_idx) for train_idx, test_idx in kf.split(all_indices)]
    
    return fold_indices


def run_fold(config: Config, fold_idx: int, train_indices: np.ndarray, test_indices: np.ndarray, 
             fold_dir: str) -> Dict[str, float]:
    """
    Run a single fold of cross-validation.
    
    Args:
        config: Configuration object
        fold_idx: Fold index
        train_indices: Training indices
        test_indices: Test indices
        fold_dir: Directory for fold outputs
        
    Returns:
        metrics: Dictionary of metrics for this fold
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running fold {fold_idx+1}/{args.n_folds}")
    
    # Create fold-specific directory
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config for this fold
    fold_config = deepcopy(config)
    fold_config.experiment_type = f"{config.experiment_type}_fold{fold_idx+1}"
    fold_config.seed = config.seed + fold_idx  # Use different seed for each fold
    
    # Set cross-validation mode and custom indices
    fold_config.data.cross_val_mode = True
    
    # Convert NumPy arrays to Python lists to avoid YAML serialization issues
    if isinstance(train_indices, np.ndarray):
        train_indices = train_indices.tolist()
    if isinstance(test_indices, np.ndarray):
        test_indices = test_indices.tolist()
        
    fold_config.data.custom_train_indices = train_indices
    fold_config.data.custom_test_indices = test_indices
    
    # Set up arguments for run_training
    class Args:
        def __init__(self, config_path, log_dir, checkpoint_dir):
            self.config = config_path
            self.log_dir = log_dir
            self.checkpoint_dir = checkpoint_dir
    
    # Save fold-specific config
    fold_config_path = fold_dir / "config.yaml"
    fold_config.to_yaml(str(fold_config_path))
    
    # Set up directories for this fold
    fold_log_dir = fold_dir / "logs"
    fold_checkpoint_dir = fold_dir / "checkpoints"
    fold_log_dir.mkdir(parents=True, exist_ok=True)
    fold_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up args for run_training
    train_args = Args(
        config_path=str(fold_config_path),
        log_dir=str(fold_log_dir),
        checkpoint_dir=str(fold_checkpoint_dir)
    )
    
    # Monkey patch the global args variable in train.py
    train.args = train_args
    
    # Set up logging for this fold
    fold_logger = logging.getLogger(f"fold_{fold_idx+1}")
    fold_logger.setLevel(logging.INFO)
    
    # Add file handler for fold-specific logging
    fold_log_file = fold_log_dir / "fold.log"
    fold_handler = logging.FileHandler(fold_log_file, mode='w')
    fold_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    fold_logger.addHandler(fold_handler)
    
    # Train model with this fold
    try:
        metrics = run_training(fold_config)
        
        # Save metrics for this fold
        metrics_path = fold_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Fold {fold_idx+1} completed successfully")
    except Exception as e:
        logger.error(f"Error in fold {fold_idx+1}: {str(e)}")
        metrics = {"error": str(e)}
    
    # Remove fold-specific handler
    fold_logger.removeHandler(fold_handler)
    
    return metrics


def aggregate_results(fold_metrics: List[Dict[str, float]], output_dir: str, metrics_list: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across all folds.
    
    Args:
        fold_metrics: List of metrics dictionaries from each fold
        output_dir: Directory to save aggregated results
        metrics_list: List of metrics to report
        
    Returns:
        aggregated_metrics: Dictionary of aggregated metrics
    """
    logger = logging.getLogger(__name__)
    logger.info("Aggregating results across folds")
    
    # Initialize aggregated metrics
    aggregated_metrics = {}
    
    # Collect all metrics
    all_metrics = {}
    for metric_name in metrics_list:
        metric_values = []
        for fold_idx, metrics in enumerate(fold_metrics):
            if metric_name in metrics:
                metric_values.append(metrics[metric_name])
        
        if metric_values:
            all_metrics[metric_name] = metric_values
    
    # Calculate statistics for each metric
    for metric_name, values in all_metrics.items():
        if not values:
            continue
            
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0
        min_value = min(values)
        max_value = max(values)
        
        # Calculate 95% confidence interval
        if len(values) > 1:
            confidence = 0.95
            n = len(values)
            mean = statistics.mean(values)
            std_err = statistics.stdev(values) / (n ** 0.5)
            h = std_err * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
            ci_low = mean - h
            ci_high = mean + h
        else:
            ci_low = values[0]
            ci_high = values[0]
        
        aggregated_metrics[metric_name] = {
            "mean": mean_value,
            "std": std_value,
            "min": min_value,
            "max": max_value,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "values": values
        }
    
    # Save aggregated metrics
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    
    # Log summary
    logger.info("=== Cross-Validation Summary ===")
    for metric_name, stats in aggregated_metrics.items():
        logger.info(f"{metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f} (95% CI: [{stats['ci_95_low']:.4f}, {stats['ci_95_high']:.4f}])")
    
    return aggregated_metrics


def main():
    """Main function for k-fold cross-validation."""
    global args
    args = parse_arguments()
    
    # Generate timestamp for organizing outputs
    timestamp = datetime.now().strftime("%d-%m_%H-%M")  # DD-MM_HH-MM format
    
    # Create output directory with timestamp
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Create job-specific directory
    job_name = f"{config.experiment_type}_{config.data.name}_{timestamp}"
    job_dir = base_output_dir / job_name
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Save a copy of the original config
    original_config_path = job_dir / "original_config.yaml"
    with open(original_config_path, 'w') as f:
        with open(args.config, 'r') as src:
            f.write(src.read())
    
    # Set up logging
    log_dir = job_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_dir / "cv.log"), mode='w')
        ],
        force=True  # Ensures our handlers are always used
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {args.n_folds}-fold cross-validation for {config.experiment_type}")
    logger.info(f"Output directory: {job_dir}")
    
    # If running on SLURM, submit the job
    if args.slurm:
        # Create SLURM script
        slurm_script_path = create_slurm_script(
            config_path=args.config,
            output_dir=job_dir,
            job_name=job_name,
            timestamp=timestamp
        )
        
        logger.info(f"Submitting SLURM job: {slurm_script_path}")
        subprocess.run(["sbatch", str(slurm_script_path)])
        return
    
    try:
        # Prepare k-fold indices
        fold_indices = prepare_k_fold_indices(
            dataset_name=config.data.name,
            n_folds=args.n_folds,
            stratified=args.stratified,
            seed=args.seed
        )
        
        # Save fold indices for reproducibility
        indices_path = job_dir / "fold_indices.npz"
        fold_data = {f"fold_{i}_train": train_idx for i, (train_idx, _) in enumerate(fold_indices)}
        fold_data.update({f"fold_{i}_test": test_idx for i, (_, test_idx) in enumerate(fold_indices)})
        np.savez(indices_path, **fold_data)
        
        # Run each fold
        fold_metrics = []
        for fold_idx, (train_indices, test_indices) in enumerate(fold_indices):
            fold_dir = job_dir / f"fold_{fold_idx+1}"
            metrics = run_fold(
                config=config,
                fold_idx=fold_idx,
                train_indices=train_indices,
                test_indices=test_indices,
                fold_dir=fold_dir
            )
            fold_metrics.append(metrics)
    except ValueError as e:
        if "Cross-validation is not supported for OGB datasets" in str(e):
            logger.error(str(e))
            logger.error("Cannot run cross-validation on OGB datasets. Exiting.")
            return
        else:
            raise
    
    # Aggregate results
    aggregate_results(fold_metrics, job_dir, args.metrics)
    
    logger.info("Cross-validation completed successfully!")


if __name__ == "__main__":
    main()