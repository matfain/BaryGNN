import os
import logging
import argparse
import yaml
import optuna
import torch
import wandb
import shutil
from pathlib import Path
from datetime import datetime

from barygnn import Config, load_dataset
from train import run_training


def load_optuna_config(config_path):
    """
    Load Optuna configuration from YAML file.
    
    Args:
        config_path: Path to Optuna configuration file
        
    Returns:
        optuna_config: Dictionary with Optuna configuration
    """
    with open(config_path, "r") as f:
        optuna_config = yaml.safe_load(f)
    
    return optuna_config


def create_study_output_dir(study_name, base_dir="barygnn/optuna_outputs", timestamp=None):
    """
    Create organized directory structure for an Optuna study.
    
    Args:
        study_name: Name of the study
        base_dir: Base directory for all Optuna outputs
        timestamp: Optional timestamp string for directory naming
        
    Returns:
        study_dir: Path to the study directory
        timestamp: Timestamp string used for the directory
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%d-%m_%H-%M")
    study_dir = os.path.join(base_dir, f"{study_name}_{timestamp}")
    
    # Create main directories
    os.makedirs(study_dir, exist_ok=True)
    os.makedirs(os.path.join(study_dir, "config"), exist_ok=True)
    os.makedirs(os.path.join(study_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(study_dir, "trials"), exist_ok=True)
    os.makedirs(os.path.join(study_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(study_dir, "logs"), exist_ok=True)
    
    return study_dir, timestamp


def generate_trial_config(trial, base_config_path, search_space):
    """
    Generate a configuration for a specific trial.
    
    Args:
        trial: Optuna trial object
        base_config_path: Path to base configuration file
        search_space: Dictionary of search space definitions
        
    Returns:
        config: Configuration for this trial
    """
    # Load base configuration
    base_config = Config.from_yaml(base_config_path)
    
    # Update configuration with trial-suggested values
    for param_name, param_spec in search_space.items():
        # Get parameter value based on type
        if param_spec["type"] == "categorical":
            value = trial.suggest_categorical(param_name, param_spec["choices"])
        elif param_spec["type"] == "int":
            step = param_spec.get("step", 1)
            value = trial.suggest_int(param_name, param_spec["low"], param_spec["high"], step=step)
        elif param_spec["type"] == "float":
            log = param_spec.get("log", False)
            value = trial.suggest_float(param_name, param_spec["low"], param_spec["high"], log=log)
        
        # Set parameter in config
        if "." in param_name:
            parts = param_name.split(".")
            obj = base_config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(base_config, param_name, value)
    
    # Update experiment name to include trial number
    base_config.experiment_type = f"{base_config.experiment_type}_trial{trial.number}"
    
    # Update wandb config if needed
    if base_config.wandb.enabled:
        if not base_config.wandb.tags:
            base_config.wandb.tags = []
        base_config.wandb.tags = base_config.wandb.tags + ["optuna", f"trial_{trial.number}"]
    
    return base_config


def objective(trial, base_config_path, search_space, metric_name, metric_direction, study_dir):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        base_config_path: Path to base configuration file
        search_space: Dictionary of search space definitions
        metric_name: Name of the metric to optimize
        metric_direction: Direction of optimization ("maximize" or "minimize")
        study_dir: Directory for study outputs
        
    Returns:
        metric_value: Value of the metric to optimize
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting trial {trial.number}")
    
    # Generate configuration for this trial
    config = generate_trial_config(trial, base_config_path, search_space)
    
    # Create trial-specific directory
    trial_dir = os.path.join(study_dir, "trials", f"trial_{trial.number}")
    trial_logs_dir = os.path.join(trial_dir, "logs")
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(trial_logs_dir, exist_ok=True)
    
    # Save trial configuration
    config_path = os.path.join(trial_dir, "config.yaml")
    config.to_yaml(config_path)
    
    # Set up file handler for this trial
    file_handler = logging.FileHandler(os.path.join(trial_logs_dir, "trial.log"), mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    try:      
        # Set up arguments for run_training
        class Args:
            def __init__(self, config_path, log_dir, checkpoint_dir):
                self.config = config_path
                self.log_dir = log_dir
                self.checkpoint_dir = checkpoint_dir
        
        # Create args object for run_training
        trial_checkpoint_dir = os.path.join(trial_dir, "checkpoints")
        os.makedirs(trial_checkpoint_dir, exist_ok=True)
        train_args = Args(config_path, trial_logs_dir, trial_checkpoint_dir)
        
        # Monkey patch the global args variable in train.py
        import train
        train.args = train_args
        
        # Train model with this configuration
        metrics = run_training(config)
        
        # Get metric value
        if metric_name == "loss":
            metric_value = metrics["loss"]
        else:
            metric_value = metrics.get(metric_name, 0.0)
        
        # Use the metric value directly - Optuna's direction parameter handles maximization/minimization
        final_value = metric_value
        
        logger.info(f"Trial {trial.number} finished with {metric_name}: {metric_value}")
        
        # Log to wandb if enabled
        if config.wandb.enabled and wandb.run is not None:
            wandb.log({
                "trial_number": trial.number,
                f"trial_{metric_name}": metric_value,
                "trial_params": trial.params
            })
        
        # Rename the best checkpoint to a standardized name
        checkpoint_path = os.path.join(trial_checkpoint_dir, f"{config.experiment_type}_best.pt")
        if os.path.exists(checkpoint_path):
            final_checkpoint_path = os.path.join(trial_checkpoint_dir, "checkpoint.pt")
            shutil.copy(checkpoint_path, final_checkpoint_path)
        
        return final_value
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {str(e)}")
        # Return worst possible value
        return float('-inf') if metric_direction == "maximize" else float('inf')
    
    finally:
        # Remove file handler
        logger.removeHandler(file_handler)


def visualize_study(study, output_dir):
    """
    Generate visualizations for an Optuna study.
    
    Args:
        study: Optuna study object
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot optimization history
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(output_dir, "optimization_history.html"))
    except Exception as e:
        print(f"Error plotting optimization history: {str(e)}")
    
    # Plot parameter importance
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(output_dir, "param_importances.html"))
    except Exception as e:
        print(f"Error plotting parameter importance: {str(e)}")
    
    # Plot parallel coordinate
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(output_dir, "parallel_coordinate.html"))
    except Exception as e:
        print(f"Error plotting parallel coordinate: {str(e)}")
    
    # Plot slice
    try:
        fig = optuna.visualization.plot_slice(study)
        fig.write_html(os.path.join(output_dir, "slice.html"))
    except Exception as e:
        print(f"Error plotting slice: {str(e)}")
    
    # Plot contour
    try:
        fig = optuna.visualization.plot_contour(study)
        fig.write_html(os.path.join(output_dir, "contour.html"))
    except Exception as e:
        print(f"Error plotting contour: {str(e)}")


def run_optuna_study(optuna_config_path, log_dir=None, timestamp=None):
    """
    Run an Optuna study.
    
    Args:
        optuna_config_path: Path to Optuna configuration file
        log_dir: Directory for logs (if None, will be created based on study name)
        timestamp: Optional timestamp string for directory naming
        
    Returns:
        study: Optuna study object
    """
    # Load Optuna configuration
    optuna_config = load_optuna_config(optuna_config_path)
    
    # Create study directory structure
    study_dir, timestamp = create_study_output_dir(optuna_config["study"]["name"], timestamp=timestamp)
    
    # Save the original Optuna config to the study directory
    config_dir = os.path.join(study_dir, "config")
    shutil.copy(optuna_config_path, os.path.join(config_dir, "optuna_config.yaml"))
    
    # Set up logging
    log_dir = os.path.join(study_dir, "logs") if log_dir is None else log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(os.path.join(log_dir, "optuna.log"), mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Starting Optuna study: {optuna_config['study']['name']}")
    logger.info(f"Study directory: {study_dir}")
    
    # Create study - use in-memory storage if not specified
    storage = optuna_config["study"].get("storage")
    
    study = optuna.create_study(
        study_name=optuna_config["study"]["name"],
        direction=optuna_config["study"]["direction"],
        storage=storage,
        load_if_exists=True if storage else False
    )
    
    # Set up pruner if enabled
    if optuna_config["pruning"]["enabled"]:
        pruner_name = optuna_config["pruning"]["pruner"]
        if pruner_name == "median":
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=optuna_config["pruning"]["warmup_steps"]
            )
        else:
            logger.warning(f"Unknown pruner: {pruner_name}, using default MedianPruner")
            pruner = optuna.pruners.MedianPruner()
        
        study.pruner = pruner
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(
                trial, 
                optuna_config["base_config"], 
                optuna_config["search_space"],
                optuna_config["metric"]["name"],
                optuna_config["metric"]["mode"],
                study_dir
            ),
            n_trials=optuna_config["study"]["n_trials"],
            timeout=optuna_config["study"]["timeout"]
        )
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
    
    # Log best trial
    logger.info("Study finished")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")
    
    # Copy best model to study checkpoints directory
    best_trial_checkpoint = os.path.join(study_dir, "trials", f"trial_{study.best_trial.number}", "checkpoints", "checkpoint.pt")
    if os.path.exists(best_trial_checkpoint):
        best_model_path = os.path.join(study_dir, "checkpoints", "best_model.pt")
        shutil.copy(best_trial_checkpoint, best_model_path)
        logger.info(f"Copied best model to {best_model_path}")
    
    # Generate visualizations
    vis_dir = os.path.join(study_dir, "visualizations")
    visualize_study(study, vis_dir)
    logger.info(f"Generated visualizations in {vis_dir}")
    
    # Log best trial to wandb if enabled
    if optuna_config["wandb"]["enabled"]:
        wandb.init(
            project=f"{optuna_config['wandb']['project']}_{optuna_config['wandb']['project_suffix']}",
            name=f"{optuna_config['study']['name']}_summary",
            config=optuna_config
        )
        
        # Log best trial
        wandb.log({
            "best_trial_number": study.best_trial.number,
            "best_trial_value": study.best_value,
            "best_trial_params": study.best_params
        })
        
        # Log top 5 trials
        sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=(optuna_config["study"]["direction"] == "maximize"))
        for i, trial in enumerate(sorted_trials[:5]):
            wandb.log({
                f"top_trial_{i+1}/number": trial.number,
                f"top_trial_{i+1}/value": trial.value,
                f"top_trial_{i+1}/params": trial.params
            })
        
        wandb.finish()
    
    # Remove file handler
    logger.removeHandler(file_handler)
    
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BaryGNN Optuna Hyperparameter Optimization')
    parser.add_argument('--config', type=str, required=True, help='Path to Optuna configuration file')
    parser.add_argument('--log_dir', type=str, help='Directory to save logs (optional, defaults to study directory)')
    parser.add_argument('--timestamp', type=str, help='Timestamp for directory naming (optional)')
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ],
        force=True  # Ensures our handlers are always used
    )
    
    # Run Optuna study
    study = run_optuna_study(args.config, args.log_dir, args.timestamp)