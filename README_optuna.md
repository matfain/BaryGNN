# BaryGNN Optuna Integration

This document describes how to use Optuna for hyperparameter optimization in BaryGNN.

## Overview

Optuna is an automatic hyperparameter optimization framework designed for machine learning. The integration with BaryGNN allows you to run hyperparameter optimization as a single SLURM job, rather than submitting a separate job for each hyperparameter combination.

## Files Added

1. `run_optuna.py`: Main script for running Optuna studies
2. `run_slurm_optuna.py`: Script for submitting Optuna studies as SLURM jobs
3. `barygnn/config/optuna_config.py`: Configuration classes for Optuna studies
4. `barygnn/config/example_optuna_config.yaml`: Example Optuna configuration file

## Dependencies

The following dependencies have been added to the project:
- `optuna>=3.0.0`: For hyperparameter optimization
- `plotly>=5.0.0`: For visualization of Optuna results

To install these dependencies, run:

```bash
uv pip compile pyproject.toml --output-file requirements.txt
uv pip install -r requirements.txt
```

## Directory Structure

The Optuna outputs are organized in a clean, hierarchical structure:

```
barygnn/
├── optuna_outputs/
│   └── [study_name]_[timestamp]/
│       ├── config/                  # Study configuration
│       │   └── optuna_config.yaml   # Original Optuna config
│       ├── checkpoints/             # Best overall model only
│       │   └── best_model.pt        # Copy of the best trial's model
│       ├── trials/                  # Individual trial data
│       │   ├── trial_0/
│       │   │   ├── config.yaml      # Trial configuration
│       │   │   ├── logs/            # Trial-specific logs
│       │   │   └── checkpoints/     # Trial-specific checkpoints
│       │   │       └── checkpoint.pt # Final model for this trial
│       │   ├── trial_1/
│       │   │   ├── config.yaml
│       │   │   ├── logs/
│       │   │   └── checkpoints/
│       │   │       └── checkpoint.pt
│       │   └── ...
│       ├── visualizations/          # Optuna visualizations
│       │   ├── optimization_history.html
│       │   ├── param_importances.html
│       │   └── ...
│       └── logs/                    # Study-level logs
│           ├── optuna.log           # Main Optuna log
│           ├── slurm.out            # SLURM output
│           └── slurm.err            # SLURM error
```

## Usage

### 1. Create an Optuna Configuration File

Create a YAML file with your Optuna study configuration. You can use `barygnn/config/example_optuna_config.yaml` as a template.

```yaml
# Example Optuna study configuration
study:
  name: barygnn_mutag_opt
  direction: maximize
  storage: null  # Use in-memory storage
  n_trials: 20
  timeout: null

base_config: barygnn/config/default_barycentric_config.yaml

metric:
  name: accuracy
  mode: max

search_space:
  model.hidden_dim:
    type: categorical
    choices: [32, 64, 128]
  model.encoder.num_layers:
    type: categorical
    choices: [3, 5, 8]
  # Add more hyperparameters as needed
```

### 2. Submit an Optuna Study as a SLURM Job

```bash
python run_slurm_optuna.py --optuna_config barygnn/config/example_optuna_config.yaml
```

This will create a SLURM script and submit it to the queue. The script will run the Optuna study with the specified configuration.

### 3. Run an Optuna Study Directly (without SLURM)

```bash
python run_optuna.py --config barygnn/config/example_optuna_config.yaml
```

### 4. Monitoring Progress

1. **During Execution**:
   - Check `logs/optuna.log` for overall study progress
   - Check `logs/slurm.err` for detailed training logs
   - Check `trials/trial_X/logs/trial.log` for individual trial results

2. **After Completion**:
   - View visualizations in the `visualizations` directory
   - Find the best model at `checkpoints/best_model.pt`
   - See the best trial's parameters in `logs/optuna.log`

### 5. Visualize Optuna Results

The visualizations are automatically generated in the `barygnn/optuna_outputs/[study_name]_[timestamp]/visualizations` directory when the study completes. These include:

- Optimization history
- Parameter importance
- Parallel coordinate plot
- Slice plot
- Contour plot

## Configuration Options

### Study Configuration

- `name`: Name of the study
- `direction`: Direction of optimization (`maximize` or `minimize`)
- `storage`: Storage URL for the study (use `null` for in-memory storage)
- `n_trials`: Number of trials to run
- `timeout`: Timeout in seconds (optional)

### Metric Configuration

- `name`: Name of the metric to optimize (`accuracy`, `macro_f1`, `roc_auc`, or `loss`)
- `mode`: Direction of optimization (`max` or `min`)

### Search Space Configuration

Each hyperparameter in the search space is defined with a type and additional parameters:

#### Categorical Parameters

```yaml
model.hidden_dim:
  type: categorical
  choices: [32, 64, 128]
```

#### Integer Parameters

```yaml
training.patience:
  type: int
  low: 20
  high: 60
  step: 10  # Optional, default is 1
```

#### Float Parameters

```yaml
model.pooling.epsilon:
  type: float
  low: 0.05
  high: 0.5
  log: true  # Optional, for log-uniform sampling
```

## Wandb Integration

The Optuna integration includes Weights & Biases logging for both individual trials and the overall study results.

```yaml
wandb:
  enabled: true
  project: barygnn
  project_suffix: optuna
  extra_tags: [optuna]
```

## Notes

- The SLURM job is configured with a 24-hour time limit
- Each trial's configuration and results are saved in the logs directory
- All outputs for a study are kept in a single timestamp-based directory
- Each trial has its own checkpoint directory
- The best model is copied to the study-level checkpoints directory