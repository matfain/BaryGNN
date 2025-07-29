# BaryGNN

This is the basic project for BaryGNN, focused on barycenter pooling methods for Graph Neural Networks (GNNs).

## Environment Setup (conda + uv)

To set up the development environment on a new machine:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/matfain/BaryGNN.git
   cd BaryGNN
   ```

2. **Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)** if not already installed.

3. **Create and activate the conda environment:**
   ```bash
   conda create -y -n barygnn python=3.11
   conda activate barygnn
   ```

4. **Install [uv](https://astral.sh/docs/uv/):**
   ```bash
   pip install uv
   ```

5. **Install all project dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```
   - If you want to regenerate the requirements file from the project definition:
     ```bash
     uv pip compile pyproject.toml --output-file requirements.txt
     uv pip install -r requirements.txt
     ```

6. **You're ready to go!**

If you encounter any issues, make sure your conda environment is active and Python version is 3.11.

## Project Structure

```
BaryGNN/
├── README.md                 # Project overview, installation, usage
├── pyproject.toml            # Project dependencies and metadata
├── requirements.txt          # Pinned dependencies
├── .gitignore                # Git ignore patterns
│
├── barygnn/                  # Main package directory
│   ├── __init__.py           # Package initialization
│   │
│   ├── models/               # Neural network models
│   │   ├── __init__.py
│   │   ├── barygnn.py        # Main BaryGNN model
│   │   │
│   │   ├── encoders/         # Graph node encoders (GNNs)
│   │   │   ├── __init__.py
│   │   │   ├── base.py       # Base encoder interface
│   │   │   ├── gin.py        # GIN encoder implementation
│   │   │   └── sage.py       # GraphSAGE encoder implementation
│   │   │
│   │   ├── pooling/          # Pooling methods
│   │   │   ├── __init__.py
│   │   │   └── barycentric_pooling.py  # Barycentric pooling implementation
│   │   │
│   │   ├── readout/          # Readout methods
│   │   │   ├── __init__.py
│   │   │   └── readout.py    # Readout implementation
│   │   │
│   │   └── classification/   # Classification heads
│   │       ├── __init__.py
│   │       └── mlp.py        # MLP classification head
│   │
│   ├── data/                 # Data loading utilities
│   │   ├── __init__.py
│   │   └── dataset.py        # Dataset loading functions
│   │
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   ├── metrics.py        # Evaluation metrics
│   │   └── logger.py         # Logging utilities with W&B integration
│   │
│   ├── config/               # Configuration utilities
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration classes
│   │   └── default_config.yaml  # Default configuration
│   │
│   └── scripts/              # Training and evaluation scripts
│       ├── train.py          # Training script
│       ├── evaluate.py       # Evaluation script
│       ├── run_experiments.py  # Script for running multiple experiments
│       └── run_slurm.py      # Script for submitting SLURM jobs
```

## Usage

### Training

To train a model with the default configuration:

```bash
python -m barygnn.scripts.train --config barygnn/config/default_config.yaml
```

### Evaluation

To evaluate a trained model:

```bash
python -m barygnn.scripts.evaluate --config barygnn/config/default_config.yaml --checkpoint checkpoints/barygnn_default/best_model.pt
```

### Running Experiments

To run multiple experiments with different configurations:

```bash
python -m barygnn.scripts.run_experiments --base_config barygnn/config/default_config.yaml
```

### Running on SLURM

To submit SLURM jobs for experiments:

```bash
python -m barygnn.scripts.run_slurm --base_config barygnn/config/default_config.yaml --partition studentrun
```

## Configuration

The default configuration is in `barygnn/config/default_config.yaml`. You can modify this file or create a new one to customize the model and training parameters.

Key configuration options:
- **Model**: Architecture, hidden dimensions, readout type
- **Encoder**: GNN type (GIN or GraphSAGE), layers, dropout
- **Pooling**: Codebook size, distribution size, Sinkhorn parameters
- **Data**: Dataset, batch size, splits
- **Training**: Epochs, learning rate, early stopping
- **Weights & Biases**: Logging configuration

## Weights & Biases Integration

To use Weights & Biases for experiment tracking:

1. Set your W&B API key in the configuration file or as an environment variable:
   ```bash
   export WANDB_API_KEY=your_api_key
   ```

2. Enable W&B in the configuration file:
   ```yaml
   wandb:
     enabled: true
     project: "BaryGNN"
     entity: your_username_or_team
   ```