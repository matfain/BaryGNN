import os
import yaml
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any, Literal

from pathlib import Path
import itertools
from copy import deepcopy
import logging


# Set up logging
logger = logging.getLogger(__name__)

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
                
        
        # Set experiment name with index instead of parameters
        config.experiment_type = f"{base_config.experiment_type}_job{i:04d}"
        
        # Store the parameter values in the wandb tags for tracking
        param_tags = [f"{name.split('.')[-1]}={value}" for name, value in zip(param_names, params)]
        config.wandb.tags = config.wandb.tags + param_tags if config.wandb.tags else param_tags
        config.wandb.project = f"{base_config.wandb.project}_{output_dir.name}"
        
        # Store parameter details in notes for reference
        param_details = ", ".join(param_tags)
        config.wandb.notes = f"Grid search job {i:04d}. Parameters: {param_details}"
        # Save configuration
        config_path = output_dir / f"{config.experiment_type}.yaml"
        config.to_yaml(config_path)
        config_paths.append(config_path)
    
    return config_paths



@dataclass
class EncoderConfig:
    """Enhanced configuration for the encoder."""
    
    type: str = "GIN"  # GIN or GraphSAGE
    in_dim: int = 0  # Will be set based on dataset
    num_layers: int = 3
    dropout: float = 0.5
    aggr: str = "mean"  # For GraphSAGE
    
    # Multi-head encoder parameters
    multi_head_type: str = "efficient"  # "full" or "efficient"
    shared_layers: int = 1
    distribution_size: int = 32  # Number of vectors per node (moved from pooling)
    projection_depth: int = 2  # Number of layers in each projection head (efficient only)
    projection_width_factor: float = 1.0  # Width multiplier for hidden layers in projection head (efficient only)
    use_categorical_encoding: bool = False

@dataclass
class PoolingConfig:
    """Simplified configuration for the pooling methods."""
    
    backend: str = "barycenter"  # "barycenter", "regular_pooling"
    standard_pooling_method: str = "global_mean_pool"  # "global_add_pool", "global_mean_pool", "global_max_pool"
    
    # Parameters only used by barycenter backend
    readout_type: str = "weighted_mean"  # Only for barycenter: "weighted_mean" or "concat"
    codebook_size: int = 16
    epsilon: float = 0.2
    p: int = 2  # Order of Wasserstein distance
    scaling: float = 0.9


@dataclass
class ClassificationConfig:
    """Enhanced configuration for the classification head."""
    
    type: str = "enhanced"  # "simple", "enhanced", "adaptive", "deep_residual"
    
    # Common parameters
    dropout: float = 0.2
    activation: str = "relu"  # "relu", "leaky_relu", "gelu", "swish"
    norm_type: Optional[str] = "batch"  # "batch", "layer", None
    
    # Enhanced/Adaptive MLP parameters
    hidden_dims: Union[List[int], int] = field(default_factory=lambda: [256, 128, 64])
    use_residual: bool = True
    residual_type: str = "add"  # "add", "concat"
    final_dropout: float = 0.5
    
    # Adaptive MLP parameters
    depth_factor: float = 1.0
    width_factor: float = 1.0
    
    # Deep Residual MLP parameters
    hidden_dim: int = 256
    num_blocks: int = 3
    
    # Simple MLP parameters (for backward compatibility)
    num_layers: int = 2


@dataclass
class RegularizationConfig:
    """Configuration for distribution regularization."""
    
    enabled: bool = True
    type: str = "variance"  # "variance", "centroid", "coherence"
    lambda_reg: float = 0.01


@dataclass
class ModelConfig:
    """Simplified configuration for the BaryGNN model."""
    
    version: str = "v2"
    hidden_dim: int = 64
    debug_mode: bool = False
    
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)


@dataclass
class DataConfig:
    """Enhanced configuration for the dataset."""
    
    name: str = "MUTAG"  # Dataset name
    batch_size: int = 32
    num_workers: int = 4
    split_seed: int = 42
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Data preprocessing
    normalize_features: bool = True
    add_self_loops: bool = True
    
    # Cross-validation specific fields
    cross_val_mode: bool = False
    custom_train_indices: Optional[np.ndarray] = None
    custom_test_indices: Optional[np.ndarray] = None


@dataclass
class TrainingConfig:
    """Enhanced configuration for training."""
    
    num_epochs: int = 300
    lr: float = 0.001
    weight_decay: float = 5e-4
    patience: int = 20  # Early stopping patience
    metric: str = "accuracy"  # Metric to track for early stopping
    device: str = "cuda"  # "cuda" or "cpu"
    
    # Advanced training parameters
    scheduler: Optional[str] = None  # "cosine", "step", "plateau"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    gradient_clip: Optional[float] = 1.0
    warmup_epochs: int = 0
    
    # Loss weighting
    class_weights: Optional[List[float]] = None
    focal_loss: bool = False
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0


@dataclass
class WandbConfig:
    """Enhanced configuration for Weights & Biases."""
    
    enabled: bool = True
    project: str = "BaryGNN"
    entity: Optional[str] = None
    api_key: Optional[str] = None  # Set this via environment variable
    
    # Advanced W&B features
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    save_code: bool = True
    log_gradients: bool = False
    log_parameters: bool = True
    watch_model: bool = True

@dataclass
class SlurmConfig:
    partition: str = "studentkillable"
    mem: str = "8G"
    timelimit: str = "10:00:00"
    gpu: int = 1

@dataclass
class Config:
    """Enhanced main configuration for BaryGNN."""
    
    experiment_type: str = "barygnn"
    seed: int = 42
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Recursively convert nested dicts to dataclasses
        def convert_encoder(encoder_dict):
            # Handle distribution_size if it exists in the encoder config
            if "distribution_size" in encoder_dict:
                return EncoderConfig(**encoder_dict)
            # Otherwise, use the default value
            return EncoderConfig(**encoder_dict)

        def convert_pooling(pooling_dict):
            # Remove distribution_size if it exists in the pooling config (for backward compatibility)
            if "distribution_size" in pooling_dict:
                pooling_dict = pooling_dict.copy()
                pooling_dict.pop("distribution_size")
            return PoolingConfig(**pooling_dict)

        def convert_classification(classification_dict):
            return ClassificationConfig(**classification_dict)
        
        def convert_regularization(reg_dict):
            return RegularizationConfig(**reg_dict)

        def convert_model(model_dict):
            encoder_dict = model_dict.get("encoder", {})
            pooling_dict = model_dict.get("pooling", {})
            
            return ModelConfig(
                version=model_dict.get("version", "v2"),
                hidden_dim=model_dict.get("hidden_dim", 64),
                debug_mode=model_dict.get("debug_mode", False),
                encoder=convert_encoder(encoder_dict),
                pooling=convert_pooling(pooling_dict),
                classification=convert_classification(model_dict.get("classification", {})),
                regularization=convert_regularization(model_dict.get("regularization", {})),
            )

        def convert_data(data_dict):
            # Handle custom indices conversion from lists back to NumPy arrays
            if "custom_train_indices" in data_dict and data_dict["custom_train_indices"] is not None:
                if isinstance(data_dict["custom_train_indices"], list):
                    data_dict["custom_train_indices"] = np.array(data_dict["custom_train_indices"], dtype=np.int64)
            
            if "custom_test_indices" in data_dict and data_dict["custom_test_indices"] is not None:
                if isinstance(data_dict["custom_test_indices"], list):
                    data_dict["custom_test_indices"] = np.array(data_dict["custom_test_indices"], dtype=np.int64)
            
            return DataConfig(**data_dict)

        def convert_training(training_dict):
            return TrainingConfig(
                num_epochs=int(training_dict.get("num_epochs", 300)),
                lr=float(training_dict.get("lr", 0.001)),
                weight_decay=float(training_dict.get("weight_decay", 5e-4)),
                patience=int(training_dict.get("patience", 20)),
                metric=training_dict.get("metric", "accuracy"),
                device=training_dict.get("device", "cuda"),
                scheduler=training_dict.get("scheduler"),
                scheduler_params=training_dict.get("scheduler_params", {}),
                gradient_clip=training_dict.get("gradient_clip"),
                warmup_epochs=int(training_dict.get("warmup_epochs", 0)),
                class_weights=training_dict.get("class_weights"),
                focal_loss=training_dict.get("focal_loss", False),
                focal_alpha=float(training_dict.get("focal_alpha", 1.0)),
                focal_gamma=float(training_dict.get("focal_gamma", 2.0)),
            )

        def convert_wandb(wandb_dict):
            return WandbConfig(
                enabled=wandb_dict.get("enabled", True),
                project=wandb_dict.get("project", "BaryGNN"),
                entity=wandb_dict.get("entity"),
                api_key=wandb_dict.get("api_key"),
                tags=wandb_dict.get("tags", []),
                notes=wandb_dict.get("notes"),
                save_code=wandb_dict.get("save_code", True),
                log_gradients=wandb_dict.get("log_gradients", False),
                log_parameters=wandb_dict.get("log_parameters", True),
                watch_model=wandb_dict.get("watch_model", True),
            )
            
        def convert_slurm(slurm_dict):
            return SlurmConfig(
                partition=slurm_dict.get("partition", "studentkillable"),
                mem=slurm_dict.get("mem", "8G"),
                timelimit=slurm_dict.get("timelimit", "10:00:00"),
                gpu=slurm_dict.get("gpu", 1),
            )

        return cls(
            experiment_type=config_dict.get("experiment_type", "barygnn"),
            seed=config_dict.get("seed", 42),
            model=convert_model(config_dict.get("model", {})),
            data=convert_data(config_dict.get("data", {})),
            training=convert_training(config_dict.get("training", {})),
            wandb=convert_wandb(config_dict.get("wandb", {})),
            slurm=convert_slurm(config_dict.get("slurm", {})),
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "experiment_type": self.experiment_type,
            "seed": self.seed,
            "model": {
                "version": self.model.version,
                "hidden_dim": self.model.hidden_dim,
                "debug_mode": self.model.debug_mode,
                "encoder": asdict(self.model.encoder),
                "pooling": {
                    "backend": self.model.pooling.backend,
                    "standard_pooling_method": self.model.pooling.standard_pooling_method,
                    **({"readout_type": self.model.pooling.readout_type,
                        "codebook_size": self.model.pooling.codebook_size,
                        "epsilon": self.model.pooling.epsilon,
                        "p": self.model.pooling.p,
                        "scaling": self.model.pooling.scaling} if self.model.pooling.backend == "barycenter" else {})
                },
                "classification": asdict(self.model.classification),
                "regularization": asdict(self.model.regularization),
            },
            "data": asdict(self.data),
            "training": {
                "num_epochs": self.training.num_epochs,
                "lr": self.training.lr,
                "weight_decay": self.training.weight_decay,
                "patience": self.training.patience,
                "metric": self.training.metric,
                "device": self.training.device,
                "scheduler": self.training.scheduler,
                "scheduler_params": self.training.scheduler_params,
                "gradient_clip": self.training.gradient_clip,
                "warmup_epochs": self.training.warmup_epochs,
                "class_weights": self.training.class_weights,
                "focal_loss": self.training.focal_loss,
                "focal_alpha": self.training.focal_alpha,
                "focal_gamma": self.training.focal_gamma,
            },
            "wandb": asdict(self.wandb),
        }
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """
        Get model initialization kwargs from configuration.
        
        Returns:
            kwargs: Dictionary of model parameters
        """
        return {
            # Data parameters
            'in_dim': self.model.encoder.in_dim,
            'num_classes': 2,  # Will be set based on dataset
            
            # Architecture parameters
            'hidden_dim': self.model.hidden_dim,
            'codebook_size': self.model.pooling.codebook_size,
            'distribution_size': self.model.encoder.distribution_size,
            'pooling_backend': self.model.pooling.backend,
            'standard_pooling_method': self.model.pooling.standard_pooling_method,
            'readout_type': self.model.pooling.readout_type,
            
            # Encoder parameters
            'encoder_type': self.model.encoder.type,
            'encoder_layers': self.model.encoder.num_layers,
            'encoder_dropout': self.model.encoder.dropout,
            'multi_head_type': self.model.encoder.multi_head_type,
            'shared_layers': self.model.encoder.shared_layers,
            'projection_depth': self.model.encoder.projection_depth,
            'projection_width_factor': self.model.encoder.projection_width_factor,
            
            # Pooling parameters
            'sinkhorn_epsilon': self.model.pooling.epsilon,
            'p': self.model.pooling.p,
            'scaling': self.model.pooling.scaling,
            
            # Classification parameters
            'classifier_type': self.model.classification.type,
            'classifier_hidden_dims': self.model.classification.hidden_dims,
            'classifier_dropout': self.model.classification.dropout,
            'classifier_activation': self.model.classification.activation,
            'classifier_depth_factor': self.model.classification.depth_factor,
            'classifier_width_factor': self.model.classification.width_factor,
            
            # Regularization parameters
            'use_distribution_reg': self.model.regularization.enabled,
            'reg_type': self.model.regularization.type,
            'reg_lambda': self.model.regularization.lambda_reg,
            
            'use_categorical_encoding': self.model.encoder.use_categorical_encoding,
            # General parameters
            'debug_mode': self.model.debug_mode,
        } 


@dataclass
class OptunaConfig:
    class Study:
        name: str
        direction: str
        storage: Optional[str] = None
        n_trials: Optional[int] = None
        timeout: Optional[Union[int, None]] = None
        n_workers: Optional[int] = None

    class Output:
        config_dir: str
        logs_prefix: str

    class Metric:
        name: str
        mode: str

    class Wandb:
        project_suffix: Optional[str] = None
        extra_tags: Optional[List[str]] = field(default_factory=list)

    class Pruning:
        enabled: bool = False
        pruner: Optional[str] = None
        warmup_steps: Optional[int] = None

    study: Study
    base_config: str
    output: Output
    metric: Metric
    wandb: Wandb
    pruning: Pruning
    search_space: Dict[str, Any]        