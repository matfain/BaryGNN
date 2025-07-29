import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any


@dataclass
class EncoderConfig:
    """Configuration for the encoder."""
    
    type: str = "GIN"  # GIN or GraphSAGE
    in_dim: int = 0  # Will be set based on dataset
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.5
    aggr: str = "mean"  # For GraphSAGE


@dataclass
class PoolingConfig:
    """Configuration for the barycentric pooling."""
    
    codebook_size: int = 16
    distribution_size: int = 32
    epsilon: float = 0.1
    num_iterations: int = 20


@dataclass
class ClassificationConfig:
    """Configuration for the classification head."""
    
    num_layers: int = 2
    dropout: float = 0.5


@dataclass
class ModelConfig:
    """Configuration for the BaryGNN model."""
    
    hidden_dim: int = 64
    readout_type: str = "weighted_mean"  # "weighted_mean" or "concat"
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)


@dataclass
class DataConfig:
    """Configuration for the dataset."""
    
    name: str = "MUTAG"  # Dataset name
    batch_size: int = 32
    num_workers: int = 4
    split_seed: int = 42
    val_ratio: float = 0.1
    test_ratio: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    num_epochs: int = 300
    lr: float = 0.001
    weight_decay: float = 5e-4
    patience: int = 20  # Early stopping patience
    metric: str = "accuracy"  # Metric to track for early stopping
    device: str = "cuda"  # "cuda" or "cpu"


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases."""
    
    enabled: bool = True
    project: str = "BaryGNN"
    entity: Optional[str] = None
    api_key: Optional[str] = None  # Set this via environment variable


@dataclass
class Config:
    """Main configuration."""
    
    experiment_name: str = "default"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "model": {
                "hidden_dim": self.model.hidden_dim,
                "readout_type": self.model.readout_type,
                "encoder": {
                    "type": self.model.encoder.type,
                    "in_dim": self.model.encoder.in_dim,
                    "hidden_dim": self.model.encoder.hidden_dim,
                    "num_layers": self.model.encoder.num_layers,
                    "dropout": self.model.encoder.dropout,
                    "aggr": self.model.encoder.aggr,
                },
                "pooling": {
                    "codebook_size": self.model.pooling.codebook_size,
                    "distribution_size": self.model.pooling.distribution_size,
                    "epsilon": self.model.pooling.epsilon,
                    "num_iterations": self.model.pooling.num_iterations,
                },
                "classification": {
                    "num_layers": self.model.classification.num_layers,
                    "dropout": self.model.classification.dropout,
                },
            },
            "data": {
                "name": self.data.name,
                "batch_size": self.data.batch_size,
                "num_workers": self.data.num_workers,
                "split_seed": self.data.split_seed,
                "val_ratio": self.data.val_ratio,
                "test_ratio": self.data.test_ratio,
            },
            "training": {
                "num_epochs": self.training.num_epochs,
                "lr": self.training.lr,
                "weight_decay": self.training.weight_decay,
                "patience": self.training.patience,
                "metric": self.training.metric,
                "device": self.training.device,
            },
            "wandb": {
                "enabled": self.wandb.enabled,
                "project": self.wandb.project,
                "entity": self.wandb.entity,
                "api_key": self.wandb.api_key,
            },
        } 