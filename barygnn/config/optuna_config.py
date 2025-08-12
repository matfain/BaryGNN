"""
Optuna configuration classes for BaryGNN hyperparameter optimization.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

from pathlib import Path


@dataclass
class OptunaStudyConfig:
    """Configuration for Optuna study."""
    
    name: str = "barygnn_study"
    direction: str = "maximize"  # "maximize" or "minimize"
    storage: str = "sqlite:///barygnn/optuna/optuna.db"
    n_trials: int = 50
    timeout: Optional[int] = None  # in seconds, None for no timeout
    n_workers: int = 1  # For parallel execution


@dataclass
class OptunaMetricConfig:
    """Configuration for the metric to optimize."""
    
    name: str = "accuracy"  # Metric to optimize
    mode: str = "max"  # "max" or "min"


@dataclass
class OptunaPruningConfig:
    """Configuration for Optuna pruning."""
    
    enabled: bool = False
    pruner: str = "median"  # "median", "percentile", "threshold", etc.
    warmup_steps: int = 5


@dataclass
class OptunaWandbConfig:
    """Configuration for Weights & Biases integration."""
    
    enabled: bool = True
    project: str = "barygnn"
    project_suffix: str = "optuna"
    extra_tags: List[str] = field(default_factory=lambda: ["optuna"])


@dataclass
class OptunaOutputConfig:
    """Configuration for output directories."""
    
    config_dir: str = "barygnn/config/optuna_configs"
    logs_prefix: str = "optuna_logs"


@dataclass
class OptunaSearchSpaceItem:
    """Configuration for a single search space item."""
    
    type: str  # "categorical", "int", "float"
    # Different fields based on type
    choices: Optional[List[Any]] = None  # For categorical
    low: Optional[float] = None  # For int, float
    high: Optional[float] = None  # For int, float
    step: Optional[int] = None  # For int
    log: bool = False  # For float


@dataclass
class OptunaConfig:
    """Main configuration for Optuna hyperparameter optimization."""
    
    study: OptunaStudyConfig = field(default_factory=OptunaStudyConfig)
    base_config: str = "barygnn/config/default_config.yaml"
    output: OptunaOutputConfig = field(default_factory=OptunaOutputConfig)
    metric: OptunaMetricConfig = field(default_factory=OptunaMetricConfig)
    wandb: OptunaWandbConfig = field(default_factory=OptunaWandbConfig)
    pruning: OptunaPruningConfig = field(default_factory=OptunaPruningConfig)
    search_space: Dict[str, OptunaSearchSpaceItem] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> "OptunaConfig":
        """Load Optuna configuration from a YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Create study config
        study_dict = config_dict.get("study", {})
        study_config = OptunaStudyConfig(
            name=study_dict.get("name", "barygnn_study"),
            direction=study_dict.get("direction", "maximize"),
            storage=study_dict.get("storage", "sqlite:///barygnn/optuna/optuna.db"),
            n_trials=study_dict.get("n_trials", 50),
            timeout=study_dict.get("timeout"),
            n_workers=study_dict.get("n_workers", 1)
        )
        
        # Create metric config
        metric_dict = config_dict.get("metric", {})
        metric_config = OptunaMetricConfig(
            name=metric_dict.get("name", "accuracy"),
            mode=metric_dict.get("mode", "max")
        )
        
        # Create pruning config
        pruning_dict = config_dict.get("pruning", {})
        pruning_config = OptunaPruningConfig(
            enabled=pruning_dict.get("enabled", False),
            pruner=pruning_dict.get("pruner", "median"),
            warmup_steps=pruning_dict.get("warmup_steps", 5)
        )
        
        # Create wandb config
        wandb_dict = config_dict.get("wandb", {})
        wandb_config = OptunaWandbConfig(
            enabled=wandb_dict.get("enabled", True),
            project=wandb_dict.get("project", "barygnn"),
            project_suffix=wandb_dict.get("project_suffix", "optuna"),
            extra_tags=wandb_dict.get("extra_tags", ["optuna"])
        )
        
        # Create output config
        output_dict = config_dict.get("output", {})
        output_config = OptunaOutputConfig(
            config_dir=output_dict.get("config_dir", "barygnn/config/optuna_configs"),
            logs_prefix=output_dict.get("logs_prefix", "optuna_logs")
        )
        
        # Create search space
        search_space = {}
        search_space_dict = config_dict.get("search_space", {})
        
        for param_name, param_spec in search_space_dict.items():
            param_type = param_spec.get("type")
            
            if param_type == "categorical":
                search_space[param_name] = OptunaSearchSpaceItem(
                    type=param_type,
                    choices=param_spec.get("choices", [])
                )
            elif param_type == "int":
                search_space[param_name] = OptunaSearchSpaceItem(
                    type=param_type,
                    low=param_spec.get("low"),
                    high=param_spec.get("high"),
                    step=param_spec.get("step", 1)
                )
            elif param_type == "float":
                search_space[param_name] = OptunaSearchSpaceItem(
                    type=param_type,
                    low=param_spec.get("low"),
                    high=param_spec.get("high"),
                    log=param_spec.get("log", False)
                )
        
        return cls(
            study=study_config,
            base_config=config_dict.get("base_config", "barygnn/config/default_config.yaml"),
            output=output_config,
            metric=metric_config,
            wandb=wandb_config,
            pruning=pruning_config,
            search_space=search_space
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        search_space_dict = {}
        for param_name, param_spec in self.search_space.items():
            if param_spec.type == "categorical":
                search_space_dict[param_name] = {
                    "type": param_spec.type,
                    "choices": param_spec.choices
                }
            elif param_spec.type == "int":
                search_space_dict[param_name] = {
                    "type": param_spec.type,
                    "low": param_spec.low,
                    "high": param_spec.high,
                    "step": param_spec.step
                }
            elif param_spec.type == "float":
                search_space_dict[param_name] = {
                    "type": param_spec.type,
                    "low": param_spec.low,
                    "high": param_spec.high,
                    "log": param_spec.log
                }
        
        return {
            "study": {
                "name": self.study.name,
                "direction": self.study.direction,
                "storage": self.study.storage,
                "n_trials": self.study.n_trials,
                "timeout": self.study.timeout,
                "n_workers": self.study.n_workers
            },
            "base_config": self.base_config,
            "output": {
                "config_dir": self.output.config_dir,
                "logs_prefix": self.output.logs_prefix
            },
            "metric": {
                "name": self.metric.name,
                "mode": self.metric.mode
            },
            "wandb": {
                "enabled": self.wandb.enabled,
                "project": self.wandb.project,
                "project_suffix": self.wandb.project_suffix,
                "extra_tags": self.wandb.extra_tags
            },
            "pruning": {
                "enabled": self.pruning.enabled,
                "pruner": self.pruning.pruner,
                "warmup_steps": self.pruning.warmup_steps
            },
            "search_space": search_space_dict
        }
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)