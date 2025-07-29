import os
import logging
import wandb
from typing import Dict, Any, Optional, Union


class Logger:
    """
    Logger utility with Weights & Biases integration.
    """
    
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        wandb_config: Dict[str, Any],
        log_dir: str = "logs",
    ):
        """
        Initialize the logger.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
            wandb_config: Weights & Biases configuration
            log_dir: Directory for log files
        """
        self.experiment_name = experiment_name
        self.config = config
        self.wandb_config = wandb_config
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up file logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{experiment_name}.log"))
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize Weights & Biases
        self.use_wandb = wandb_config.get("enabled", False)
        if self.use_wandb:
            # Set API key if provided
            api_key = wandb_config.get("api_key")
            if api_key is not None:
                os.environ["WANDB_API_KEY"] = api_key
            
            # Initialize wandb
            wandb.init(
                project=wandb_config.get("project", "BaryGNN"),
                entity=wandb_config.get("entity"),
                name=experiment_name,
                config=config,
            )
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step (optional)
        """
        # Log to file and console
        metrics_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        step_str = f" (Step {step})" if step is not None else ""
        self.logger.info(f"Metrics{step_str}: {metrics_str}")
        
        # Log to wandb
        if self.use_wandb:
            if step is not None:
                wandb.log(metrics, step=step)
            else:
                wandb.log(metrics)
    
    def close(self) -> None:
        """
        Close the logger.
        """
        # Close wandb
        if self.use_wandb:
            wandb.finish()
        
        # Close file handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler) 