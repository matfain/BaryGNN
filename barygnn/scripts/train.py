import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import yaml
import time
import logging
from pathlib import Path

from barygnn.config import Config
from barygnn.models import BaryGNN
from barygnn.models.encoders import GIN, GraphSAGE
from barygnn.data import load_dataset
from barygnn.utils import compute_metrics, Logger

# Set up logger
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad_norm: float = 1.0,
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        loader: Data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        clip_grad_norm: Maximum norm for gradient clipping
        
    Returns:
        metrics: Dictionary of metrics
    """
    model.train()
    
    total_loss = 0
    y_true_list = []
    y_pred_list = []
    y_score_list = []
    
    for batch in loader:
        # Move batch to device
        batch = batch.to(device)
        
        # Forward pass
        logits = model(batch)
        
        # Check for NaN in logits
        if torch.isnan(logits).any():
            logger.warning("NaN detected in logits during training, skipping batch")
            continue
            
        # Compute loss
        try:
            loss = criterion(logits, batch.y)
            
            # Check for NaN in loss
            if torch.isnan(loss).item():
                logger.warning("NaN loss detected during training, skipping batch")
                continue
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            
            # Compute metrics
            total_loss += loss.item() * batch.num_graphs
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            continue
        
        # Store predictions and targets
        y_true = batch.y.cpu()
        y_score = torch.softmax(logits, dim=1).detach().cpu()
        y_pred = torch.argmax(y_score, dim=1)
        
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        y_score_list.append(y_score)
    
    # Concatenate predictions and targets
    if len(y_true_list) == 0:
        logger.error("No valid batches during training")
        return {"loss": float('nan'), "accuracy": 0.0, "macro_f1": 0.0}
    
    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)
    y_score = torch.cat(y_score_list)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["loss"] = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else float('nan')
    
    return metrics


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate the model.
    
    Args:
        model: Model to evaluate
        loader: Data loader
        criterion: Loss function
        device: Device to use
        
    Returns:
        metrics: Dictionary of metrics
    """
    model.eval()
    
    total_loss = 0
    y_true_list = []
    y_pred_list = []
    y_score_list = []
    
    with torch.no_grad():
        for batch in loader:
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            logits = model(batch)
            
            # Check for NaN in logits
            if torch.isnan(logits).any():
                logger.warning("NaN detected in logits during evaluation, skipping batch")
                continue
                
            # Compute loss
            try:
                loss = criterion(logits, batch.y)
                
                # Check for NaN in loss
                if torch.isnan(loss).item():
                    logger.warning("NaN loss detected during evaluation, skipping batch")
                    continue
                    
                # Compute metrics
                total_loss += loss.item() * batch.num_graphs
                
            except Exception as e:
                logger.error(f"Error during evaluation: {str(e)}")
                continue
            
            # Store predictions and targets
            y_true = batch.y.cpu()
            y_score = torch.softmax(logits, dim=1).detach().cpu()
            y_pred = torch.argmax(y_score, dim=1)
            
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            y_score_list.append(y_score)
    
    # Concatenate predictions and targets
    if len(y_true_list) == 0:
        logger.error("No valid batches during evaluation")
        return {"loss": float('nan'), "accuracy": 0.0, "macro_f1": 0.0}
    
    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)
    y_score = torch.cat(y_score_list)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["loss"] = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else float('nan')
    
    return metrics


def create_model(config: Config, num_features: int, num_classes: int) -> nn.Module:
    """
    Create a model based on the configuration.
    
    Args:
        config: Configuration
        num_features: Number of input features
        num_classes: Number of output classes
        
    Returns:
        model: Model
    """
    # Update encoder input dimension
    config.model.encoder.in_dim = num_features
    
    # Create encoder
    if config.model.encoder.type == "GIN":
        encoder = GIN(
            in_dim=config.model.encoder.in_dim,
            hidden_dim=config.model.encoder.hidden_dim,
            num_layers=config.model.encoder.num_layers,
            dropout=config.model.encoder.dropout,
        )
    elif config.model.encoder.type == "GraphSAGE":
        encoder = GraphSAGE(
            in_dim=config.model.encoder.in_dim,
            hidden_dim=config.model.encoder.hidden_dim,
            num_layers=config.model.encoder.num_layers,
            dropout=config.model.encoder.dropout,
            aggr=config.model.encoder.aggr,
        )
    else:
        raise ValueError(f"Encoder type {config.model.encoder.type} not supported")
    
    # Create BaryGNN model
    model = BaryGNN(
        encoder=encoder,
        hidden_dim=config.model.hidden_dim,
        codebook_size=config.model.pooling.codebook_size,
        distribution_size=config.model.pooling.distribution_size,
        readout_type=config.model.readout_type,
        num_classes=num_classes,
        sinkhorn_epsilon=config.model.pooling.epsilon,
        sinkhorn_iterations=config.model.pooling.num_iterations,
        dropout=config.model.classification.dropout,
        debug_mode=True,  # Enable debug mode to track NaNs
    )
    
    return model


def train(config: Config) -> None:
    """
    Train the model.
    
    Args:
        config: Configuration
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/{config.experiment_name}_debug.log"),
            logging.StreamHandler()
        ]
    )
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Set device
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {config.data.name}")
    train_loader, val_loader, test_loader, num_features, num_classes = load_dataset(
        name=config.data.name,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        split_seed=config.data.split_seed,
    )
    
    logger.info(f"Dataset loaded: {num_features} features, {num_classes} classes")
    logger.info(f"Train: {len(train_loader.dataset)} graphs, Val: {len(val_loader.dataset)} graphs, Test: {len(test_loader.dataset)} graphs")
    
    # Create model
    model = create_model(config, num_features, num_classes)
    model = model.to(device)
    logger.info(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    
    # Create logger
    wandb_logger = Logger(
        experiment_name=config.experiment_name,
        config=config.to_dict(),
        wandb_config=config.wandb.__dict__,
    )
    
    # Create directory for saving models
    save_dir = Path(f"checkpoints/{config.experiment_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    best_val_metric = 0
    best_epoch = 0
    patience_counter = 0
    
    logger.info(f"Starting training for {config.training.num_epochs} epochs...")
    
    for epoch in range(1, config.training.num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            clip_grad_norm=1.0,  # Add gradient clipping
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Log metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        
        if "roc_auc" in train_metrics:
            metrics["train_roc_auc"] = train_metrics["roc_auc"]
        if "roc_auc" in val_metrics:
            metrics["val_roc_auc"] = val_metrics["roc_auc"]
        
        wandb_logger.log(metrics, step=epoch)
        
        logger.info(f"Epoch {epoch}/{config.training.num_epochs}: "
                   f"Train Loss: {train_metrics['loss']:.4f}, "
                   f"Train Acc: {train_metrics['accuracy']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Check for early stopping
        if not np.isnan(val_metrics[config.training.metric]):
            current_val_metric = val_metrics[config.training.metric]
            
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    save_dir / "best_model.pt",
                )
            else:
                patience_counter += 1
                if patience_counter >= config.training.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
    
    # Load best model
    if os.path.exists(save_dir / "best_model.pt"):
        checkpoint = torch.load(save_dir / "best_model.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    else:
        logger.warning("No best model found, using current model")
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    # Log test metrics
    test_log = {
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
    }
    
    if "roc_auc" in test_metrics:
        test_log["test_roc_auc"] = test_metrics["roc_auc"]
    
    wandb_logger.log(test_log)
    
    logger.info("\nTraining completed!")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best validation {config.training.metric}: {best_val_metric:.4f}")
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test macro F1: {test_metrics['macro_f1']:.4f}")
    
    if "roc_auc" in test_metrics:
        logger.info(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
    
    # Close logger
    wandb_logger.close()


def main():
    """
    Main function.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train BaryGNN")
    parser.add_argument("--config", type=str, default="barygnn/config/default_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Train
    train(config)


if __name__ == "__main__":
    main() 