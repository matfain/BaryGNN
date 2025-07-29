import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import yaml
import time
from pathlib import Path

from barygnn.config import Config
from barygnn.models import BaryGNN
from barygnn.models.encoders import GIN, GraphSAGE
from barygnn.data import load_dataset
from barygnn.utils import compute_metrics, Logger


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        loader: Data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        
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
        loss = criterion(logits, batch.y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        total_loss += loss.item() * batch.num_graphs
        
        # Store predictions and targets
        y_true = batch.y.cpu()
        y_score = torch.softmax(logits, dim=1).detach().cpu()
        y_pred = torch.argmax(y_score, dim=1)
        
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        y_score_list.append(y_score)
    
    # Concatenate predictions and targets
    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)
    y_score = torch.cat(y_score_list)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["loss"] = total_loss / len(loader.dataset)
    
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
            loss = criterion(logits, batch.y)
            
            # Compute metrics
            total_loss += loss.item() * batch.num_graphs
            
            # Store predictions and targets
            y_true = batch.y.cpu()
            y_score = torch.softmax(logits, dim=1).detach().cpu()
            y_pred = torch.argmax(y_score, dim=1)
            
            y_true_list.append(y_true)
            y_pred_list.append(y_pred)
            y_score_list.append(y_score)
    
    # Concatenate predictions and targets
    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)
    y_score = torch.cat(y_score_list)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["loss"] = total_loss / len(loader.dataset)
    
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
    )
    
    return model


def train(config: Config) -> None:
    """
    Train the model.
    
    Args:
        config: Configuration
    """
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Set device
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    train_loader, val_loader, test_loader, num_features, num_classes = load_dataset(
        name=config.data.name,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        split_seed=config.data.split_seed,
    )
    
    # Create model
    model = create_model(config, num_features, num_classes)
    model = model.to(device)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    
    # Create logger
    logger = Logger(
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
    
    for epoch in range(1, config.training.num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
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
        
        logger.log(metrics, step=epoch)
        
        # Check for early stopping
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
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    checkpoint = torch.load(save_dir / "best_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    # Log test metrics
    logger.log(
        {
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "best_epoch": best_epoch,
            "best_val_metric": best_val_metric,
        }
    )
    
    if "roc_auc" in test_metrics:
        logger.log({"test_roc_auc": test_metrics["roc_auc"]})
    
    # Close logger
    logger.close()


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