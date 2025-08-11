import os
import logging
import math
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import argparse

from barygnn import create_barygnn, Config, load_dataset, evaluate_model


def _select_metric_value(metric_name : str, val_loss: float, val_metrics: dict) -> float:
    """Return the scalar to optimize based on metric_name.

    Supported names (case-insensitive): 'accuracy', 'macro_f1', 'roc_auc', 'loss'.
    Defaults to 'accuracy' if unknown.
    """
    if metric_name == "loss":
        return float(val_loss)
    if metric_name in ("accuracy", "macro_f1", "roc_auc"):
        return float(val_metrics.get(metric_name, 0.0))
        
    return float(val_metrics.get("accuracy", 0.0))


def _is_maximize_metric(metric_name: str = "accuracy") -> bool:
    """Whether higher is better for the given metric."""
    return metric_name.lower() != "loss"


def _sanitize_selected(value: float, to_maximize: bool) -> float:
    """Guard against NaN/inf values so comparisons and schedulers behave."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float('-inf') if to_maximize else float('inf')
    return float(value)

def train_epoch(model, train_loader, optimizer, device, clip_grad_norm=None):
    logger = logging.getLogger(__name__)
    
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    total_classification_loss = 0
    total_regularization_loss = 0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        try:
            # Forward pass
            logits = model(batch)
            
            # Classification loss
            classification_loss = F.cross_entropy(logits, batch.y)
            
            # Distribution regularization loss
            regularization_loss = model.compute_regularization_loss()
            
            # Total loss
            total_loss_batch = classification_loss + regularization_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            
            # Accumulate losses and accuracy
            total_loss += total_loss_batch.item()
            total_classification_loss += classification_loss.item()
            total_regularization_loss += regularization_loss.item()
            
            # Compute accuracy
            pred = logits.argmax(dim=1)
            total_correct += (pred == batch.y).sum().item()
            total_samples += batch.y.size(0)
            
        except Exception as e:
            logger.error(f"Error in training batch {batch_idx}: {str(e)}")
            continue
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_classification_loss = total_classification_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_regularization_loss = total_regularization_loss / len(train_loader) if len(train_loader) > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_classification_loss, avg_regularization_loss, accuracy


def run_training(config: Config) -> None:
    logger = logging.getLogger(__name__)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = Config.from_yaml(args.config)
    selected_metric = (getattr(config.training, "metric", "accuracy") or "accuracy").lower()
    to_maximize = _is_maximize_metric(selected_metric)
    
    logger.info(f"Starting BaryGNN training: {config.experiment_type}")
    
    # Initialize Weights & Biases
    if config.wandb.enabled and config.wandb.api_key:
        os.environ['WANDB_API_KEY'] = config.wandb.api_key
        wandb.init(
            project=config.wandb.project,
            name=config.experiment_type,
            config=config.to_dict(),
            tags=config.wandb.tags,
            notes=config.wandb.notes,
        )
        logger.info("Weights & Biases initialized")
    else:
        logger.warning("W&B not configured, running without logging")
    
    # Set device
    device = torch.device(config.training.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Load dataset
    logger.info(f"Loading {config.data.name} dataset...")
    train_loader, val_loader, test_loader, num_features, num_classes = load_dataset(
        config.data.name, 
        batch_size=config.data.batch_size,
        split_seed=config.data.split_seed
    )
    
    # Update config with dataset info
    config.model.encoder.in_dim = num_features
    
    # Create model
    model_kwargs = config.get_model_kwargs()
    model_kwargs['in_dim'] = num_features
    model_kwargs['num_classes'] = num_classes
    
    model = create_barygnn(version=config.model.version, **model_kwargs).to(device)
    # gradient debugging: register hooks if needed
    # model.pooling.register_gradient_hooks()
    
    # Log model information
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        logger.info(f"Model created: {model_info}")
        logger.info(f"Total parameters: {model_info['total_parameters']:,}")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {total_params:,} parameters")
    
    # Create optimizer
    optimizer = Adam(
        model.parameters(), 
        lr=config.training.lr, 
        weight_decay=config.training.weight_decay
    )
    
    # Create scheduler
    if config.training.scheduler == "plateau":
        # Filter out any unsupported parameters for ReduceLROnPlateau
        valid_params = {}
        for key, value in config.training.scheduler_params.items():
            if key in ['mode', 'factor', 'patience', 'threshold', 'threshold_mode', 
                      'cooldown', 'min_lr', 'eps']:
                valid_params[key] = value
        # Align mode with the chosen metric direction unless explicitly set correctly
        forced_mode = 'max' if to_maximize else 'min'
        if valid_params.get('mode') != forced_mode:
            valid_params['mode'] = forced_mode
        scheduler = ReduceLROnPlateau(optimizer, **valid_params)
        logger.info(f"Created ReduceLROnPlateau scheduler with params: {valid_params} (selected_metric={selected_metric})")
    else:
        scheduler = None
    
    # Training loop
    best_val_score = float('-inf') if to_maximize else float('inf')
    patience_counter = 0
    
    logger.info("Starting training...")
    
    for epoch in range(config.training.num_epochs):
        # Training
        train_loss, train_class_loss, train_reg_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, 
            clip_grad_norm=config.training.gradient_clip
        )
        
        # Validation
        val_loss, val_metrics = evaluate_model(model, val_loader, device)
        val_accuracy = val_metrics['accuracy']
        selected_val = _sanitize_selected(_select_metric_value(selected_metric, val_loss, val_metrics), to_maximize)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(selected_val)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Logging
        logger.info(
            f"Epoch {epoch+1:3d}/{config.training.num_epochs} | "
            f"Train Loss: {train_loss:.4f} (Class: {train_class_loss:.4f}, Reg: {train_reg_loss:.4f}) | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f} | "
            f"Val {selected_metric}: {selected_val:.4f} | "
            f"LR: {current_lr:.6f}"
        )
        
        # Weights & Biases logging
        if config.wandb.enabled and 'wandb' in globals() and wandb.run is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/classification_loss': train_class_loss,
                'train/regularization_loss': train_reg_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_accuracy,
                'val/macro_f1': val_metrics['macro_f1'],
                'val/roc_auc': val_metrics['roc_auc'],
                f'val/{selected_metric}': selected_val,
                'learning_rate': current_lr,
            })
        
        # Save best model
        is_better = (selected_val > best_val_score) if to_maximize else (selected_val < best_val_score)
        if is_better:
            best_val_score = selected_val
            patience_counter = 0
            
            # Save model checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            
            # Include pooling backend to help with debugging
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_metric': best_val_score,
                'best_metric_name': selected_metric,
                'config': config.to_dict(),
                'pooling_backend': config.model.pooling.backend
            }, f'checkpoints/{config.experiment_type}_best.pt')
            
            logger.info(f"New best model saved with validation {selected_metric}: {best_val_score:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.training.patience:
            logger.info(f"Early stopping after {config.training.patience} epochs without improvement")
            break
    
    # Load best model for final evaluation
    checkpoint_path = f'checkpoints/{config.experiment_type}_best.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        logger.info(f"Loading best model from {checkpoint_path} (epoch {checkpoint['epoch']})")
        logger.info(f"Model config: {config.model.pooling.backend}, Checkpoint pooling: {checkpoint.get('pooling_backend', 'unknown')}")
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded best model for final evaluation")
    
    # Final evaluation on test set
    test_loss, test_metrics = evaluate_model(model, test_loader, device)
    
    logger.info("=== Final Results ===")
    logger.info(f"Best Validation {selected_metric}: {best_val_score:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"Test ROC AUC: {test_metrics['roc_auc']:.4f}")
    
    # Final W&B logging
    if config.wandb.enabled and 'wandb' in globals() and wandb.run is not None:
        wandb.log({
            f'final/best_val_{selected_metric}': best_val_score,
            'final/test_loss': test_loss,
            'final/test_accuracy': test_metrics['accuracy'],
            'final/test_macro_f1': test_metrics['macro_f1'],
            'final/test_roc_auc': test_metrics['roc_auc'],
        })
        wandb.finish()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BaryGNN v2 Configuration-Based Training')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--log_dir', type=str, help='Directory to save logs')
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.log_dir, 'logger.log'), mode='w')
        ],
        force=True  # Ensures our handlers are always used
    )
    
    run_training(args.config)