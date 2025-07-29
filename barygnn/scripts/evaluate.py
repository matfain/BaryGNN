import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
import json

from barygnn.config import Config
from barygnn.models import BaryGNN
from barygnn.models.encoders import GIN, GraphSAGE
from barygnn.data import load_dataset
from barygnn.utils import compute_metrics
from barygnn.scripts.train import create_model, evaluate


def main():
    """
    Main function.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate BaryGNN")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Path to output file")
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    # Print metrics
    print(f"Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save metrics
    if args.output is not None:
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(args.output, "w") as f:
            json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main() 