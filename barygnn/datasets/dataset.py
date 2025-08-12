import torch
import numpy as np
import logging
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures, OneHotDegree, Compose
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional

# Import OGB datasets if available
try:
    from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
    HAS_OGB = True
except ImportError:
    HAS_OGB = False

logger = logging.getLogger(__name__)


def load_dataset(
    name: str,
    batch_size: int,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Load a graph dataset and split it into train, validation, and test sets.
    
    Args:
        name: Dataset name
        batch_size: Batch size
        num_workers: Number of workers for data loading
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        split_seed: Random seed for splitting
        
    Returns:
        train_loader: Training set data loader
        val_loader: Validation set data loader
        test_loader: Test set data loader
        num_features: Number of node features
        num_classes: Number of classes
    """
    # Set random seed for reproducibility
    np.random.seed(split_seed)
    torch.manual_seed(split_seed)
    
    # Load dataset
    if name.startswith("ogbg"):
        # Check if OGB is installed
        if not HAS_OGB:
            raise ImportError(
                "OGB package is required to use OGB datasets. "
                "Install it with: pip install ogb"
            )
            
        # OGB dataset
        logger.info(f"Loading OGB dataset: {name}")
        dataset = PygGraphPropPredDataset(name=name, root="data")
        split_idx = dataset.get_split_idx()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        
        # Get dataset information
        num_features = dataset[0].x.shape[1] if hasattr(dataset[0], 'x') and dataset[0].x is not None else 0
        num_classes = dataset.num_classes
        
        logger.info(f"OGB Dataset {name}: {len(dataset)} graphs, {num_features} features, {num_classes} classes")
    else:
        # TU dataset
        # Special handling for IMDB datasets which have no node features
        if name in ["IMDB-BINARY", "IMDB-MULTI"]:
            logger.info(f"Using OneHotDegree transform for {name} dataset")
            
            # First, load the dataset without transforms to calculate max degree
            temp_dataset = TUDataset(root="data", name=name, transform=None)
            
            # Calculate max degree across all graphs
            max_degree = 0
            for data in temp_dataset:
                if hasattr(data, 'edge_index') and data.edge_index is not None:
                    # Count unique source nodes for each target node
                    src, dst = data.edge_index
                    degrees = torch.bincount(dst)
                    if degrees.numel() > 0:
                        max_degree = max(max_degree, degrees.max().item())
            
            # Add 1 to max_degree to include degree 0
            max_degree = int(max_degree) + 1
            logger.info(f"Calculated max degree for {name}: {max_degree}")
            
            # Use OneHotDegree transform - no need to normalize one-hot vectors
            transform = OneHotDegree(max_degree)
        else:
            transform = NormalizeFeatures()
            
        dataset = TUDataset(root="data", name=name, transform=transform)
        
        # Get number of features and classes
        num_features = dataset.num_features
        num_classes = dataset.num_classes if hasattr(dataset, "num_classes") else int(dataset.data.y.max().item()) + 1
        
        logger.info(f"Dataset {name}: {len(dataset)} graphs, {num_features} features, {num_classes} classes")
        
        # Split dataset
        num_samples = len(dataset)
        indices = np.random.permutation(num_samples)
        
        test_size = int(num_samples * test_ratio)
        val_size = int(num_samples * val_ratio)
        train_size = num_samples - test_size - val_size
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_dataset = dataset[train_indices]
        val_dataset = dataset[val_indices]
        test_dataset = dataset[test_indices]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader, num_features, num_classes 