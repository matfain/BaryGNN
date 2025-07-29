import torch
import numpy as np
from torch_geometric.datasets import TUDataset, OGBG
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional


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
        # OGB dataset
        dataset = OGBG(name=name, root="data")
        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        
        num_features = dataset.num_features
        num_classes = dataset.num_classes
    else:
        # TU dataset
        transform = NormalizeFeatures()
        dataset = TUDataset(root="data", name=name, transform=transform)
        
        # Get number of features and classes
        num_features = dataset.num_features
        num_classes = dataset.num_classes if hasattr(dataset, "num_classes") else int(dataset.data.y.max().item()) + 1
        
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