"""
BaryGNN: A Graph Neural Network with Barycentric Pooling

This package implements a novel graph neural network architecture that uses
barycentric pooling with optimal transport to generate graph-level embeddings 
for classification tasks.
"""

__version__ = "0.1.0"

from barygnn.models import BaryGNN, create_barygnn
from barygnn.config import Config
from barygnn.data import load_dataset
from barygnn.utils import compute_metrics, Logger

__all__ = ["BaryGNN", "create_barygnn", "Config", "load_dataset", "compute_metrics", "Logger"]
