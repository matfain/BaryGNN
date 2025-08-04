"""
BaryGNN: A Graph Neural Network with Barycentric Pooling

This package implements a novel graph neural network architecture that uses
barycentric pooling with optimal transport to generate graph-level embeddings 
for classification tasks.
"""

__version__ = "0.1.0"

from .models import BaryGNN, create_barygnn
from .config import Config
from .datasets import load_dataset
from .utils import compute_metrics, Logger
from .scripts.evaluate import evaluate_model

__all__ = ["BaryGNN", "create_barygnn", "Config", "load_dataset", "compute_metrics", "Logger", "evaluate_model"]
