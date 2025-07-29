import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple
import logging

from barygnn.models.encoders.base import BaseEncoder
from barygnn.models.pooling.barycentric_pooling import BarycentricPooling
from barygnn.models.readout.readout import Readout
from barygnn.models.classification.mlp import MLP

# Set up logger
logger = logging.getLogger(__name__)


class BaryGNN(nn.Module):
    """
    BaryGNN: A Graph Neural Network with Barycentric Pooling
    
    This model maps each node to an empirical distribution (set of vectors),
    computes a barycenter of these distributions using a learned codebook,
    and performs graph-level classification.
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        hidden_dim: int,
        codebook_size: int = 16,
        distribution_size: int = 32,
        readout_type: str = "weighted_mean",
        num_classes: int = 2,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iterations: int = 20,
        dropout: float = 0.5,
        debug_mode: bool = True,
        **kwargs
    ):
        """
        Initialize the BaryGNN model.
        
        Args:
            encoder: Graph encoder model (GNN)
            hidden_dim: Dimension of node embeddings
            codebook_size: Number of atoms in the barycenter codebook
            distribution_size: Number of vectors in each node's empirical distribution
            readout_type: Type of readout ("weighted_mean" or "concat")
            num_classes: Number of output classes
            sinkhorn_epsilon: Regularization parameter for Sinkhorn algorithm
            sinkhorn_iterations: Number of iterations for Sinkhorn algorithm
            dropout: Dropout rate for classification head
            debug_mode: Whether to print debugging information
        """
        super(BaryGNN, self).__init__()
        
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.distribution_size = distribution_size
        self.readout_type = readout_type
        self.debug_mode = debug_mode
        
        # Node distribution projector (maps node features to distribution samples)
        self.node_distribution_projector = nn.Linear(
            hidden_dim, hidden_dim * distribution_size
        )
        
        # Barycentric pooling with learned codebook
        self.pooling = BarycentricPooling(
            hidden_dim=hidden_dim,
            codebook_size=codebook_size,
            epsilon=sinkhorn_epsilon,
            num_iterations=sinkhorn_iterations,
            debug_mode=debug_mode
        )
        
        # Readout layer
        self.readout = Readout(
            hidden_dim=hidden_dim,
            codebook_size=codebook_size,
            readout_type=readout_type
        )
        
        # Classification head
        classifier_input_dim = hidden_dim if readout_type == "weighted_mean" else hidden_dim * codebook_size
        self.classifier = MLP(
            in_dim=classifier_input_dim,
            hidden_dim=hidden_dim,
            out_dim=num_classes,
            dropout=dropout
        )
    
    def _check_nan(self, tensor, name):
        """Check if tensor contains NaN values and log information."""
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            total_count = tensor.numel()
            logger.warning(f"NaN detected in {name}: {nan_count}/{total_count} values are NaN")
            return True
        return False
    
    def forward(self, batch):
        """
        Forward pass of the BaryGNN model.
        
        Args:
            batch: PyG batch containing graph data
            
        Returns:
            logits: Classification logits
        """
        # Get node embeddings from the encoder
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        # Debug input
        if self.debug_mode:
            self._check_nan(x, "input features")
        
        # Encode nodes
        node_embeddings = self.encoder(x, edge_index)
        
        # Debug node embeddings
        if self.debug_mode:
            self._check_nan(node_embeddings, "node_embeddings")
            if torch.isnan(node_embeddings).any():
                # Replace NaNs with zeros to continue
                node_embeddings = torch.nan_to_num(node_embeddings, nan=0.0)
        
        # Project node embeddings to distribution samples
        batch_size, num_nodes = batch_idx.max().item() + 1, node_embeddings.size(0)
        distribution_embeddings = self.node_distribution_projector(node_embeddings)
        
        # Debug distribution embeddings
        if self.debug_mode:
            self._check_nan(distribution_embeddings, "distribution_embeddings before reshape")
        
        # Reshape to [num_nodes, distribution_size, hidden_dim]
        distribution_embeddings = distribution_embeddings.view(
            num_nodes, self.distribution_size, self.hidden_dim
        )
        
        # Debug distribution embeddings after reshape
        if self.debug_mode:
            self._check_nan(distribution_embeddings, "distribution_embeddings after reshape")
            # Normalize distribution embeddings to prevent extreme values
            if torch.isnan(distribution_embeddings).any() or torch.max(torch.abs(distribution_embeddings)) > 1e6:
                logger.warning("Normalizing distribution embeddings due to extreme values")
                distribution_embeddings = F.normalize(distribution_embeddings.view(-1, self.hidden_dim), dim=1).view(
                    num_nodes, self.distribution_size, self.hidden_dim
                )
        
        # Apply barycentric pooling
        barycenter_weights = self.pooling(distribution_embeddings, batch_idx)
        
        # Debug barycenter weights
        if self.debug_mode:
            self._check_nan(barycenter_weights, "barycenter_weights")
            if torch.isnan(barycenter_weights).any():
                # Use uniform weights as fallback
                logger.warning("Using uniform weights as fallback for NaN barycenter weights")
                barycenter_weights = torch.ones(batch_size, self.codebook_size, device=barycenter_weights.device)
                barycenter_weights = barycenter_weights / self.codebook_size
        
        # Apply readout to get graph embeddings
        graph_embeddings = self.readout(barycenter_weights, self.pooling.codebook)
        
        # Debug graph embeddings
        if self.debug_mode:
            self._check_nan(graph_embeddings, "graph_embeddings")
            if torch.isnan(graph_embeddings).any():
                # Replace NaNs with zeros
                graph_embeddings = torch.nan_to_num(graph_embeddings, nan=0.0)
        
        # Apply classification head
        logits = self.classifier(graph_embeddings)
        
        # Debug logits
        if self.debug_mode:
            self._check_nan(logits, "logits")
            if torch.isnan(logits).any():
                # Replace NaNs with zeros
                logits = torch.ones_like(logits) / logits.size(1)  # Uniform distribution
        
        return logits 