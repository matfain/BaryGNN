import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple

from barygnn.models.encoders.base import BaseEncoder
from barygnn.models.pooling.barycentric_pooling import BarycentricPooling
from barygnn.models.readout.readout import Readout
from barygnn.models.classification.mlp import MLP


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
        """
        super(BaryGNN, self).__init__()
        
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.distribution_size = distribution_size
        self.readout_type = readout_type
        
        # Node distribution projector (maps node features to distribution samples)
        self.node_distribution_projector = nn.Linear(
            hidden_dim, hidden_dim * distribution_size
        )
        
        # Barycentric pooling with learned codebook
        self.pooling = BarycentricPooling(
            hidden_dim=hidden_dim,
            codebook_size=codebook_size,
            epsilon=sinkhorn_epsilon,
            num_iterations=sinkhorn_iterations
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
        node_embeddings = self.encoder(x, edge_index)
        
        # Project node embeddings to distribution samples
        batch_size, num_nodes = batch_idx.max().item() + 1, node_embeddings.size(0)
        distribution_embeddings = self.node_distribution_projector(node_embeddings)
        distribution_embeddings = distribution_embeddings.view(
            num_nodes, self.distribution_size, self.hidden_dim
        )
        
        # Apply barycentric pooling
        barycenter_weights = self.pooling(distribution_embeddings, batch_idx)
        
        # Apply readout to get graph embeddings
        graph_embeddings = self.readout(barycenter_weights, self.pooling.codebook)
        
        # Apply classification head
        logits = self.classifier(graph_embeddings)
        
        return logits 