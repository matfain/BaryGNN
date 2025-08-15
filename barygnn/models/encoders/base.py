import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEncoder(nn.Module, ABC):
    """
    Base class for graph node encoders (GNNs).
    """
    
    def __init__(self, 
                 in_dim: int = 0,
                 hidden_dim: int = 0,
                 use_categorical_encoding: bool = False,
                 categorical_embed_dim: int = 64):
        """
        Initialize the base encoder.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension
            use_categorical_encoding: Whether to use categorical encoding for molecular features
            categorical_embed_dim: Embedding dimension for categorical features
        """
        super(BaseEncoder, self).__init__()
        self.use_categorical_encoding = use_categorical_encoding
        self.categorical_embed_dim = categorical_embed_dim
        
        # Initialize atom encoder if categorical encoding is enabled
        if use_categorical_encoding:
            from ogb.graphproppred.mol_encoder import AtomEncoder
            self.atom_encoder = AtomEncoder(emb_dim=categorical_embed_dim)
        else:
            self.atom_encoder = None
    
    def _preprocess_features(self, x):
        """Helper method to apply categorical encoding if enabled."""
        if self.use_categorical_encoding and self.atom_encoder is not None:
            # Ensure input is long type for categorical encoding
            if x.dtype in [torch.float32, torch.float64]:
                x = x.long()
            return self.atom_encoder(x)
        return x
    
    @abstractmethod
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass of the encoder.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            
        Returns:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
        """
        pass 