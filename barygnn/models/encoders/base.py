import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseEncoder(nn.Module, ABC):
    """
    Base class for graph node encoders (GNNs).
    """
    
    def __init__(self):
        super(BaseEncoder, self).__init__()
    
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