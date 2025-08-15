import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from barygnn.models.encoders.base import BaseEncoder


class GraphSAGE(BaseEncoder):
    """
    GraphSAGE encoder.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        aggr: str = "mean",
        use_categorical_encoding: bool = False,
        categorical_embed_dim: int = 64,
    ):
        """
        Initialize GraphSAGE encoder.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            aggr: Aggregation method ("mean", "sum", "max")
            use_categorical_encoding: Whether to use categorical encoding for molecular features
            categorical_embed_dim: Embedding dimension for categorical features
        """
        super(GraphSAGE, self).__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            use_categorical_encoding=use_categorical_encoding,
            categorical_embed_dim=categorical_embed_dim
        )
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Determine actual input dimension based on categorical encoding
        actual_in_dim = categorical_embed_dim if use_categorical_encoding else in_dim
        
        # Input projection
        self.input_proj = nn.Linear(actual_in_dim, hidden_dim)
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass of the GraphSAGE encoder.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            
        Returns:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
        """
        # Apply categorical encoding if enabled
        x = self._preprocess_features(x)
        
        # Project input features
        x = self.input_proj(x)
        
        # Apply GraphSAGE layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x 