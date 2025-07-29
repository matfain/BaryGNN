import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multi-layer perceptron for classification.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        """
        Initialize the MLP.
        
        Args:
            in_dim: Input dimension
            hidden_dim: Hidden dimension
            out_dim: Output dimension (number of classes)
            num_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super(MLP, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
    
    def forward(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x: Input features [batch_size, in_dim]
            
        Returns:
            logits: Classification logits [batch_size, out_dim]
        """
        # Apply hidden layers with ReLU, BatchNorm, and dropout
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply output layer
        logits = self.output_layer(x)
        
        return logits 