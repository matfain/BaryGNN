import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Dict, Any, Optional
import logging

from barygnn.models.encoders.base import BaseEncoder
from barygnn.models.encoders.gin import GIN
from barygnn.models.encoders.sage import GraphSAGE

# Set up logger
logger = logging.getLogger(__name__)


class MultiHeadEncoder(nn.Module):
    """
    Multi-head encoder that generates multiple embeddings per node.
    
    Instead of using a single GNN and then projecting to multiple vectors,
    this encoder uses multiple GNN heads to learn truly distinct representations
    for each node, creating a rich empirical distribution.
    """
    
    def __init__(
        self,
        base_encoder_type: str,
        in_dim: int,
        hidden_dim: int,
        num_heads: int = 32,
        shared_layers: int = 1,
        **encoder_kwargs
    ):
        """
        Initialize the multi-head encoder.
        
        Args:
            base_encoder_type: Type of base encoder ("GIN" or "GraphSAGE")
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension for each head
            num_heads: Number of encoder heads (distribution_size)
            shared_layers: Number of initial shared layers before branching
            **encoder_kwargs: Additional arguments for base encoder
        """
        super(MultiHeadEncoder, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.shared_layers = shared_layers
        self.base_encoder_type = base_encoder_type
        
        logger.info(f"Initializing MultiHeadEncoder with {num_heads} heads, "
                   f"base_encoder={base_encoder_type}, shared_layers={shared_layers}")
        
        # Get the base encoder class
        if base_encoder_type == "GIN":
            encoder_class = GIN
        elif base_encoder_type == "GraphSAGE":
            encoder_class = GraphSAGE
        else:
            raise ValueError(f"Unknown encoder type: {base_encoder_type}")
        
        # Filter out multi-head specific parameters that shouldn't be passed to base encoder
        multi_head_params = {'shared_layers', 'multi_head_type', 'distribution_size'}
        base_encoder_kwargs = {k: v for k, v in encoder_kwargs.items() if k not in multi_head_params}
        
        # Shared initial layers (if any)
        if shared_layers > 0:
            self.shared_encoder = encoder_class(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=shared_layers,
                **{k: v for k, v in base_encoder_kwargs.items() if k != 'num_layers'}
            )
            head_in_dim = hidden_dim
        else:
            self.shared_encoder = None
            head_in_dim = in_dim
        
        # Individual encoder heads
        self.encoder_heads = nn.ModuleList()
        for i in range(num_heads):
            # Each head gets its own encoder with remaining layers
            remaining_layers = base_encoder_kwargs.get('num_layers', 3) - shared_layers
            if remaining_layers <= 0:
                remaining_layers = 1
                
            head_encoder = encoder_class(
                in_dim=head_in_dim,
                hidden_dim=hidden_dim,
                num_layers=remaining_layers,
                **{k: v for k, v in base_encoder_kwargs.items() if k != 'num_layers'}
            )
            self.encoder_heads.append(head_encoder)
        
        logger.debug(f"Created {len(self.encoder_heads)} encoder heads with "
                    f"{remaining_layers} layers each")
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass of the multi-head encoder.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            
        Returns:
            node_distributions: Multi-head embeddings [num_nodes, num_heads, hidden_dim]
        """
        # Apply shared layers if they exist
        if self.shared_encoder is not None:
            shared_features = self.shared_encoder(x, edge_index, edge_attr)
        else:
            shared_features = x
        
        # Apply each encoder head
        head_embeddings = []
        for i, head_encoder in enumerate(self.encoder_heads):
            head_embedding = head_encoder(shared_features, edge_index, edge_attr)
            head_embeddings.append(head_embedding)
        
        # Stack embeddings: [num_nodes, num_heads, hidden_dim]
        node_distributions = torch.stack(head_embeddings, dim=1)
        
        logger.debug(f"Generated node distributions with shape: {node_distributions.shape}")
        
        return node_distributions


class EfficientMultiHeadEncoder(nn.Module):
    """
    More efficient multi-head encoder that shares most parameters.
    
    This version uses a shared backbone and only has separate final layers
    for each head, reducing memory usage while still providing diversity.
    """
    
    def __init__(
        self,
        base_encoder_type: str,
        in_dim: int,
        hidden_dim: int,
        num_heads: int = 32,
        **encoder_kwargs
    ):
        """
        Initialize the efficient multi-head encoder.
        
        Args:
            base_encoder_type: Type of base encoder ("GIN" or "GraphSAGE")
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension for each head
            num_heads: Number of encoder heads (distribution_size)
            **encoder_kwargs: Additional arguments for base encoder
        """
        super(EfficientMultiHeadEncoder, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.base_encoder_type = base_encoder_type
        
        logger.info(f"Initializing EfficientMultiHeadEncoder with {num_heads} heads, "
                   f"base_encoder={base_encoder_type}")
        
        # Get the base encoder class
        if base_encoder_type == "GIN":
            encoder_class = GIN
        elif base_encoder_type == "GraphSAGE":
            encoder_class = GraphSAGE
        else:
            raise ValueError(f"Unknown encoder type: {base_encoder_type}")
        
        # Filter out multi-head specific parameters that shouldn't be passed to base encoder
        multi_head_params = {'shared_layers', 'multi_head_type', 'distribution_size'}
        base_encoder_kwargs = {k: v for k, v in encoder_kwargs.items() if k not in multi_head_params}
        
        # Shared backbone encoder
        self.backbone = encoder_class(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            **base_encoder_kwargs
        )
        
        # Individual projection heads
        self.projection_heads = nn.ModuleList()
        for i in range(num_heads):
            # Each head has a small MLP to create diversity
            head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.projection_heads.append(head)
        
        logger.debug(f"Created backbone encoder and {len(self.projection_heads)} projection heads")
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass of the efficient multi-head encoder.
        
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            
        Returns:
            node_distributions: Multi-head embeddings [num_nodes, num_heads, hidden_dim]
        """
        # Get shared backbone features
        backbone_features = self.backbone(x, edge_index, edge_attr)
        
        # Apply each projection head
        head_embeddings = []
        for head in self.projection_heads:
            head_embedding = head(backbone_features)
            head_embeddings.append(head_embedding)
        
        # Stack embeddings: [num_nodes, num_heads, hidden_dim]
        node_distributions = torch.stack(head_embeddings, dim=1)
        
        logger.debug(f"Generated node distributions with shape: {node_distributions.shape}")
        
        return node_distributions


def create_multi_head_encoder(
    encoder_type: str,
    multi_head_type: str = "full",
    **kwargs
) -> nn.Module:
    """
    Factory function to create multi-head encoders.
    
    Args:
        encoder_type: Base encoder type ("GIN" or "GraphSAGE")
        multi_head_type: Multi-head architecture type ("full" or "efficient")
        **kwargs: Additional arguments for the encoder
        
    Returns:
        Multi-head encoder instance
    """
    if multi_head_type == "full":
        return MultiHeadEncoder(encoder_type, **kwargs)
    elif multi_head_type == "efficient":
        return EfficientMultiHeadEncoder(encoder_type, **kwargs)
    else:
        raise ValueError(f"Unknown multi-head type: {multi_head_type}. "
                        f"Choose from ['full', 'efficient']") 