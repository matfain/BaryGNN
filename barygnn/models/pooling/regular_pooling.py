import torch
import torch.nn as nn
import logging
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

logger = logging.getLogger(__name__)

class RegularPooling(nn.Module):
    """
    Regular graph pooling without optimal transport or codebook.
    Serves as a baseline comparison for BarycentricPooling.
    
    This implementation uses standard graph pooling methods from PyTorch Geometric
    to directly pool node distributions into graph-level embeddings.
    """
    
    def __init__(self,
                 hidden_dim: int,
                 standard_pooling_method: str = "global_mean_pool",  # "global_add_pool", "global_mean_pool", "global_max_pool"
                 backend: str = "regular_pooling",  # Ignored, just for compatibility
                 **kwargs):  # Accept but ignore other BarycentricPooling params
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.pooling_method = standard_pooling_method
        
        # Validate pooling method
        valid_pooling_methods = ["global_add_pool", "global_mean_pool", "global_max_pool"]
        if standard_pooling_method not in valid_pooling_methods:
            raise ValueError(f"Unknown pooling method: {standard_pooling_method}. Choose from: {valid_pooling_methods}")
        
        # No learnable parameters
        logger.info(f"Initialized RegularPooling with method: {standard_pooling_method}")
    
    def forward(self, node_distributions: torch.Tensor, batch_idx: torch.Tensor):
        """
        Apply standard graph pooling to node distributions.
        
        Args:
            node_distributions: [N, S, hidden_dim] - Node feature distributions
            batch_idx: [N] - Graph assignment for each node
            
        Returns:
            graph_embeddings: [B, hidden_dim] - Graph-level embeddings
        """
        N, S, d = node_distributions.shape
        
        # First, collapse the distribution dimension by mean pooling
        # This converts each node's distribution to a single vector
        node_embeddings = node_distributions.mean(dim=1)  # [N, hidden_dim]
        
        # Apply the specified pooling method to get graph embeddings
        if self.pooling_method == "global_add_pool":
            graph_embeddings = global_add_pool(node_embeddings, batch_idx)
        elif self.pooling_method == "global_mean_pool":
            graph_embeddings = global_mean_pool(node_embeddings, batch_idx)
        elif self.pooling_method == "global_max_pool":
            graph_embeddings = global_max_pool(node_embeddings, batch_idx)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        return graph_embeddings