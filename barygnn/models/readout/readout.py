import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class Readout(nn.Module):
    """
    Readout module that generates graph embeddings from barycenter weights and codebook.
    Always uses combined approach with traditional graph pooling and barycentric pooling.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        codebook_size: int,
        standard_pooling_method: str = "global_mean_pool",
        readout_type: str = "weighted_mean",  # For barycentric part: "weighted_mean" or "concat"
        node_distributions: torch.Tensor = None,
        batch_idx: torch.Tensor = None,
    ):
        """
        Initialize the readout module.
        
        Args:
            hidden_dim: Dimension of node embeddings
            codebook_size: Number of atoms in the codebook
            standard_pooling_method: Traditional pooling method 
                                   ("global_add_pool", "global_mean_pool", "global_max_pool")
            readout_type: Barycentric readout method ("weighted_mean" or "concat")
            node_distributions: Node distributions [N, S, hidden_dim]
            batch_idx: Batch indices [N]
        """
        super(Readout, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.standard_pooling_method = standard_pooling_method
        self.readout_type = readout_type
        
        # Store node distributions and batch indices
        self.node_distributions = node_distributions
        self.batch_idx = batch_idx
        
        # Validate readout type
        valid_readout_types = ["weighted_mean", "concat"]
        if readout_type not in valid_readout_types:
            raise ValueError(f"Readout type {readout_type} not supported. Use {valid_readout_types}.")
        
        # Validate standard pooling method
        valid_pooling_methods = ["global_add_pool", "global_mean_pool", "global_max_pool"]
        if standard_pooling_method not in valid_pooling_methods:
            raise ValueError(f"Standard pooling method {standard_pooling_method} not supported. Use {valid_pooling_methods}.")
    
    def set_node_data(self, node_distributions: torch.Tensor, batch_idx: torch.Tensor):
        """
        Set node distributions and batch indices for combined readout.
        
        Args:
            node_distributions: Node distributions [N, S, hidden_dim]
            batch_idx: Batch indices [N]
        """
        self.node_distributions = node_distributions
        self.batch_idx = batch_idx
    
    def forward(self, barycenter_weights, codebook):
        """
        Generate graph embeddings from barycenter weights and codebook.
        Always combines barycentric pooling with traditional graph pooling.
        
        Args:
            barycenter_weights: Barycenter weights [batch_size, codebook_size]
            codebook: Codebook atoms [codebook_size, hidden_dim]
            
        Returns:
            graph_embeddings: Graph embeddings [batch_size, combined_dim]
        """
        batch_size = barycenter_weights.size(0)
        
        # Check if node data is available
        if self.node_distributions is None or self.batch_idx is None:
            raise ValueError("Node distributions and batch indices must be set for readout.")
        
        # Get barycentric embedding based on readout type
        if self.readout_type == "concat":
            # Concat: histogram_weights + flattened_codebook_atoms
            flattened_codebook = codebook.flatten()  # [codebook_size * hidden_dim]
            barycentric_emb = torch.cat([barycenter_weights, flattened_codebook.expand(batch_size, -1)], dim=1)
        elif self.readout_type == "weighted_mean":
            # Use weighted_mean for barycentric part
            barycentric_emb = torch.matmul(barycenter_weights, codebook)
        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}. Supported types: 'concat', 'weighted_mean'")
        
        # Get traditional pooling embedding
        # Flatten node distributions to treat as node features
        N, S, d = self.node_distributions.shape
        flattened_nodes = self.node_distributions.reshape(N * S, d)  # [N*S, hidden_dim]
        
        # Apply standard pooling method
        if self.standard_pooling_method == "global_add_pool":
            traditional_emb = global_add_pool(flattened_nodes, self.batch_idx.repeat_interleave(S))
        elif self.standard_pooling_method == "global_mean_pool":
            traditional_emb = global_mean_pool(flattened_nodes, self.batch_idx.repeat_interleave(S))
        elif self.standard_pooling_method == "global_max_pool":
            traditional_emb = global_max_pool(flattened_nodes, self.batch_idx.repeat_interleave(S))
        else:
            raise ValueError(f"Unknown standard pooling method: {self.standard_pooling_method}")
        
        # Concatenate barycentric and traditional embeddings
        graph_embeddings = torch.cat([barycentric_emb, traditional_emb], dim=1)
        
        return graph_embeddings 