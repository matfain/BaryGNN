import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class Readout(nn.Module):
    """
    Readout module that generates graph embeddings from barycenter weights and codebook.
    Supports traditional graph pooling methods combined with barycentric pooling.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        codebook_size: int,
        readout_type: str = "weighted_mean",
        combined_readout: str = None,
        barycentric_readout: str = "weighted_mean",  # For combined readout: "weighted_mean" or "concat"
        node_distributions: torch.Tensor = None,
        batch_idx: torch.Tensor = None,
    ):
        """
        Initialize the readout module.
        
        Args:
            hidden_dim: Dimension of node embeddings
            codebook_size: Number of atoms in the codebook
            readout_type: Type of readout ("weighted_mean", "concat", or "combined")
            combined_readout: Traditional pooling method for combined readout 
                            ("global_add_pool", "global_mean_pool", "global_max_pool")
            barycentric_readout: Barycentric readout method for combined readout
                               ("weighted_mean" or "concat")
            node_distributions: Node distributions for combined readout [N, S, hidden_dim]
            batch_idx: Batch indices for combined readout [N]
        """
        super(Readout, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.readout_type = readout_type
        self.combined_readout = combined_readout
        self.barycentric_readout = barycentric_readout
        
        # Store node distributions and batch indices for combined readout
        self.node_distributions = node_distributions
        self.batch_idx = batch_idx
        
        # Validate readout type
        valid_types = ["weighted_mean", "concat", "combined"]
        if readout_type not in valid_types:
            raise ValueError(f"Readout type {readout_type} not supported. Use {valid_types}.")
        
        # Validate combined readout method
        if readout_type == "combined":
            valid_combined = ["global_add_pool", "global_mean_pool", "global_max_pool"]
            if combined_readout not in valid_combined:
                raise ValueError(f"Combined readout {combined_readout} not supported. Use {valid_combined}.")
            
            valid_barycentric = ["weighted_mean", "concat"]
            if barycentric_readout not in valid_barycentric:
                raise ValueError(f"Barycentric readout {barycentric_readout} not supported. Use {valid_barycentric}.")
    
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
        
        Args:
            barycenter_weights: Barycenter weights [batch_size, codebook_size]
            codebook: Codebook atoms [codebook_size, hidden_dim]
            
        Returns:
            graph_embeddings: Graph embeddings
                - If readout_type is "weighted_mean": [batch_size, hidden_dim]
                - If readout_type is "concat": [batch_size, codebook_size * hidden_dim]
                - If readout_type is "combined": [batch_size, combined_dim]
        """
        batch_size = barycenter_weights.size(0)
        
        if self.readout_type == "weighted_mean":
            # Weighted mean of codebook atoms
            # [batch_size, codebook_size] @ [codebook_size, hidden_dim] -> [batch_size, hidden_dim]
            graph_embeddings = torch.matmul(barycenter_weights, codebook)
            
        elif self.readout_type == "concat":
            # Concatenate histogram weights with flattened codebook atoms
            # histogram_weights: [batch_size, codebook_size]
            # codebook: [codebook_size, hidden_dim]
            # flattened_codebook: [codebook_size * hidden_dim]
            flattened_codebook = codebook.flatten()  # [codebook_size * hidden_dim]
            
            # Concatenate histogram weights with flattened codebook atoms
            # [batch_size, codebook_size] + [codebook_size * hidden_dim] -> [batch_size, codebook_size + codebook_size * hidden_dim]
            graph_embeddings = torch.cat([barycenter_weights, flattened_codebook.expand(batch_size, -1)], dim=1)
            
        elif self.readout_type == "combined":
            # Combine barycentric pooling with traditional graph pooling
            if self.node_distributions is None or self.batch_idx is None:
                raise ValueError("Node distributions and batch indices must be set for combined readout.")
            
            # Get barycentric embedding (weighted_mean or concat)
            if self.barycentric_readout == "concat":
                # Concat: histogram_weights + flattened_codebook_atoms
                flattened_codebook = codebook.flatten()  # [codebook_size * hidden_dim]
                barycentric_emb = torch.cat([barycenter_weights, flattened_codebook.expand(batch_size, -1)], dim=1)
            else:  # weighted_mean
                # Use weighted_mean for barycentric part
                barycentric_emb = torch.matmul(barycenter_weights, codebook)
            
            # Get traditional pooling embedding
            # Flatten node distributions to treat as node features
            N, S, d = self.node_distributions.shape
            flattened_nodes = self.node_distributions.reshape(N * S, d)  # [N*S, hidden_dim]
            
            # Apply traditional pooling
            if self.combined_readout == "global_add_pool":
                traditional_emb = global_add_pool(flattened_nodes, self.batch_idx.repeat_interleave(S))
            elif self.combined_readout == "global_mean_pool":
                traditional_emb = global_mean_pool(flattened_nodes, self.batch_idx.repeat_interleave(S))
            elif self.combined_readout == "global_max_pool":
                traditional_emb = global_max_pool(flattened_nodes, self.batch_idx.repeat_interleave(S))
            else:
                raise ValueError(f"Unknown combined readout: {self.combined_readout}")
            
            # Concatenate barycentric and traditional embeddings
            graph_embeddings = torch.cat([barycentric_emb, traditional_emb], dim=1)
        
        return graph_embeddings 