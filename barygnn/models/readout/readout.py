import torch
import torch.nn as nn
import torch.nn.functional as F


class Readout(nn.Module):
    """
    Readout module that generates graph embeddings from barycenter weights and codebook.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        codebook_size: int,
        readout_type: str = "weighted_mean",
    ):
        """
        Initialize the readout module.
        
        Args:
            hidden_dim: Dimension of node embeddings
            codebook_size: Number of atoms in the codebook
            readout_type: Type of readout ("weighted_mean" or "concat")
        """
        super(Readout, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.readout_type = readout_type
        
        if readout_type not in ["weighted_mean", "concat"]:
            raise ValueError(f"Readout type {readout_type} not supported. Use 'weighted_mean' or 'concat'.")
    
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
        """
        batch_size = barycenter_weights.size(0)
        
        if self.readout_type == "weighted_mean":
            # Weighted mean of codebook atoms
            # [batch_size, codebook_size] @ [codebook_size, hidden_dim] -> [batch_size, hidden_dim]
            graph_embeddings = torch.matmul(barycenter_weights, codebook)
            
        else:  # "concat"
            # Weighted codebook atoms, flattened
            # [batch_size, codebook_size, 1] * [1, codebook_size, hidden_dim] -> [batch_size, codebook_size, hidden_dim]
            weighted_atoms = barycenter_weights.unsqueeze(-1) * codebook.unsqueeze(0)
            
            # Flatten to [batch_size, codebook_size * hidden_dim]
            graph_embeddings = weighted_atoms.reshape(batch_size, -1)
        
        return graph_embeddings 