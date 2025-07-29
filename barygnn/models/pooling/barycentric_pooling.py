import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class BarycentricPooling(nn.Module):
    """
    Barycentric pooling module that computes the barycenter weights
    for node distributions using the Sinkhorn algorithm.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        codebook_size: int = 16,
        epsilon: float = 0.1,
        num_iterations: int = 20,
    ):
        """
        Initialize the barycentric pooling module.
        
        Args:
            hidden_dim: Dimension of node embeddings
            codebook_size: Number of atoms in the codebook
            epsilon: Regularization parameter for Sinkhorn algorithm
            num_iterations: Number of iterations for Sinkhorn algorithm
        """
        super(BarycentricPooling, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        
        # Initialize the codebook (atoms for barycenter)
        self.codebook = nn.Parameter(torch.randn(codebook_size, hidden_dim))
        nn.init.xavier_uniform_(self.codebook)
    
    def forward(self, node_distributions, batch_idx):
        """
        Compute barycenter weights for node distributions.
        
        Args:
            node_distributions: Node distribution samples [num_nodes, distribution_size, hidden_dim]
            batch_idx: Batch indices for nodes [num_nodes]
            
        Returns:
            barycenter_weights: Barycenter weights [batch_size, codebook_size]
        """
        num_nodes = node_distributions.size(0)
        distribution_size = node_distributions.size(1)
        batch_size = batch_idx.max().item() + 1
        
        # Initialize barycenter weights
        barycenter_weights = torch.zeros(batch_size, self.codebook_size, device=node_distributions.device)
        
        # Process each graph in the batch
        for b in range(batch_size):
            # Get nodes belonging to this graph
            mask = (batch_idx == b)
            graph_nodes = node_distributions[mask]  # [graph_num_nodes, distribution_size, hidden_dim]
            
            if graph_nodes.size(0) == 0:
                continue
            
            # Flatten node distributions
            flattened_nodes = graph_nodes.reshape(-1, self.hidden_dim)  # [graph_num_nodes * distribution_size, hidden_dim]
            
            # Compute cost matrix between node samples and codebook atoms
            cost_matrix = self._compute_cost_matrix(flattened_nodes, self.codebook)
            
            # Compute optimal transport weights using Sinkhorn algorithm
            weights = self._sinkhorn(cost_matrix)
            
            # Aggregate weights for the graph
            weights_sum = weights.sum(dim=0)  # [codebook_size]
            weights_sum = weights_sum / weights_sum.sum()  # Normalize
            
            barycenter_weights[b] = weights_sum
        
        return barycenter_weights
    
    def _compute_cost_matrix(self, x, y):
        """
        Compute pairwise squared Euclidean distances between points.
        
        Args:
            x: First set of points [n, d]
            y: Second set of points [m, d]
            
        Returns:
            cost_matrix: Pairwise distance matrix [n, m]
        """
        n = x.size(0)
        m = y.size(0)
        
        x_norm = torch.sum(x ** 2, dim=1).view(n, 1)
        y_norm = torch.sum(y ** 2, dim=1).view(1, m)
        
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.transpose(0, 1))
        dist = torch.clamp(dist, min=0.0)
        
        return dist
    
    def _sinkhorn(self, cost_matrix):
        """
        Sinkhorn algorithm for regularized optimal transport.
        
        Args:
            cost_matrix: Cost matrix [n, m]
            
        Returns:
            transport_matrix: Transport matrix [n, m]
        """
        n, m = cost_matrix.shape
        
        # Initialize marginals
        a = torch.ones(n, device=cost_matrix.device) / n
        b = torch.ones(m, device=cost_matrix.device) / m
        
        # Initialize kernel
        K = torch.exp(-cost_matrix / self.epsilon)
        
        # Sinkhorn iterations
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        
        for _ in range(self.num_iterations):
            u = a / (torch.matmul(K, v) + 1e-8)
            v = b / (torch.matmul(K.t(), u) + 1e-8)
        
        # Compute transport matrix
        transport_matrix = torch.diag(u) @ K @ torch.diag(v)
        
        return transport_matrix 