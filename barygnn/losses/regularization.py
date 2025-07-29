import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal


def distribution_variance_loss(node_distributions: torch.Tensor, lambda_reg: float = 0.01) -> torch.Tensor:
    """
    Compute variance regularization loss to keep node distribution vectors close.
    
    This loss penalizes high variance between the distribution_size vectors for each node,
    encouraging them to stay close while still allowing some diversity.
    
    Args:
        node_distributions: Node distribution samples [num_nodes, distribution_size, hidden_dim]
        lambda_reg: Regularization strength
        
    Returns:
        loss: Variance regularization loss (scalar)
    """
    # Compute mean across distribution dimension for each node
    mean_embedding = node_distributions.mean(dim=1, keepdim=True)  # [num_nodes, 1, hidden_dim]
    
    # Compute squared deviations from mean
    squared_deviations = (node_distributions - mean_embedding) ** 2  # [num_nodes, distribution_size, hidden_dim]
    
    # Sum across feature dimension and average across distribution and nodes
    variance = squared_deviations.sum(dim=2).mean()  # scalar
    
    return lambda_reg * variance


def distribution_centroid_loss(node_distributions: torch.Tensor, lambda_reg: float = 0.01) -> torch.Tensor:
    """
    Compute centroid regularization loss to prevent vectors from straying too far from their mean.
    
    This loss penalizes the L2 distance between each distribution vector and the centroid
    of that node's distribution.
    
    Args:
        node_distributions: Node distribution samples [num_nodes, distribution_size, hidden_dim]
        lambda_reg: Regularization strength
        
    Returns:
        loss: Centroid regularization loss (scalar)
    """
    # Compute mean across distribution dimension for each node
    mean_embedding = node_distributions.mean(dim=1, keepdim=True)  # [num_nodes, 1, hidden_dim]
    
    # Compute L2 distance to centroid
    centroid_distances = torch.norm(node_distributions - mean_embedding, dim=2)  # [num_nodes, distribution_size]
    
    # Average across all nodes and distribution samples
    centroid_dist = centroid_distances.mean()  # scalar
    
    return lambda_reg * centroid_dist


def distribution_coherence_loss(node_distributions: torch.Tensor, lambda_reg: float = 0.01) -> torch.Tensor:
    """
    Compute coherence regularization loss to encourage diversity while maintaining structure.
    
    This loss encourages the distribution vectors to be diverse (not identical) while
    still maintaining some coherence around a central representation.
    
    Args:
        node_distributions: Node distribution samples [num_nodes, distribution_size, hidden_dim]
        lambda_reg: Regularization strength
        
    Returns:
        loss: Coherence regularization loss (scalar)
    """
    num_nodes, distribution_size, hidden_dim = node_distributions.shape
    
    # Compute pairwise distances within each node's distribution
    # Reshape to [num_nodes * distribution_size, hidden_dim]
    flat_distributions = node_distributions.view(-1, hidden_dim)
    
    # Create indices to compute pairwise distances within each node
    coherence_loss = 0.0
    for i in range(num_nodes):
        start_idx = i * distribution_size
        end_idx = (i + 1) * distribution_size
        node_vectors = flat_distributions[start_idx:end_idx]  # [distribution_size, hidden_dim]
        
        # Compute pairwise distances
        pairwise_dists = torch.cdist(node_vectors, node_vectors, p=2)  # [distribution_size, distribution_size]
        
        # Remove diagonal (distance to self)
        mask = ~torch.eye(distribution_size, dtype=torch.bool, device=node_vectors.device)
        pairwise_dists = pairwise_dists[mask]
        
        # Add penalty for vectors being too similar (small distances)
        # and reward for having some diversity (not too large distances)
        min_dist = 0.1  # Minimum desired distance
        max_dist = 2.0  # Maximum desired distance
        
        # Penalty for being too close
        too_close_penalty = F.relu(min_dist - pairwise_dists).mean()
        
        # Penalty for being too far
        too_far_penalty = F.relu(pairwise_dists - max_dist).mean()
        
        coherence_loss += too_close_penalty + too_far_penalty
    
    coherence_loss /= num_nodes
    return lambda_reg * coherence_loss


def compute_distribution_regularization(
    node_distributions: torch.Tensor,
    reg_type: Literal["variance", "centroid", "coherence"] = "variance",
    lambda_reg: float = 0.01
) -> torch.Tensor:
    """
    Compute distribution regularization loss with different strategies.
    
    Args:
        node_distributions: Node distribution samples [num_nodes, distribution_size, hidden_dim]
        reg_type: Type of regularization ("variance", "centroid", "coherence")
        lambda_reg: Regularization strength
        
    Returns:
        loss: Regularization loss (scalar)
    """
    if reg_type == "variance":
        return distribution_variance_loss(node_distributions, lambda_reg)
    elif reg_type == "centroid":
        return distribution_centroid_loss(node_distributions, lambda_reg)
    elif reg_type == "coherence":
        return distribution_coherence_loss(node_distributions, lambda_reg)
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}. "
                        f"Choose from ['variance', 'centroid', 'coherence']")


class DistributionRegularizer(nn.Module):
    """
    Module wrapper for distribution regularization losses.
    """
    
    def __init__(
        self,
        reg_type: Literal["variance", "centroid", "coherence"] = "variance",
        lambda_reg: float = 0.01
    ):
        """
        Initialize the distribution regularizer.
        
        Args:
            reg_type: Type of regularization ("variance", "centroid", "coherence")
            lambda_reg: Regularization strength
        """
        super().__init__()
        self.reg_type = reg_type
        self.lambda_reg = lambda_reg
    
    def forward(self, node_distributions: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss.
        
        Args:
            node_distributions: Node distribution samples [num_nodes, distribution_size, hidden_dim]
            
        Returns:
            loss: Regularization loss (scalar)
        """
        return compute_distribution_regularization(
            node_distributions, 
            self.reg_type, 
            self.lambda_reg
        ) 