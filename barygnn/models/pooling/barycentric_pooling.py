import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

# Set up logger
logger = logging.getLogger(__name__)


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
        stability_eps: float = 1e-8,
        debug_mode: bool = False,
    ):
        """
        Initialize the barycentric pooling module.
        
        Args:
            hidden_dim: Dimension of node embeddings
            codebook_size: Number of atoms in the codebook
            epsilon: Regularization parameter for Sinkhorn algorithm
            num_iterations: Number of iterations for Sinkhorn algorithm
            stability_eps: Small constant for numerical stability
            debug_mode: Whether to print debugging information
        """
        super(BarycentricPooling, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.stability_eps = stability_eps
        self.debug_mode = debug_mode
        
        logger.info(f"Initializing BarycentricPooling with codebook_size={codebook_size}, "
                   f"epsilon={epsilon}, num_iterations={num_iterations}")
        
        # Initialize the codebook (atoms for barycenter)
        self.codebook = nn.Parameter(torch.randn(codebook_size, hidden_dim))
        nn.init.xavier_uniform_(self.codebook)
        logger.debug(f"Codebook initialized with shape {self.codebook.shape}")
    
    def _check_nan(self, tensor, name):
        """Check if tensor contains NaN values and log information."""
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            total_count = tensor.numel()
            logger.warning(f"NaN detected in {name}: {nan_count}/{total_count} values are NaN")
            
            # Log additional statistics to help diagnose the issue
            if not torch.isnan(tensor).all():
                non_nan_values = tensor[~torch.isnan(tensor)]
                logger.debug(f"{name} statistics - "
                           f"min: {non_nan_values.min().item():.4e}, "
                           f"max: {non_nan_values.max().item():.4e}, "
                           f"mean: {non_nan_values.mean().item():.4e}, "
                           f"std: {non_nan_values.std().item():.4e}")
            return True
        return False
    
    def forward(self, node_distributions, batch_idx):
        """
        Compute barycenter weights for node distributions.
        
        Args:
            node_distributions: Node distribution samples [num_nodes, distribution_size, hidden_dim]
            batch_idx: Batch indices for nodes [num_nodes]
            
        Returns:
            barycenter_weights: Barycenter weights [batch_size, codebook_size]
        """
        # Check for NaN in inputs
        if self.debug_mode:
            self._check_nan(node_distributions, "node_distributions")
            self._check_nan(self.codebook, "codebook")
        
        num_nodes = node_distributions.size(0)
        distribution_size = node_distributions.size(1)
        batch_size = batch_idx.max().item() + 1
        
        logger.debug(f"Processing batch with {num_nodes} nodes, {distribution_size} samples per node, "
                   f"batch_size={batch_size}")
        
        # Initialize barycenter weights
        barycenter_weights = torch.zeros(batch_size, self.codebook_size, device=node_distributions.device)
        
        # Process each graph in the batch
        for b in range(batch_size):
            # Get nodes belonging to this graph
            mask = (batch_idx == b)
            graph_nodes = node_distributions[mask]  # [graph_num_nodes, distribution_size, hidden_dim]
            
            if graph_nodes.size(0) == 0:
                logger.warning(f"Graph {b} has no nodes, skipping")
                continue
            
            logger.debug(f"Graph {b}: Processing {graph_nodes.size(0)} nodes")
            
            # Flatten node distributions
            flattened_nodes = graph_nodes.reshape(-1, self.hidden_dim)  # [graph_num_nodes * distribution_size, hidden_dim]
            
            if self.debug_mode and self._check_nan(flattened_nodes, f"flattened_nodes for graph {b}"):
                continue  # Skip this graph if NaNs are detected
            
            # Compute cost matrix between node samples and codebook atoms
            cost_matrix = self._compute_cost_matrix(flattened_nodes, self.codebook)
            
            if self.debug_mode:
                if self._check_nan(cost_matrix, f"cost_matrix for graph {b}"):
                    continue
                
                # Log cost matrix statistics
                logger.debug(f"Cost matrix for graph {b}: shape={cost_matrix.shape}, "
                           f"min={cost_matrix.min().item():.4e}, "
                           f"max={cost_matrix.max().item():.4e}, "
                           f"mean={cost_matrix.mean().item():.4e}")
            
            # Compute optimal transport weights using Sinkhorn algorithm
            try:
                logger.debug(f"Running Sinkhorn algorithm for graph {b} with epsilon={self.epsilon}, "
                           f"iterations={self.num_iterations}")
                weights = self._sinkhorn(cost_matrix)
                
                if self.debug_mode:
                    self._check_nan(weights, f"sinkhorn weights for graph {b}")
                
                # Handle NaN values in weights
                if torch.isnan(weights).any():
                    logger.warning(f"NaN detected in weights for graph {b}, replacing with uniform weights")
                    n, m = weights.shape
                    weights = torch.ones(n, m, device=weights.device) / m
                
                # Aggregate weights for the graph
                weights_sum = weights.sum(dim=0)  # [codebook_size]
                
                # Normalize weights, handling zeros
                weights_sum_sum = weights_sum.sum()
                if weights_sum_sum > self.stability_eps:
                    weights_sum = weights_sum / weights_sum_sum  # Normalize
                    logger.debug(f"Graph {b}: Normalized weights sum = {weights_sum_sum:.4e}")
                else:
                    # If sum is too small, use uniform weights
                    logger.warning(f"Graph {b}: Weights sum too small ({weights_sum_sum:.4e}), using uniform weights")
                    weights_sum = torch.ones_like(weights_sum) / weights_sum.size(0)
                
                barycenter_weights[b] = weights_sum
                
                if self.debug_mode:
                    # Log weight statistics
                    logger.debug(f"Graph {b} barycenter weights: "
                               f"min={weights_sum.min().item():.4e}, "
                               f"max={weights_sum.max().item():.4e}, "
                               f"mean={weights_sum.mean().item():.4e}, "
                               f"std={weights_sum.std().item():.4e}")
            
            except Exception as e:
                logger.error(f"Error in Sinkhorn algorithm for graph {b}: {str(e)}")
                # Use uniform weights as fallback
                barycenter_weights[b] = torch.ones(self.codebook_size, device=node_distributions.device) / self.codebook_size
        
        # Final check for NaNs
        if self.debug_mode:
            self._check_nan(barycenter_weights, "final barycenter_weights")
        
        # Replace any remaining NaNs with uniform weights
        if torch.isnan(barycenter_weights).any():
            nan_mask = torch.isnan(barycenter_weights)
            uniform_weights = torch.ones_like(barycenter_weights[nan_mask]) / self.codebook_size
            barycenter_weights[nan_mask] = uniform_weights
            logger.warning(f"Replaced {nan_mask.sum().item()} NaN values in final barycenter weights")
        
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
        
        # Compute cost matrix with better numerical stability
        x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
        y_norm = torch.sum(y ** 2, dim=1, keepdim=True).t()
        
        # Use torch.baddbmm for better numerical stability
        dist = x_norm + y_norm
        dist = torch.addmm(dist, x, y.t(), alpha=-2.0)
        
        # Ensure non-negative distances
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
        
        # Initialize kernel with numerical stability
        K = torch.exp(-cost_matrix / self.epsilon)
        
        # Check for NaN or Inf in kernel
        if self.debug_mode:
            if torch.isnan(K).any():
                logger.warning(f"NaN detected in Sinkhorn kernel: {torch.isnan(K).sum().item()} values")
            if torch.isinf(K).any():
                logger.warning(f"Inf detected in Sinkhorn kernel: {torch.isinf(K).sum().item()} values")
        
        # Replace NaN or Inf with small values
        K = torch.where(torch.isnan(K) | torch.isinf(K), torch.ones_like(K) * self.stability_eps, K)
        
        # Sinkhorn iterations with numerical stability
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        
        # Stabilized Sinkhorn iteration
        for i in range(self.num_iterations):
            # Update u
            Kv = torch.matmul(K, v)
            Kv = torch.max(Kv, torch.ones_like(Kv) * self.stability_eps)  # Ensure positive values
            u = a / Kv
            
            # Check for NaN in u
            if self.debug_mode and i % 5 == 0:
                if torch.isnan(u).any():
                    logger.warning(f"NaN detected in u at iteration {i}")
                    # Log statistics
                    if not torch.isnan(u).all():
                        non_nan_u = u[~torch.isnan(u)]
                        logger.debug(f"u at iteration {i}: "
                                   f"min={non_nan_u.min().item():.4e}, "
                                   f"max={non_nan_u.max().item():.4e}, "
                                   f"mean={non_nan_u.mean().item():.4e}")
            
            # Update v
            KTu = torch.matmul(K.t(), u)
            KTu = torch.max(KTu, torch.ones_like(KTu) * self.stability_eps)  # Ensure positive values
            v = b / KTu
            
            # Check for NaN in v
            if self.debug_mode and i % 5 == 0:
                if torch.isnan(v).any():
                    logger.warning(f"NaN detected in v at iteration {i}")
                    # Log statistics
                    if not torch.isnan(v).all():
                        non_nan_v = v[~torch.isnan(v)]
                        logger.debug(f"v at iteration {i}: "
                                   f"min={non_nan_v.min().item():.4e}, "
                                   f"max={non_nan_v.max().item():.4e}, "
                                   f"mean={non_nan_v.mean().item():.4e}")
        
        # Compute transport matrix
        # Use clamp to avoid numerical issues
        u_diag = torch.diag(torch.clamp(u, min=self.stability_eps, max=1e6))
        v_diag = torch.diag(torch.clamp(v, min=self.stability_eps, max=1e6))
        transport_matrix = torch.matmul(torch.matmul(u_diag, K), v_diag)
        
        return transport_matrix 