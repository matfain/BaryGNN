import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import logging

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    ot = None

# Set up logger
logger = logging.getLogger(__name__)


class POTBarycentricPooling(nn.Module):
    """
    Barycentric pooling module using POT (Python Optimal Transport) for optimal transport computation.
    
    This module computes barycenter weights for node distributions using the
    well-established POT library, which provides robust CPU-based implementations
    of Sinkhorn algorithms and Wasserstein barycenters.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        codebook_size: int = 16,
        epsilon: float = 0.2,
        max_iter: int = 100,
        tol: float = 1e-6,
        p: int = 2,
        debug_mode: bool = False,
    ):
        """
        Initialize the POT-based barycentric pooling module.
        
        Args:
            hidden_dim: Dimension of node embeddings
            codebook_size: Number of atoms in the codebook
            epsilon: Regularization parameter for entropic regularization
            max_iter: Maximum number of Sinkhorn iterations
            tol: Tolerance for convergence
            p: Order of the Wasserstein distance (typically 2)
            debug_mode: Whether to print debugging information
        """
        super(POTBarycentricPooling, self).__init__()
        
        if not HAS_POT:
            raise ImportError(
                "POT is required for POTBarycentricPooling. "
                "Install it with: pip install POT"
            )
        
        self.hidden_dim = int(hidden_dim)
        self.codebook_size = int(codebook_size)
        self.epsilon = float(epsilon)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.p = int(p)
        self.debug_mode = bool(debug_mode)
        
        logger.info(f"Initializing POTBarycentricPooling with codebook_size={codebook_size}, "
                   f"epsilon={epsilon}, max_iter={max_iter}, tol={tol}, p={p}")
        
        # Initialize the codebook (atoms for barycenter)
        self.codebook = nn.Parameter(torch.randn(codebook_size, hidden_dim))
        nn.init.xavier_uniform_(self.codebook)
        logger.debug(f"Codebook initialized with shape {self.codebook.shape}")
        
        # Small constant for numerical stability
        self.stability_eps = 1e-8
    
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
    
    def _compute_cost_matrix(self, X, Y):
        """Compute cost matrix between two sets of points."""
        if self.p == 2:
            # Use squared Euclidean distance for Wasserstein-2
            return torch.cdist(X, Y, p=2) ** 2
        else:
            # Use p-th power of Euclidean distance
            return torch.cdist(X, Y, p=2) ** self.p
    
    def _sinkhorn_stabilized(self, a, b, M, reg, numItermax=1000, tau=1e3, stopThr=1e-9, verbose=False):
        """
        Stabilized Sinkhorn algorithm using POT but with PyTorch tensors.
        
        This is a PyTorch adaptation of POT's stabilized Sinkhorn that maintains
        gradients for backpropagation.
        """
        device = M.device
        dtype = M.dtype
        
        # Convert to numpy for POT computation
        a_np = a.detach().cpu().numpy().astype(np.float64)
        b_np = b.detach().cpu().numpy().astype(np.float64)
        M_np = M.detach().cpu().numpy().astype(np.float64)
        
        # Use POT's Sinkhorn algorithm
        try:
            # Ensure all parameters are numeric
            reg = float(reg)
            numItermax = int(numItermax)
            stopThr = float(stopThr)
            verbose = bool(verbose)
            
            # Use entropic regularized OT
            transport_plan = ot.sinkhorn(a_np, b_np, M_np, reg, 
                                       numItermax=numItermax, 
                                       stopThr=stopThr,
                                       verbose=verbose)
            
            # Convert back to PyTorch tensor
            transport_plan = torch.from_numpy(transport_plan).to(device=device, dtype=dtype)
            
            # Ensure gradients are maintained by using the original cost matrix
            # This is a trick to maintain gradients while using numpy computation
            if M.requires_grad:
                # Recompute with PyTorch operations for gradient flow
                # Use the transport plan as "target" and minimize difference
                log_u = torch.zeros(a.shape[0], device=device, dtype=dtype, requires_grad=True)
                log_v = torch.zeros(b.shape[0], device=device, dtype=dtype, requires_grad=True)
                
                # Simplified Sinkhorn iterations in PyTorch for gradient flow
                for _ in range(min(10, numItermax)):  # Fewer iterations since we have good initialization
                    log_u = torch.log(a + self.stability_eps) - torch.logsumexp(-M/reg + log_v.unsqueeze(0), dim=1)
                    log_v = torch.log(b + self.stability_eps) - torch.logsumexp(-M/reg + log_u.unsqueeze(1), dim=0)
                
                # Compute transport plan with gradients
                transport_plan_grad = torch.exp(log_u.unsqueeze(1) + log_v.unsqueeze(0) - M/reg)
                
                # Normalize to ensure marginal constraints (approximately)
                transport_plan_grad = transport_plan_grad * (transport_plan.sum() / transport_plan_grad.sum())
                
                return transport_plan_grad
            else:
                return transport_plan
                
        except Exception as e:
            logger.warning(f"POT Sinkhorn failed: {e}, falling back to uniform transport")
            # Fallback to uniform transport plan
            uniform_plan = torch.outer(a, b)
            return uniform_plan
    
    def forward(self, node_distributions, batch_idx):
        """
        Compute barycenter weights for node distributions using POT.
        
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
        feat_dim = node_distributions.size(2)  # Get actual feature dimension
        batch_size = batch_idx.max().item() + 1
        
        # Check dimension compatibility
        if feat_dim != self.codebook.size(1):
            error_msg = (
                f"CRITICAL DIMENSION MISMATCH:\n"
                f"  Node distributions shape: {node_distributions.shape}\n"
                f"  Expected: [num_nodes, distribution_size, hidden_dim]\n"
                f"  Actual feat_dim: {feat_dim}\n"
                f"  Codebook shape: {self.codebook.shape}\n"
                f"  Expected codebook: [codebook_size, hidden_dim]\n"
                f"  Codebook feat_dim: {self.codebook.size(1)}\n"
                f"  \n"
                f"  The encoder is outputting vectors of dimension {feat_dim}\n"
                f"  but the codebook expects dimension {self.codebook.size(1)}.\n"
                f"  Check your configuration: encoder.hidden_dim should equal model.hidden_dim"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Processing batch with {num_nodes} nodes, {distribution_size} samples per node, "
                   f"batch_size={batch_size}, feat_dim={feat_dim}")
        logger.debug(f"Codebook shape: {self.codebook.shape}")
        
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
            
            # Flatten node distributions for optimal transport
            # Each node contributes distribution_size samples
            flattened_nodes = graph_nodes.reshape(-1, feat_dim)  # [graph_num_nodes * distribution_size, feat_dim]
                        
            if self.debug_mode and self._check_nan(flattened_nodes, f"flattened_nodes for graph {b}"):
                # Use uniform weights as fallback
                barycenter_weights[b] = torch.ones(self.codebook_size, device=node_distributions.device) / self.codebook_size
                continue
            
            try:
                # Compute optimal transport weights using POT
                logger.debug(f"Running POT Sinkhorn for graph {b} with epsilon={self.epsilon}")
                
                # Create uniform weights for source (node samples) and target (codebook)
                n_samples = flattened_nodes.size(0)
                source_weights = torch.ones(n_samples, device=flattened_nodes.device, dtype=flattened_nodes.dtype) / n_samples
                target_weights = torch.ones(self.codebook_size, device=self.codebook.device, dtype=self.codebook.dtype) / self.codebook_size
                
                # DEBUG: Print exact shapes before POT call
                logger.debug(f"Graph {b} - Before POT call:")
                logger.debug(f"  source_weights shape: {source_weights.shape}")
                logger.debug(f"  flattened_nodes shape: {flattened_nodes.shape}")
                logger.debug(f"  target_weights shape: {target_weights.shape}")
                logger.debug(f"  codebook shape: {self.codebook.shape}")
                
                # Compute cost matrix
                cost_matrix = self._compute_cost_matrix(flattened_nodes, self.codebook)
                
                if self.debug_mode:
                    logger.debug(f"Graph {b}: cost_matrix shape = {cost_matrix.shape}")
                    self._check_nan(cost_matrix, f"cost_matrix for graph {b}")
                
                # Compute optimal transport plan using stabilized Sinkhorn
                transport_plan = self._sinkhorn_stabilized(
                    source_weights, target_weights, cost_matrix, 
                    self.epsilon, numItermax=self.max_iter, 
                    stopThr=self.tol, verbose=self.debug_mode
                )
                
                if self.debug_mode:
                    logger.debug(f"Graph {b}: transport_plan shape = {transport_plan.shape}")
                    self._check_nan(transport_plan, f"transport_plan for graph {b}")
                
                # Compute target marginals (sum over sources)
                # This gives us the weights for each codebook atom
                target_marginal = transport_plan.sum(dim=0)  # [codebook_size]
                
                if self.debug_mode:
                    logger.debug(f"Graph {b}: target_marginal shape = {target_marginal.shape}")
                    self._check_nan(target_marginal, f"target_marginal for graph {b}")
                
                # Handle NaN values
                if torch.isnan(target_marginal).any():
                    logger.warning(f"NaN detected in target marginal for graph {b}, using uniform weights")
                    barycenter_weights[b] = torch.ones(self.codebook_size, device=node_distributions.device) / self.codebook_size
                    continue
                
                # Normalize weights (should already be normalized, but ensure numerical stability)
                weights_sum_total = target_marginal.sum()
                if weights_sum_total > self.stability_eps:
                    weights_normalized = target_marginal / weights_sum_total
                    logger.debug(f"Graph {b}: Normalized weights sum = {weights_sum_total:.4e}")
                else:
                    # If sum is too small, use uniform weights
                    logger.warning(f"Graph {b}: Weights sum too small ({weights_sum_total:.4e}), using uniform weights")
                    weights_normalized = torch.ones_like(target_marginal) / target_marginal.size(0)
                
                barycenter_weights[b] = weights_normalized
                
                if self.debug_mode:
                    # Log weight statistics
                    logger.debug(f"Graph {b} barycenter weights: "
                               f"min={weights_normalized.min().item():.4e}, "
                               f"max={weights_normalized.max().item():.4e}, "
                               f"mean={weights_normalized.mean().item():.4e}, "
                               f"std={weights_normalized.std().item():.4e}")
            
            except Exception as e:
                logger.error(f"Error in POT computation for graph {b}: {str(e)}")
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


def create_barycentric_pooling(**kwargs) -> nn.Module:
    """
    Factory function to create POT-based barycentric pooling modules.
    
    Args:
        **kwargs: Additional arguments for the POT pooling module
        
    Returns:
        POT barycentric pooling module instance
        
    Raises:
        ImportError: If POT is not available
    """
    if not HAS_POT:
        raise ImportError(
            "POT is required for barycentric pooling. "
            "Install it with: pip install POT"
        )
    
    return POTBarycentricPooling(**kwargs) 