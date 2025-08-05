import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from geomloss import SamplesLoss

logger = logging.getLogger(__name__)

class BarycentricPooling(nn.Module):
    """
    True Wasserstein Barycenter Pooling with learnable codebook prior.
    
    This implementation addresses two key improvements:
    1. Uses a learnable prior distribution for the codebook atoms
    2. Computes the true Wasserstein barycenter by:
       - Computing OT between each node's distribution and the codebook separately
       - Averaging the resulting histograms across all nodes in each graph
    """

    def __init__(self,
                 hidden_dim: int,
                 codebook_size: int = 32,
                 epsilon: float = 0.1,
                 p: int = 2,
                 scaling: float = 0.9,
                 debug_mode: bool = False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.epsilon = epsilon
        self.p = p
        self.scaling = scaling
        self.debug = debug_mode

        # Learnable codebook
        self.codebook = nn.Parameter(torch.empty(codebook_size, hidden_dim))
        nn.init.xavier_uniform_(self.codebook)
        
        # Learnable codebook prior (initialized as uniform)
        # Using log-space parameterization for numerical stability and to ensure positive values
        self.log_codebook_prior = nn.Parameter(torch.zeros(codebook_size))

        # Sinkhorn with potentials (needed for transport plan reconstruction)
        self.sinkhorn = SamplesLoss(
            "sinkhorn",
            p=self.p,
            blur=self.epsilon,
            scaling=self.scaling,
            potentials=True,     # Essential for getting dual potentials
            backend="auto"
        )

    @property
    def codebook_prior(self):
        """
        Get the normalized codebook prior distribution.
        Uses softmax to ensure it's a valid probability distribution.
        """
        return F.softmax(self.log_codebook_prior, dim=0)

    def forward(self, node_distributions: torch.Tensor, batch_idx: torch.Tensor):
        """
        Compute true Wasserstein barycenter for each graph using vectorized operations.
        
        Args:
            node_distributions: [N, S, hidden_dim] - node feature distributions
            batch_idx: [N] - graph assignment for each node
            
        Returns:
            histograms: [B, codebook_size] - Barycenter histogram for each graph
        """
        N, S, d = node_distributions.shape
        if d != self.hidden_dim:
            raise ValueError(f"Dimension mismatch: got {d}, expected {self.hidden_dim}")

        B = int(batch_idx.max().item()) + 1
        K = self.codebook_size
        
        # Get normalized codebook prior
        prior = self.codebook_prior
        
        # Compute OT histograms for all nodes in parallel
        node_histograms = self._compute_ot_histogram_batch(node_distributions, prior)  # [N, K]
        
        # Pre-allocate output tensor for graph histograms
        graph_histograms = torch.zeros(B, K, device=node_distributions.device, dtype=node_distributions.dtype)
        graph_node_counts = torch.zeros(B, device=node_distributions.device, dtype=torch.long)
        
        # Count nodes per graph for averaging later (vectorized)
        unique_graphs, counts = torch.unique(batch_idx, return_counts=True)
        graph_node_counts[unique_graphs] = counts
        
        # Accumulate histograms by graph (using index_add for efficiency)
        for n in range(N):
            graph_idx = batch_idx[n].item()
            graph_histograms[graph_idx] += node_histograms[n]
        
        # Average node histograms for each graph (vectorized)
        valid_graphs = graph_node_counts > 0
        graph_histograms[valid_graphs] = graph_histograms[valid_graphs] / graph_node_counts[valid_graphs].unsqueeze(1)
        
        # Handle empty graphs (if any)
        empty_graphs = ~valid_graphs
        if empty_graphs.any():
            graph_histograms[empty_graphs] = prior.unsqueeze(0).expand(empty_graphs.sum(), K)
                
        return graph_histograms
    
    def _compute_ot_histogram_batch(self, X_batch: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        Compute OT histograms for a batch of node distributions in parallel.
        
        Args:
            X_batch: [N, S, hidden_dim] - Batch of node feature distributions
            prior: [codebook_size] - Prior distribution over codebook atoms
            
        Returns:
            hist_batch: [N, codebook_size] - Batch of OT histograms
        """
        N, S, d = X_batch.shape
        K = self.codebook_size
        device = X_batch.device
        dtype = X_batch.dtype
        
        # Create uniform weights for each node's distribution samples
        # Shape: [N, S]
        a_batch = torch.full((N, S), 1.0 / S, device=device, dtype=dtype)
        
        # Expand prior for each node
        # Shape: [N, K]
        b_batch = prior.unsqueeze(0).expand(N, K).to(dtype=dtype)
        
        # Reshape inputs for batch processing with GeomLoss
        # GeomLoss expects weights as [N, S] and [N, K]
        # and points as [N, S, D] and [N, K, D]
        
        # Expand codebook for each node in batch
        # Shape: [N, K, d]
        # Use contiguous() to ensure memory layout is contiguous
        codebook_expanded = self.codebook.unsqueeze(0).expand(N, K, d).contiguous()
        
        # Ensure all inputs are contiguous
        a_batch = a_batch.contiguous()
        X_batch = X_batch.contiguous()
        b_batch = b_batch.contiguous()
        
        # Compute batch OT with GeomLoss
        # This returns dual potentials Fi: [N, S] and Gj: [N, K]
        Fi, Gj = self.sinkhorn(a_batch, X_batch, b_batch, codebook_expanded)
        
        # Compute cost matrices for all nodes at once
        # Shape: [N, S, K]
        # Use contiguous tensors for cdist operation
        C_ij_batch = torch.cdist(X_batch, self.codebook.unsqueeze(0).expand(N, K, d).contiguous(), p=self.p).pow(self.p)
        
        # Compute log transport plans for all nodes
        # Shape: [N, S, K]
        log_pi_batch = (Fi.unsqueeze(-1) + Gj.unsqueeze(1) - C_ij_batch) / self.epsilon
        log_pi_batch = log_pi_batch + torch.log(a_batch).unsqueeze(-1) + torch.log(b_batch).unsqueeze(1)
        
        # Column sums: log-sum-exp over S dimension for each node
        # Shape: [N, K]
        hist_batch = torch.exp(torch.logsumexp(log_pi_batch, dim=1))
        
        # Normalize each histogram for safety
        # Shape: [N, K]
        row_sums = hist_batch.sum(dim=1, keepdim=True)
        hist_batch = hist_batch / (row_sums + 1e-12)
        
        # Handle NaN values if debugging is enabled
        if self.debug and torch.isnan(hist_batch).any():
            self._warn_and_fix_hist_batch(hist_batch, prior)
            
        return hist_batch
        
    def _compute_ot_histogram(self, X: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        Compute OT histogram for a single node's distribution.
        
        Args:
            X: [S, hidden_dim] - Node's feature distribution
            prior: [codebook_size] - Prior distribution over codebook atoms
            
        Returns:
            hist: [codebook_size] - OT histogram
        """
        # Reshape to batch of size 1 and use the batch version
        X_batch = X.unsqueeze(0)  # [1, S, hidden_dim]
        hist_batch = self._compute_ot_histogram_batch(X_batch, prior)
        return hist_batch.squeeze(0)  # [codebook_size]
    
    def _warn_and_fix_hist_batch(self, hist_batch: torch.Tensor, prior: torch.Tensor):
        """
        Handle NaN values in a batch of histograms by falling back to the prior.
        
        Args:
            hist_batch: [N, codebook_size] - Batch of computed histograms
            prior: [codebook_size] - Prior distribution
        """
        # Find which histograms have NaN values
        nan_mask = torch.isnan(hist_batch).any(dim=1)
        if nan_mask.any():
            num_nans = nan_mask.sum().item()
            logger.warning(f"NaN in {num_nans} histograms, falling back to prior.")
            
            # Replace NaN histograms with the prior
            hist_batch[nan_mask] = prior.unsqueeze(0).expand(num_nans, -1)
    
    def _warn_and_fix_hist(self, hist: torch.Tensor, prior: torch.Tensor):
        """
        Handle NaN values in histogram by falling back to the prior.
        
        Args:
            hist: [codebook_size] - Computed histogram
            prior: [codebook_size] - Prior distribution
        """
        logger.warning("NaN in histogram, falling back to prior.")
        hist.copy_(prior)
    
    def register_gradient_hooks(self):
        """
        Register gradient hooks on the codebook and prior to track gradient flow.
        Call this method after model initialization.
        """
        def codebook_hook_fn(grad):
            with torch.no_grad():
                grad_norm = grad.norm().item()
                has_nan = torch.isnan(grad).any().item()
                if has_nan:
                    logger.warning("NaN gradients detected in codebook!")
                else:
                    logger.info(f"Codebook gradient norm: {grad_norm:.6f}")
                    logger.info(f"Codebook grad min/max: {grad.min().item():.6f}/{grad.max().item():.6f}")
            return grad
        
        def prior_hook_fn(grad):
            with torch.no_grad():
                grad_norm = grad.norm().item()
                has_nan = torch.isnan(grad).any().item()
                if has_nan:
                    logger.warning("NaN gradients detected in codebook prior!")
                else:
                    logger.info(f"Codebook prior gradient norm: {grad_norm:.6f}")
                    logger.info(f"Codebook prior grad min/max: {grad.min().item():.6f}/{grad.max().item():.6f}")
            return grad
        
        # Register hooks
        self.codebook_hook = self.codebook.register_hook(codebook_hook_fn)
        self.prior_hook = self.log_codebook_prior.register_hook(prior_hook_fn)
        logger.info("Gradient hooks registered on codebook and prior")
        
    def remove_gradient_hooks(self):
        """Remove gradient hooks to stop tracking."""
        if hasattr(self, 'codebook_hook'):
            self.codebook_hook.remove()
        if hasattr(self, 'prior_hook'):
            self.prior_hook.remove()
        logger.info("Gradient hooks removed from codebook and prior")