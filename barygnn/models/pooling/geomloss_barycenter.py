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
        Compute true Wasserstein barycenter for each graph.
        
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
        
        # Pre-allocate output tensor
        graph_histograms = torch.zeros(B, K, device=node_distributions.device, dtype=node_distributions.dtype)
        graph_node_counts = torch.zeros(B, device=node_distributions.device, dtype=torch.long)
        
        # Count nodes per graph for averaging later
        for b in range(B):
            graph_node_counts[b] = (batch_idx == b).sum().item()
        
        # Process each node's distribution and accumulate results by graph
        for n in range(N):
            # Get node's distribution and graph index
            node_dist = node_distributions[n]  # [S, hidden_dim]
            graph_idx = batch_idx[n].item()
            
            # Compute OT histogram for this node
            node_hist = self._compute_ot_histogram(node_dist, prior)
            
            # Accumulate histogram for the graph
            graph_histograms[graph_idx] += node_hist
        
        # Average node histograms for each graph
        for b in range(B):
            if graph_node_counts[b] > 0:
                graph_histograms[b] = graph_histograms[b] / graph_node_counts[b]
            else:
                # Fallback to prior if graph has no nodes (shouldn't happen but just in case)
                graph_histograms[b] = prior
                
        return graph_histograms
    
    def _compute_ot_histogram(self, X: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        """
        Compute OT histogram for a single node's distribution.
        
        Args:
            X: [S, hidden_dim] - Node's feature distribution
            prior: [codebook_size] - Prior distribution over codebook atoms
            
        Returns:
            hist: [codebook_size] - OT histogram
        """
        S = X.size(0)  # Number of samples in node distribution
        K = self.codebook_size
        
        # Uniform source mass (each sample in node distribution has equal weight)
        a = torch.full((S,), 1.0 / S, device=X.device, dtype=X.dtype)
        
        # Target mass is the learnable prior
        b = prior.to(device=X.device, dtype=X.dtype)
        
        # GeomLoss returns (1, S) and (1, K); squeeze to 1-D
        Fi, Gj = self.sinkhorn(a, X, b, self.codebook)
        Fi, Gj = Fi.squeeze(0), Gj.squeeze(0)  # (S,), (K,)
        
        # Cost matrix ||x_i – c_j||_p^p (broadcasted)
        C_ij = torch.cdist(X, self.codebook, p=self.p).pow(self.p)  # (S, K)
        
        # Log-space transport plan: log π_ij = (Fi_i + Gj_j − C_ij) / ε + log a_i + log b_j
        log_pi = (Fi[:, None] + Gj[None, :] - C_ij) / self.epsilon
        log_pi = log_pi + torch.log(a)[:, None] + torch.log(b)[None, :]
        
        # Column sums: log-sum-exp over i
        hist = torch.exp(torch.logsumexp(log_pi, dim=0))  # (K,)
        
        # Normalize for safety
        hist = hist / (hist.sum() + 1e-12)
        
        if self.debug and torch.isnan(hist).any():
            self._warn_and_fix_hist(hist, prior)
            
        return hist
    
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