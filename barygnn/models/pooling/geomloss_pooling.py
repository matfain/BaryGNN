import torch
import torch.nn as nn
import logging
from geomloss import SamplesLoss

logger = logging.getLogger(__name__)

class GeomLossBarycentricPooling(nn.Module):
    """
    Optimal Transport Histogram Pooling - finds the exact distribution
    of graph samples across codebook entries using Sinkhorn OT.
    
    This is your original approach, cleaned up and optimized.
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

        # Sinkhorn with potentials (needed for transport plan reconstruction)
        self.sinkhorn = SamplesLoss(
            "sinkhorn",
            p=self.p,
            blur=self.epsilon,
            scaling=self.scaling,
            potentials=True,     # Essential for getting dual potentials
            backend="auto"
        )

    def forward(self, node_distributions: torch.Tensor, batch_idx: torch.Tensor):
        """
        Compute optimal transport histogram for each graph.
        
        Args:
            node_distributions: [N, S, hidden_dim] - node feature distributions
            batch_idx: [N] - graph assignment for each node
            
        Returns:
            histograms: [B, codebook_size] - OT histogram for each graph
        """
        N, S, d = node_distributions.shape
        if d != self.hidden_dim:
            raise ValueError(f"Dimension mismatch: got {d}, expected {self.hidden_dim}")

        B = int(batch_idx.max().item()) + 1
        K = self.codebook_size
        
        histograms = node_distributions.new_zeros(B, K)
        
        for b in range(B):
            mask = (batch_idx == b)
            if not mask.any():
                histograms[b] = 1.0 / K  # Uniform fallback
                continue
                
            # Flatten all samples from graph b
            X = node_distributions[mask].reshape(-1, d)  # (n_samples, d)
            
            # Compute OT histogram
            hist = self._compute_ot_histogram(X)
            histograms[b] = hist
            
        return histograms
    
    @torch.no_grad()
    def _compute_ot_histogram(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return the column-sums of the optimal transport plan between the
        empirical distribution defined by X and the (learnable) codebook.

        Args
        ----
        X : (n, hidden_dim) tensor of node samples for one graph.

        Returns
        -------
        hist : (codebook_size,) OT histogram that sums to 1.
        """
        n = X.size(0)             # number of samples
        K = self.codebook_size     # shorthand

        # Uniform source / target masses
        a = torch.full((n,), 1.0 / n,  device=X.device, dtype=X.dtype)   # (n,)
        b = torch.full((K,), 1.0 / K,  device=X.device, dtype=X.dtype)   # (K,)

        # GeomLoss returns (1, n) and (1, K); squeeze to 1-D
        Fi, Gj = self.sinkhorn(a, X, b, self.codebook)   # potentials
        Fi, Gj = Fi.squeeze(0), Gj.squeeze(0)            # (n,), (K,)

        # Cost matrix  ||x_i – c_j||_p^p  (broadcasted)
        C_ij = torch.cdist(X, self.codebook, p=self.p).pow(self.p)  # (n, K)

        # log-space transport plan: log π_ij = (Fi_i + Gj_j − C_ij) / ε + log a_i + log b_j
        log_pi = (Fi[:, None] + Gj[None, :] - C_ij) / self.epsilon
        log_pi = log_pi + torch.log(a)[:, None] + torch.log(b)[None, :]

        # Column sums: log-sum-exp over i
        hist = torch.exp(torch.logsumexp(log_pi, dim=0))  # (K,)

        # Normalise for safety
        hist = hist / (hist.sum() + 1e-12)

        if self.debug and torch.isnan(hist).any():
            self._warn_and_fix_hist(hist)
        return hist


    # Optional helper for cleaner warnings / fallbacks
    def _warn_and_fix_hist(self, hist: torch.Tensor):
        logger.warning("NaN in histogram, falling back to uniform.")
        hist.fill_(1.0 / self.codebook_size)
            
    
    def get_transport_plan(self, X: torch.Tensor) -> torch.Tensor:
        """
        Optional: Get the full transport plan π_{ij} if needed for analysis.
        
        Returns:
            transport_plan: [n_samples, codebook_size] where entry (i,j) is
                           the amount of mass from sample i assigned to codebook entry j
        """
        n_samples = len(X)
        K = self.codebook_size
        
        a = torch.full((n_samples,), 1.0/n_samples, device=X.device, dtype=X.dtype)
        b = torch.full((K,), 1.0/K, device=self.codebook.device, dtype=self.codebook.dtype)
        
        Fi, Gj = self.sinkhorn(a, X, b, self.codebook)
        
        # Cost matrix
        if self.p == 2:
            C_ij = torch.cdist(X, self.codebook, p=2).pow(2)
        else:
            C_ij = torch.cdist(X, self.codebook, p=self.p).pow(self.p)
        
        # Full transport plan
        log_pi = (Fi[:, None] + Gj[None, :] - C_ij) / self.epsilon
        log_pi = log_pi + torch.log(a[:, None]) + torch.log(b[None, :])
        transport_plan = log_pi.exp()
        
        return transport_plan