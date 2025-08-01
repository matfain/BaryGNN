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
    
    def _compute_ot_histogram(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the optimal transport histogram: how is mass from X
        distributed across the codebook entries?
        
        This is the COLUMN SUMS of the optimal transport plan π.
        """
        n_samples = len(X)
        K = self.codebook_size
        
        # Uniform source and target measures
        a = torch.full((n_samples,), 1.0/n_samples, device=X.device, dtype=X.dtype)
        b = torch.full((K,), 1.0/K, device=self.codebook.device, dtype=self.codebook.dtype)
        
        try:
            # Get dual potentials from Sinkhorn
            Fi, Gj = self.sinkhorn(a, X, b, self.codebook)
            
            # Reconstruct transport plan and get column sums
            histogram = self._reconstruct_histogram_from_potentials(
                Fi, Gj, X, a, b
            )
            
            # Normalize (should already sum to 1, but numerical safety)
            histogram = histogram / (histogram.sum() + 1e-8)
            
            # Debug check
            if self.debug and torch.isnan(histogram).any():
                logger.warning("NaN in histogram, using uniform fallback")
                histogram = torch.full((K,), 1.0/K, device=histogram.device)
                
            return histogram
            
        except Exception as e:
            if self.debug:
                logger.error(f"OT computation failed: {e}")
            # Fallback to uniform
            return torch.full((K,), 1.0/K, device=X.device)
    
    def _reconstruct_histogram_from_potentials(self, Fi, Gj, X, a, b):
        """
        Reconstruct the column sums of optimal transport plan from dual potentials.
        
        π_{ij} = exp((Fi + Gj - C_{ij})/ε) * a_i * b_j
        histogram_j = Σ_i π_{ij}
        """
        # Cost matrix (matching GeomLoss's cost function)
        if self.p == 2:
            C_ij = torch.cdist(X, self.codebook, p=2).pow(2)
        else:
            C_ij = torch.cdist(X, self.codebook, p=self.p).pow(self.p)
        
        # Compute transport plan entries in log space for numerical stability
        log_pi = (Fi[:, None] + Gj[None, :] - C_ij) / self.epsilon
        log_pi = log_pi + torch.log(a[:, None]) + torch.log(b[None, :])
        
        # Column sums using log-sum-exp for stability
        histogram = torch.logsumexp(log_pi, dim=0).exp()
        
        return histogram
    
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