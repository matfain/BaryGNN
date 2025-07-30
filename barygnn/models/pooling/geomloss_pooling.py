import torch
import torch.nn as nn
import logging
from geomloss import SamplesLoss               # auto‑switches to KeOps for large arrays

logger = logging.getLogger(__name__)

class GeomLossBarycentricPooling(nn.Module):
    """
    Barycentric pooling (Sinkhorn OT) implemented with GeomLoss + KeOps.
    * API identical to POTBarycentricPooling *
       forward(node_distributions, batch_idx) -> barycenter_weights (B, K)
    """

    def __init__(self,
                 hidden_dim: int,
                 codebook_size: int = 32,
                 epsilon: float = 0.1,
                 max_iter: int = 100,          # kept for signature parity; GeomLoss ignores it
                 p: int = 2,
                 scaling: float = 0.9,         # GeomLoss’ α parameter for faster converg.
                 debug_mode: bool = False):
        super().__init__()

        self.hidden_dim     = int(hidden_dim)
        self.codebook_size  = int(codebook_size)
        self.epsilon        = float(epsilon)
        self.p              = int(p)
        self.scaling        = float(scaling)
        self.debug          = bool(debug_mode)

        # Shared learnable code‑book (same as current POT version)
        self.codebook = nn.Parameter(torch.empty(codebook_size, hidden_dim))
        nn.init.xavier_uniform_(self.codebook)

        # Pre‑instantiate SamplesLoss to reuse CUDA kernels
        # blur = ε; scaling default .9; “sparse” backend triggered automatically by KeOps
        self.sinkhorn = SamplesLoss("sinkhorn",
                                    p=self.p,
                                    blur=self.epsilon,
                                    scaling=self.scaling,
                                    backend="auto")   # auto → PyTorch or KeOps

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _uniform(n, device, dtype):
        return torch.full((n,), 1.0 / n, device=device, dtype=dtype)

    def _log_if_nan(self, tensor, name):
        if self.debug and torch.isnan(tensor).any():
            nan_cnt = torch.isnan(tensor).sum().item()
            logger.warning(f"NaN detected in {name}: {nan_cnt}/{tensor.numel()}")
            return True
        return False

    # ------------------------------------------------------------------ forward
    def forward(self, node_distributions: torch.Tensor, batch_idx: torch.Tensor):
        """
        node_distributions : [N, S, hidden_dim]
        batch_idx          : [N]  (values 0 … B‑1)
        Returns:
            barycenter_weights : [B, codebook_size] ; each row sums to 1
        """
        N, S, d = node_distributions.shape
        if d != self.hidden_dim:
            raise ValueError(f"Dim mismatch: encoder outputs {d}, codebook expects {self.hidden_dim}")

        B = int(batch_idx.max().item()) + 1
        K = self.codebook_size
        w_graph = node_distributions.new_zeros(B, K)      # output buffer

        for b in range(B):
            # collect all samples of graph b and flatten
            X = node_distributions[batch_idx == b].reshape(-1, d)  # (n_b * S, d)
            if X.numel() == 0:
                w_graph[b].fill_(1.0 / K)
                continue

            # uniform source & target weights
            a = self._uniform(len(X), X.device, X.dtype)
            b_vec = self._uniform(K,    self.codebook.device, self.codebook.dtype)

            # GeomLoss returns sparse transport plan; column‑sum = weights
            plan = self.sinkhorn.transport(a, X, b_vec, self.codebook)  # (len(X), K)
            w    = plan.sum(dim=0)                                     # (K,)

            # normalise (in theory already summing to 1 but safe against fp error)
            w = w / w.sum()

            if self._log_if_nan(w, f"graph {b} weights"):
                w.fill_(1.0 / K)                                       # fallback uniform

            w_graph[b] = w

        return w_graph