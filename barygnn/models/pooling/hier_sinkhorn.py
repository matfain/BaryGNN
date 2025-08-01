import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from geomloss import SamplesLoss

logger = logging.getLogger(__name__)


class Codebook(nn.Module):
    """
    Shared learnable codebook for hierarchical pooling.
    
    This codebook is used by both stages:
    - Stage 1: Node samples are transported to codebook atoms
    - Stage 2: Node histograms are transported using atom-atom distances
    """
    
    def __init__(self, codebook_size: int, hidden_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.hidden_dim = hidden_dim
        
        # Learnable codebook atoms
        self.C = nn.Parameter(torch.empty(codebook_size, hidden_dim))
        nn.init.xavier_uniform_(self.C)
        
        logger.info(f"Initialized Codebook with {codebook_size} atoms of dimension {hidden_dim}")
    
    def forward(self):
        """Return the codebook for access by pooling stages."""
        return self.C


class NodeSinkhornPooling(nn.Module):
    """
    Stage-1 OT: move S node-level samples to the K shared codebook atoms,
    returning a K-dim histogram for every node.
    """

    def __init__(self, epsilon_node: float = 0.3):
        super().__init__()
        self.epsilon = epsilon_node            # blur in Sinkhorn
        # We ask GeomLoss for dual potentials (size ≪ transport plan)
        self.sinkhorn = SamplesLoss(
            "sinkhorn", p=2, blur=self.epsilon,
            scaling=0.9, backend="auto", potentials=True
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _safe_uniform(size: int, device, dtype):
        return torch.full((size,), 1.0 / size, device=device, dtype=dtype)

    # ----------------------------------------------------------------- forward
    def forward(self, samples: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        samples   : (N, S, d)   node-level embeddings
        codebook  : (K, d)      shared atoms

        Returns
        -------
        (N, K) tensor of per-node histograms.
        """
        N, S, d = samples.shape
        K, _ = codebook.shape
        device, dtype = samples.device, samples.dtype

        # Uniform source / target masses (shared across nodes)
        a = self._safe_uniform(S, device, dtype)      # (S,)
        b = self._safe_uniform(K, device, dtype)      # (K,)

        histograms = []

        for i in range(N):
            X = samples[i]                            # (S, d)

            # 1. dual potentials (GeomLoss adds a leading batch dim → squeeze)
            Fi, Gj = self.sinkhorn(a, X, b, codebook)
            Fi, Gj = Fi.squeeze(0), Gj.squeeze(0)     # (S,), (K,)

            # 2. cost matrix ‖x_i - c_j‖²
            C = torch.cdist(X, codebook, p=2).pow(2)  # (S, K)

            # 3. log-space transport plan
            log_pi = (Fi[:, None] + Gj[None, :] - C) / self.epsilon
            log_pi += torch.log(a)[:, None] + torch.log(b)[None, :]

            # 4. column sums → histogram
            hist = torch.exp(torch.logsumexp(log_pi, dim=0))  # (K,)
            hist = hist / (hist.sum() + 1e-12)                # normalise

            histograms.append(hist)

        return torch.stack(histograms, dim=0)          # (N, K)


class GraphSinkhornPooling(nn.Module):
    """
    Stage-2 OT: move each node histogram to a *single* graph-level histogram,
    using the Wasserstein geometry *induced* by the codebook.
    """

    def __init__(self, epsilon_graph: float = 0.1, max_iter: int = 100, tol: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon_graph
        self.max_iter, self.tol = max_iter, tol
        self.register_buffer("atom_distances", None)

    # ---------------------------------------------------------- pre-compute Dᵢⱼ
    def _precompute_atom_distances(self, codebook: torch.Tensor):
        # D[i,j] = ‖c_i − c_j‖²  (used as cost between simplex vertices)
        self.atom_distances = torch.cdist(codebook, codebook, p=2).pow(2)

    # -------------------------------------------------------- manual Sinkhorn
    def _sinkhorn_log(self, a: torch.Tensor, b: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Log-domain Sinkhorn with a *fixed* cost matrix M.

        Args
        ----
        a : (n,)   source masses  (nodes)
        b : (m,)   target masses  (atoms)
        M : (n,m)  cost matrix    (<node hist>, <atom>)

        Returns
        -------
        π : (n,m)  optimal transport plan.
        """
        n, m = M.shape
        K_log = -M / self.epsilon                     # log K = −C/ε

        u = torch.zeros(n, device=M.device, dtype=M.dtype)
        v = torch.zeros(m, device=M.device, dtype=M.dtype)

        for _ in range(self.max_iter):
            u_prev = u.clone()

            u = torch.log(a + 1e-12) - torch.logsumexp(K_log + v[None, :], dim=1)
            v = torch.log(b + 1e-12) - torch.logsumexp(K_log + u[:, None], dim=0)

            if torch.norm(u - u_prev, p=1) < self.tol:
                break

        return torch.exp(u[:, None] + v[None, :] + K_log)   # π = diag(e^u) K diag(e^v)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        node_histograms: torch.Tensor,   # (N, K)
        batch_idx:       torch.Tensor,   # (N,)
        codebook:        torch.Tensor    # (K, d)
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # -- ensure we have D (K×K)
        if self.atom_distances is None or self.atom_distances.size(0) != codebook.size(0):
            self._precompute_atom_distances(codebook)

        K = codebook.size(0)
        B = int(batch_idx.max().item()) + 1
        device, dtype = codebook.device, codebook.dtype

        graph_hists = []

        for b in range(B):
            mask = (batch_idx == b)
            V = mask.sum().item()

            if V == 0:   # empty graph → uniform
                graph_hists.append(torch.full((K,), 1.0 / K, device=device, dtype=dtype))
                continue

            H = node_histograms[mask]                  # (V, K)

            # Cost matrix M_{vk} = h_v · D[:,k}
            M = H @ self.atom_distances                # (V, K)

            a = torch.full((V,), 1.0 / V, device=device, dtype=dtype)
            b = torch.full((K,), 1.0 / K, device=device, dtype=dtype)

            π = self._sinkhorn_log(a, b, M)            # (V, K)

            hist = π.sum(dim=0)                        # column sums
            hist = hist / (hist.sum() + 1e-12)
            graph_hists.append(hist)

        graph_hists = torch.stack(graph_hists, dim=0)           # (B, K)
        graph_embs  = graph_hists @ codebook                    # (B, d)

        return graph_hists, graph_embs


class HierarchicalPooling(nn.Module):
    """
    Hierarchical Optimal Transport Pooling with two stages:
    
    1. Node samples → Node histograms (using encoder geometry)
    2. Node histograms → Graph histogram (using codebook atom geometry)
    
    Both stages share a single learnable codebook for consistency.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        codebook_size: int = 32,
        epsilon_node: float = 0.3,
        epsilon_graph: float = 0.1,
        debug_mode: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.epsilon_node = epsilon_node
        self.epsilon_graph = epsilon_graph
        self.debug = debug_mode
        
        # Shared codebook
        self.codebook = Codebook(codebook_size, hidden_dim)
        
        # Use default values for max_iter and tol
        self.node_pool = NodeSinkhornPooling(epsilon_node, max_iter=100, tol=1e-6)
        self.graph_pool = GraphSinkhornPooling(epsilon_graph, max_iter=100, tol=1e-6)
        
        logger.info(f"Initialized HierarchicalPooling: {codebook_size} atoms, "
                   f"ε_node={epsilon_node}, ε_graph={epsilon_graph}")
    
    def forward(self, node_distributions: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical pooling forward pass.
        
        Args:
            node_distributions: Node samples [N, S, d]
            batch_idx: Batch indices [N]
            
        Returns:
            graph_histograms: Graph histograms [B, K] (compatible with existing interface)
        """
        # Get shared codebook
        C = self.codebook()  # [K, d]
        
        # Stage 1: Node samples → Node histograms
        node_histograms = self.node_pool(node_distributions, C)  # [N, K]
        
        # Stage 2: Node histograms → Graph histograms + embeddings
        graph_histograms, graph_embeddings = self.graph_pool(node_histograms, batch_idx, C)
        
        # Debug logging
        if self.debug:
            self._log_debug_info(node_histograms, graph_histograms, graph_embeddings)
        
        # Store intermediate outputs for potential use in regularization
        self._last_node_histograms = node_histograms
        self._last_graph_embeddings = graph_embeddings
        
        # Return graph histograms (compatible with existing BaryGNN interface)
        return graph_histograms
    
    def _log_debug_info(self, node_hists: torch.Tensor, graph_hists: torch.Tensor, 
                       graph_embs: torch.Tensor):
        """Log debug information about the pooling process."""
        # Node histogram statistics
        node_entropy = -torch.sum(node_hists * torch.log(node_hists + 1e-8), dim=1)
        logger.debug(f"Node histogram entropy - mean: {node_entropy.mean():.4f}, "
                    f"std: {node_entropy.std():.4f}")
        
        # Graph histogram statistics
        graph_entropy = -torch.sum(graph_hists * torch.log(graph_hists + 1e-8), dim=1)
        logger.debug(f"Graph histogram entropy - mean: {graph_entropy.mean():.4f}, "
                    f"std: {graph_entropy.std():.4f}")
        
        # Codebook usage
        codebook_usage = graph_hists.sum(dim=0)  # [K]
        logger.debug(f"Codebook usage variance: {codebook_usage.var():.4f}")
        
        # Graph embedding statistics
        logger.debug(f"Graph embedding norm - mean: {torch.norm(graph_embs, dim=1).mean():.4f}")