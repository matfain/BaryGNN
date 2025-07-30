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
    Stage 1: Transform node samples into histograms over the shared codebook.
    
    For each node, transport its S samples to K codebook atoms using Sinkhorn OT.
    Uses encoder feature geometry (Euclidean distance).
    """
    
    def __init__(self, epsilon_node: float = 0.3, max_iter: int = 100, tol: float = 1e-6):
        super().__init__()
        self.epsilon_node = epsilon_node
        self.max_iter = max_iter
        self.tol = tol
        
        # GeomLoss Sinkhorn solver for node-level transport
        self.sinkhorn = SamplesLoss(
            "sinkhorn", 
            p=2, 
            blur=epsilon_node,
            scaling=0.9,
            backend="auto"
        )
        
        logger.info(f"Initialized NodeSinkhornPooling with epsilon={epsilon_node}")
    
    def forward(self, samples: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """
        Transport node samples to codebook atoms.
        
        Args:
            samples: Node samples [N, S, d]
            codebook: Codebook atoms [K, d]
            
        Returns:
            node_histograms: Node histograms over codebook [N, K]
        """
        N, S, d = samples.shape
        K, _ = codebook.shape
        
        node_histograms = []
        
        for i in range(N):
            # Get samples for this node
            node_samples = samples[i]  # [S, d]
            
            # Uniform weights
            a = torch.ones(S, device=samples.device, dtype=samples.dtype) / S
            b = torch.ones(K, device=codebook.device, dtype=codebook.dtype) / K
            
            # Compute transport plan using GeomLoss
            transport_plan = self.sinkhorn.transport(a, node_samples, b, codebook)  # [S, K]
            
            # Get histogram as column sum
            histogram = transport_plan.sum(dim=0)  # [K]
            histogram = histogram / histogram.sum()  # Normalize
            
            node_histograms.append(histogram)
        
        return torch.stack(node_histograms, dim=0)  # [N, K]


class GraphSinkhornPooling(nn.Module):
    """
    Stage 2: Transform node histograms into graph histogram using Wasserstein-on-atoms.
    
    Uses pre-computed atom-atom distances to define costs between node histograms
    and canonical simplex points. No nested Sinkhorn - just matrix-vector operations.
    """
    
    def __init__(self, epsilon_graph: float = 0.1, max_iter: int = 100, tol: float = 1e-6):
        super().__init__()
        self.epsilon_graph = epsilon_graph
        self.max_iter = max_iter
        self.tol = tol
        
        # GeomLoss Sinkhorn solver for graph-level transport
        self.sinkhorn = SamplesLoss(
            "sinkhorn",
            p=2,
            blur=epsilon_graph,
            scaling=0.9,
            backend="auto"
        )
        
        # Will store pre-computed atom-atom distances
        self.register_buffer('atom_distances', None)
        
        logger.info(f"Initialized GraphSinkhornPooling with epsilon={epsilon_graph}")
    
    def _precompute_atom_distances(self, codebook: torch.Tensor):
        """Pre-compute pairwise distances between codebook atoms."""
        # D[i,j] = ||C_i - C_j||^2
        self.atom_distances = torch.cdist(codebook, codebook, p=2).pow(2)  # [K, K]
        logger.debug(f"Pre-computed atom distances matrix: {self.atom_distances.shape}")
    
    def forward(self, node_histograms: torch.Tensor, batch_idx: torch.Tensor, 
                codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transport node histograms to graph histogram using atom geometry.
        
        Args:
            node_histograms: Node histograms [N, K]
            batch_idx: Batch indices [N]
            codebook: Codebook atoms [K, d]
            
        Returns:
            graph_histograms: Graph histograms [B, K]
            graph_embeddings: Graph embeddings [B, d]
        """
        # Pre-compute atom distances if needed
        if self.atom_distances is None or self.atom_distances.shape[0] != codebook.shape[0]:
            self._precompute_atom_distances(codebook)
        
        B = int(batch_idx.max().item()) + 1
        K = codebook.shape[0]
        
        graph_histograms = []
        
        for b in range(B):
            # Get node histograms for this graph
            mask = batch_idx == b
            graph_node_hists = node_histograms[mask]  # [|V_g|, K]
            
            if graph_node_hists.numel() == 0:
                # Empty graph - use uniform histogram
                uniform_hist = torch.ones(K, device=codebook.device, dtype=codebook.dtype) / K
                graph_histograms.append(uniform_hist)
                continue
            
            num_nodes = graph_node_hists.shape[0]
            
            # Compute cost matrix M[i,k] = w_i @ D[:,k]
            # This is the expected distance from node i's histogram to atom k
            cost_matrix = torch.matmul(graph_node_hists, self.atom_distances)  # [|V_g|, K]
            
            # Uniform weights for nodes and atoms
            p = torch.ones(num_nodes, device=codebook.device, dtype=codebook.dtype) / num_nodes
            q = torch.ones(K, device=codebook.device, dtype=codebook.dtype) / K
            
            # Use GeomLoss with pre-computed cost matrix
            # We need to create dummy point clouds that will give us the desired cost matrix
            # For now, use a simpler approach with manual Sinkhorn
            transport_plan = self._manual_sinkhorn(p, q, cost_matrix)  # [|V_g|, K]
            
            # Graph histogram as column sum
            graph_hist = transport_plan.sum(dim=0)  # [K]
            graph_hist = graph_hist / graph_hist.sum()  # Normalize
            
            graph_histograms.append(graph_hist)
        
        graph_histograms = torch.stack(graph_histograms, dim=0)  # [B, K]
        
        # Compute graph embeddings as weighted sum of codebook atoms
        graph_embeddings = torch.matmul(graph_histograms, codebook)  # [B, d]
        
        return graph_histograms, graph_embeddings
    
    def _manual_sinkhorn(self, a: torch.Tensor, b: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Manual Sinkhorn iteration with pre-computed cost matrix.
        
        Args:
            a: Source weights [n]
            b: Target weights [m] 
            M: Cost matrix [n, m]
            
        Returns:
            transport_plan: Transport plan [n, m]
        """
        n, m = M.shape
        
        # Initialize dual variables
        u = torch.zeros(n, device=M.device, dtype=M.dtype)
        v = torch.zeros(m, device=M.device, dtype=M.dtype)
        
        # Sinkhorn iterations
        K = torch.exp(-M / self.epsilon_graph)  # [n, m]
        
        for _ in range(self.max_iter):
            u_prev = u.clone()
            
            # Update u
            u = torch.log(a + 1e-8) - torch.logsumexp(
                torch.log(K + 1e-8) + v.unsqueeze(0), dim=1
            )
            
            # Update v  
            v = torch.log(b + 1e-8) - torch.logsumexp(
                torch.log(K + 1e-8) + u.unsqueeze(1), dim=0
            )
            
            # Check convergence
            if torch.norm(u - u_prev) < self.tol:
                break
        
        # Compute transport plan
        transport_plan = torch.exp(u.unsqueeze(1) + v.unsqueeze(0)) * K
        
        return transport_plan


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
        max_iter: int = 100,
        tol: float = 1e-6,
        debug_mode: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.debug_mode = debug_mode
        
        # Shared codebook
        self.codebook = Codebook(codebook_size, hidden_dim)
        
        # Two-stage pooling
        self.node_pool = NodeSinkhornPooling(epsilon_node, max_iter, tol)
        self.graph_pool = GraphSinkhornPooling(epsilon_graph, max_iter, tol)
        
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
        if self.debug_mode:
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