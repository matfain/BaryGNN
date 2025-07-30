import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union, List, Tuple, Literal
import logging

from barygnn.models.encoders.multi_head import create_multi_head_encoder
from barygnn.models.pooling import create_barycentric_pooling
from barygnn.models.readout.readout import Readout
from barygnn.models.classification.enhanced_mlp import create_classifier
from barygnn.losses.regularization import compute_distribution_regularization

# Set up logger
logger = logging.getLogger(__name__)


class BaryGNN(nn.Module):
    """
    BaryGNN v2: Enhanced Graph Neural Network with Barycentric Pooling
    
    This is the improved version that includes:
    - Multi-head encoders for true node distribution representation
    - GeomLoss-based Sinkhorn for numerical stability
    - Enhanced MLP classifiers with residual connections
    - Distribution regularization for meaningful node distributions
    """
    
    def __init__(
        self,
        # Data parameters
        in_dim: int,
        num_classes: int = 2,
        
        # Architecture parameters
        hidden_dim: int = 64,
        codebook_size: int = 16,
        distribution_size: int = 32,
        readout_type: str = "weighted_mean",
        
        # Encoder parameters
        encoder_type: str = "GIN",
        encoder_layers: int = 3,
        encoder_dropout: float = 0.5,
        multi_head_type: str = "efficient",  # "full" or "efficient"
        shared_layers: int = 1,
        
        # Pooling parameters  
        sinkhorn_epsilon: float = 0.2,
        max_iter: int = 100,
        tol: float = 1e-6,
        
        # Classification parameters
        classifier_type: str = "enhanced",  # "simple", "enhanced", "adaptive", "deep_residual"
        classifier_hidden_dims: Union[List[int], int] = [256, 128, 64],
        classifier_dropout: float = 0.2,
        classifier_activation: str = "relu",
        classifier_depth_factor: float = 1.0,  # For AdaptiveMLP
        classifier_width_factor: float = 1.0,  # For AdaptiveMLP
        
        # Regularization parameters
        use_distribution_reg: bool = True,
        reg_type: str = "variance",  # "variance", "centroid", "coherence"
        reg_lambda: float = 0.01,
        
        # General parameters
        debug_mode: bool = False,
        **kwargs
    ):
        """
        Initialize the BaryGNN model.
        
        Args:
            in_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for embeddings
            codebook_size: Number of atoms in the barycenter codebook
            distribution_size: Number of vectors in each node's empirical distribution
            readout_type: Type of readout ("weighted_mean" or "concat")
            encoder_type: Base encoder type ("GIN" or "GraphSAGE")
            encoder_layers: Number of GNN layers
            encoder_dropout: Dropout rate for encoder
            multi_head_type: Multi-head architecture type ("full" or "efficient")
            shared_layers: Number of shared layers in multi-head encoder
            sinkhorn_epsilon: Regularization parameter for Sinkhorn/POT
            max_iter: Maximum iterations for POT Sinkhorn
            tol: Convergence tolerance for POT
            classifier_type: Type of classifier
            classifier_hidden_dims: Hidden dimensions for classifier
            classifier_dropout: Dropout rate for classifier
            classifier_activation: Activation function for classifier
            use_distribution_reg: Whether to use distribution regularization
            reg_type: Type of distribution regularization
            reg_lambda: Regularization strength
            debug_mode: Whether to enable debug logging
        """
        super(BaryGNN, self).__init__()
        
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.distribution_size = distribution_size
        self.readout_type = readout_type
        self.use_distribution_reg = use_distribution_reg
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.debug_mode = debug_mode
        
        logger.info(f"Initializing BaryGNN with:")
        logger.info(f"  Architecture: {in_dim} -> {hidden_dim} -> {codebook_size} atoms -> {num_classes}")
        logger.info(f"  Encoder: {encoder_type} ({multi_head_type} multi-head, {distribution_size} heads)")
        logger.info(f"  Pooling: POT (ε={sinkhorn_epsilon})")
        logger.info(f"  Classifier: {classifier_type}")
        logger.info(f"  Regularization: {reg_type if use_distribution_reg else 'None'} (λ={reg_lambda})")
        
        # Multi-head encoder for node distribution representation
        self.encoder = create_multi_head_encoder(
            encoder_type=encoder_type,
            multi_head_type=multi_head_type,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_heads=distribution_size,
            shared_layers=shared_layers,
            num_layers=encoder_layers,
            dropout=encoder_dropout
        )
        
        # POT barycentric pooling with learned codebook
        pooling_kwargs = {
            'hidden_dim': hidden_dim,
            'codebook_size': codebook_size,
            'epsilon': sinkhorn_epsilon,
            'max_iter': max_iter,
            'tol': tol,
            'p': 2,
            'debug_mode': debug_mode
        }
        
        self.pooling = create_barycentric_pooling(**pooling_kwargs)
        
        # Readout layer
        self.readout = Readout(
            hidden_dim=hidden_dim,
            codebook_size=codebook_size,
            readout_type=readout_type
        )
        
        # Classification head
        classifier_input_dim = hidden_dim if readout_type == "weighted_mean" else hidden_dim * codebook_size
        
        classifier_kwargs = {
            'in_dim': classifier_input_dim,
            'out_dim': num_classes,
            'dropout': classifier_dropout,
            'activation': classifier_activation
        }
        
        if classifier_type == "enhanced":
            if isinstance(classifier_hidden_dims, list):
                classifier_kwargs['hidden_dims'] = classifier_hidden_dims
            else:
                classifier_kwargs['hidden_dims'] = [classifier_hidden_dims, classifier_hidden_dims // 2]
        elif classifier_type == "adaptive":
            # AdaptiveMLP doesn't use hidden_dims, it uses depth_factor and width_factor
            classifier_kwargs['depth_factor'] = classifier_depth_factor
            classifier_kwargs['width_factor'] = classifier_width_factor
        elif classifier_type == "deep_residual":
            classifier_kwargs['hidden_dim'] = classifier_hidden_dims if isinstance(classifier_hidden_dims, int) else classifier_hidden_dims[0]
            classifier_kwargs['num_blocks'] = 3
        elif classifier_type == "simple":
            classifier_kwargs['hidden_dim'] = classifier_hidden_dims if isinstance(classifier_hidden_dims, int) else classifier_hidden_dims[0]
            classifier_kwargs['num_layers'] = 2
        
        self.classifier = create_classifier(
            classifier_type=classifier_type,
            **classifier_kwargs
        )
        
        # Store parameters for loss computation
        self._store_intermediate_outputs = debug_mode or use_distribution_reg
        self._last_node_distributions = None
        
        # Count and log parameters
        self._log_parameter_count()
    
    def _log_parameter_count(self):
        """Count and log model parameters by component."""
        # Count parameters for each component
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        pooling_params = sum(p.numel() for p in self.pooling.parameters())
        readout_params = sum(p.numel() for p in self.readout.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        # Count trainable vs non-trainable
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        # Log detailed parameter breakdown
        logger.info("=" * 60)
        logger.info("MODEL PARAMETER BREAKDOWN")
        logger.info("=" * 60)
        logger.info(f"Multi-Head Encoder:     {encoder_params:>12,} parameters")
        logger.info(f"Barycentric Pooling:    {pooling_params:>12,} parameters")
        logger.info(f"Readout Layer:          {readout_params:>12,} parameters")
        logger.info(f"Classification Head:    {classifier_params:>12,} parameters")
        logger.info("-" * 60)
        logger.info(f"Total Parameters:       {total_params:>12,} parameters")
        logger.info(f"Trainable Parameters:   {trainable_params:>12,} parameters")
        logger.info(f"Non-trainable Params:   {non_trainable_params:>12,} parameters")
        logger.info("=" * 60)
        
        # Log parameter density information
        param_density = {
            'encoder_ratio': encoder_params / total_params * 100,
            'pooling_ratio': pooling_params / total_params * 100,
            'readout_ratio': readout_params / total_params * 100,
            'classifier_ratio': classifier_params / total_params * 100
        }
        
        logger.info("PARAMETER DISTRIBUTION:")
        logger.info(f"Encoder:        {param_density['encoder_ratio']:>6.1f}%")
        logger.info(f"Pooling:        {param_density['pooling_ratio']:>6.1f}%")
        logger.info(f"Readout:        {param_density['readout_ratio']:>6.1f}%")
        logger.info(f"Classifier:     {param_density['classifier_ratio']:>6.1f}%")
        logger.info("=" * 60)
        
        # Store for get_model_info method
        self._parameter_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'encoder_parameters': encoder_params,
            'pooling_parameters': pooling_params,
            'readout_parameters': readout_params,
            'classifier_parameters': classifier_params,
            'parameter_distribution': param_density
        }
    
    def _check_nan(self, tensor, name):
        """Check if tensor contains NaN values and log information."""
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            total_count = tensor.numel()
            logger.warning(f"NaN detected in {name}: {nan_count}/{total_count} values are NaN")
            return True
        return False
    
    def forward(self, batch):
        """
        Forward pass of the BaryGNN v2 model.
        
        Args:
            batch: PyG batch containing graph data
            
        Returns:
            logits: Classification logits
        """
        # Get input data
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        
        # Debug input
        if self.debug_mode:
            self._check_nan(x, "input features")
            logger.debug(f"Input shape: {x.shape}, Batch size: {batch_idx.max().item() + 1}")
        
        # Encode nodes to distributions using multi-head encoder
        node_distributions = self.encoder(x, edge_index)
        
        # Debug node distributions
        if self.debug_mode:
            self._check_nan(node_distributions, "node_distributions")
            logger.debug(f"Node distributions shape: {node_distributions.shape}")
            
            # Log distribution statistics
            dist_mean = node_distributions.mean(dim=1)  # Mean across distribution dimension
            dist_std = node_distributions.std(dim=1)    # Std across distribution dimension
            logger.debug(f"Distribution diversity - mean std: {dist_std.mean().item():.4f}, "
                        f"max std: {dist_std.max().item():.4f}")
        
        # Store for regularization loss computation
        if self._store_intermediate_outputs:
            self._last_node_distributions = node_distributions
        
        # Handle NaN values in node distributions
        if torch.isnan(node_distributions).any():
            logger.warning("NaN detected in node distributions, applying fallback")
            node_distributions = torch.nan_to_num(node_distributions, nan=0.0)
        
        # Apply barycentric pooling
        barycenter_weights = self.pooling(node_distributions, batch_idx)
        
        # Debug barycenter weights
        if self.debug_mode:
            self._check_nan(barycenter_weights, "barycenter_weights")
            logger.debug(f"Barycenter weights shape: {barycenter_weights.shape}")
            
            # Log weight statistics
            weight_entropy = -torch.sum(barycenter_weights * torch.log(barycenter_weights + 1e-8), dim=1)
            logger.debug(f"Weight entropy - mean: {weight_entropy.mean().item():.4f}, "
                        f"std: {weight_entropy.std().item():.4f}")
        
        # Handle NaN values in barycenter weights
        if torch.isnan(barycenter_weights).any():
            logger.warning("NaN detected in barycenter weights, using uniform fallback")
            batch_size = barycenter_weights.size(0)
            barycenter_weights = torch.ones(batch_size, self.codebook_size, 
                                          device=barycenter_weights.device) / self.codebook_size
        
        # Apply readout to get graph embeddings
        graph_embeddings = self.readout(barycenter_weights, self.pooling.codebook)
        
        # Debug graph embeddings
        if self.debug_mode:
            self._check_nan(graph_embeddings, "graph_embeddings")
            logger.debug(f"Graph embeddings shape: {graph_embeddings.shape}")
        
        # Handle NaN values in graph embeddings
        if torch.isnan(graph_embeddings).any():
            logger.warning("NaN detected in graph embeddings, applying fallback")
            graph_embeddings = torch.nan_to_num(graph_embeddings, nan=0.0)
        
        # Apply classification head
        logits = self.classifier(graph_embeddings)
        
        # Debug logits
        if self.debug_mode:
            self._check_nan(logits, "logits")
            logger.debug(f"Logits shape: {logits.shape}")
        
        # Handle NaN values in logits
        if torch.isnan(logits).any():
            logger.warning("NaN detected in logits, using uniform distribution")
            logits = torch.zeros_like(logits)
        
        return logits
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute distribution regularization loss.
        
        Returns:
            reg_loss: Regularization loss (scalar)
        """
        if not self.use_distribution_reg or self._last_node_distributions is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        try:
            reg_loss = compute_distribution_regularization(
                self._last_node_distributions,
                reg_type=self.reg_type,
                lambda_reg=self.reg_lambda
            )
            
            if self.debug_mode:
                logger.debug(f"Distribution regularization loss ({self.reg_type}): {reg_loss.item():.6f}")
            
            return reg_loss
        
        except Exception as e:
            logger.error(f"Error computing regularization loss: {str(e)}")
            return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def get_barycenter_weights(self, batch) -> torch.Tensor:
        """
        Get barycenter weights for analysis (without computing full forward pass).
        
        Args:
            batch: PyG batch containing graph data
            
        Returns:
            barycenter_weights: Barycenter weights [batch_size, codebook_size]
        """
        with torch.no_grad():
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
            node_distributions = self.encoder(x, edge_index)
            barycenter_weights = self.pooling(node_distributions, batch_idx)
            return barycenter_weights
    
    def get_node_distributions(self, batch) -> torch.Tensor:
        """
        Get node distributions for analysis.
        
        Args:
            batch: PyG batch containing graph data
            
        Returns:
            node_distributions: Node distributions [num_nodes, distribution_size, hidden_dim]
        """
        with torch.no_grad():
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
            node_distributions = self.encoder(x, edge_index)
            return node_distributions
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for logging and analysis.
        
        Returns:
            info: Dictionary with model information
        """
        # Use stored parameter info if available, otherwise compute
        if hasattr(self, '_parameter_info'):
            param_info = self._parameter_info
        else:
            # Fallback computation if _log_parameter_count wasn't called
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            param_info = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'non_trainable_parameters': total_params - trainable_params,
                'encoder_parameters': sum(p.numel() for p in self.encoder.parameters()),
                'pooling_parameters': sum(p.numel() for p in self.pooling.parameters()),
                'readout_parameters': sum(p.numel() for p in self.readout.parameters()),
                'classifier_parameters': sum(p.numel() for p in self.classifier.parameters()),
            }
        
        base_info = {
            'model_type': 'BaryGNN',
            'hidden_dim': self.hidden_dim,
            'codebook_size': self.codebook_size,
            'distribution_size': self.distribution_size,
            'readout_type': self.readout_type,
            'regularization': self.reg_type if self.use_distribution_reg else None,
            'encoder_type': type(self.encoder).__name__,
            'pooling_type': type(self.pooling).__name__,
            'classifier_type': type(self.classifier).__name__
        }
        
        # Merge parameter info with base info
        return {**base_info, **param_info}


# Factory function for backward compatibility and easy model creation
def create_barygnn(
    version: str = "v2",
    **kwargs
) -> nn.Module:
    """
    Factory function to create BaryGNN models.
    
    Args:
        version: Model version ("v1" or "v2")
        **kwargs: Model parameters
        
    Returns:
        BaryGNN model instance
    """
    # Since we only have one BaryGNN implementation now, version parameter is ignored
    if version in ["v1", "v2"]:
        return BaryGNN(**kwargs)
    else:
        raise ValueError(f"Unknown model version: {version}. Choose from ['v1', 'v2']") 