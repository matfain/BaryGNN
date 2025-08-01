import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Literal, Union
import logging

# Set up logger
logger = logging.getLogger(__name__)


class EnhancedMLP(nn.Module):
    """
    Enhanced Multi-layer perceptron for classification with advanced features.
    
    Features:
    - Configurable layer widths (e.g., decreasing pattern)
    - Residual connections for better gradient flow
    - Multiple activation functions (ReLU, LeakyReLU, GELU, Swish)
    - Choice of normalization (BatchNorm, LayerNorm, or None)
    - Dropout with different rates per layer
    - Skip connections where dimensionally feasible
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Union[List[int], int] = [256, 128, 64],
        out_dim: int = 2,
        dropout: Union[float, List[float]] = 0.2,
        activation: Literal["relu", "leaky_relu", "gelu", "swish"] = "relu",
        norm_type: Optional[Literal["batch", "layer"]] = "batch",
        use_residual: bool = True,
        residual_type: Literal["add", "concat"] = "add",
        final_dropout: float = 0.5,
    ):
        """
        Initialize the enhanced MLP.
        
        Args:
            in_dim: Input dimension
            hidden_dims: Hidden layer dimensions (list) or single dimension (int)
            out_dim: Output dimension (number of classes)
            dropout: Dropout probability (float) or per-layer rates (list)
            activation: Activation function type
            norm_type: Normalization type ("batch", "layer", or None)
            use_residual: Whether to use residual connections
            residual_type: Type of residual connection ("add" or "concat")
            final_dropout: Dropout rate before final layer
        """
        super(EnhancedMLP, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual
        self.residual_type = residual_type
        self.final_dropout = final_dropout
        
        # Handle hidden dimensions
        if isinstance(hidden_dims, int):
            # If single int provided, create 3 layers with decreasing width
            hidden_dims = [hidden_dims, hidden_dims // 2, hidden_dims // 4]
        self.hidden_dims = hidden_dims
        
        # Handle dropout rates
        if isinstance(dropout, float):
            self.dropout_rates = [dropout] * len(hidden_dims)
        else:
            assert len(dropout) == len(hidden_dims), "Dropout list must match hidden_dims length"
            self.dropout_rates = dropout
        
        logger.info(f"Initializing EnhancedMLP: {in_dim} -> {hidden_dims} -> {out_dim}")
        logger.info(f"Activation: {activation}, Norm: {norm_type}, Residual: {use_residual}")
        
        # Set activation function
        self.activation_fn = self._get_activation(activation)
        
        # Build layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_layers = nn.ModuleList() if use_residual else None
        
        layer_dims = [in_dim] + hidden_dims
        
        for i in range(len(hidden_dims)):
            prev_dim = layer_dims[i]
            curr_dim = layer_dims[i + 1]
            
            # Main linear layer
            self.layers.append(nn.Linear(prev_dim, curr_dim))
            
            # Normalization layer
            if norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(curr_dim))
            elif norm_type == "layer":
                self.norms.append(nn.LayerNorm(curr_dim))
            else:
                self.norms.append(nn.Identity())
            
            # Residual connection layer
            if use_residual:
                if residual_type == "add" and prev_dim == curr_dim:
                    # Direct addition when dimensions match
                    self.residual_layers.append(nn.Identity())
                elif residual_type == "add" and prev_dim != curr_dim:
                    # Linear projection for dimension matching
                    self.residual_layers.append(nn.Linear(prev_dim, curr_dim))
                elif residual_type == "concat":
                    # Concatenation doesn't need special handling here
                    self.residual_layers.append(nn.Identity())
        
        # Output layer
        final_input_dim = hidden_dims[-1]
        if use_residual and residual_type == "concat":
            # For concat residuals, we need to account for concatenated dimensions
            # This is a simplified version - in practice, you'd track all concat dims
            final_input_dim = hidden_dims[-1]  # Keeping simple for now
        
        self.output_layer = nn.Linear(final_input_dim, out_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function."""
        if activation == "relu":
            return F.relu
        elif activation == "leaky_relu":
            return F.leaky_relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "swish":
            return lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of the enhanced MLP.
        
        Args:
            x: Input features [batch_size, in_dim]
            
        Returns:
            logits: Classification logits [batch_size, out_dim]
        """
        # Apply hidden layers
        for i, (layer, norm, dropout_rate) in enumerate(zip(self.layers, self.norms, self.dropout_rates)):
            # Store input for potential residual connection
            if self.use_residual:
                residual = x
            
            # Apply main transformation
            x = layer(x)
            x = norm(x)
            x = self.activation_fn(x)
            
            # Apply residual connection
            if self.use_residual:
                if self.residual_type == "add":
                    # Project residual if needed and add
                    residual_proj = self.residual_layers[i](residual)
                    x = x + residual_proj
                elif self.residual_type == "concat":
                    # Concatenate with residual (simplified version)
                    # In a full implementation, you'd handle dimension tracking
                    pass  # Skip concat for simplicity in this version
            
            # Apply dropout
            x = F.dropout(x, p=dropout_rate, training=self.training)
        
        # Apply final dropout before output
        x = F.dropout(x, p=self.final_dropout, training=self.training)
        
        # Apply output layer
        logits = self.output_layer(x)
        
        return logits


class AdaptiveMLP(nn.Module):
    """
    Adaptive MLP that automatically adjusts architecture based on input dimension.
    
    This version automatically scales the hidden dimensions based on the input size
    and provides sensible defaults for different input dimension ranges.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 2,
        depth_factor: float = 1.0,
        width_factor: float = 1.0,
        dropout: float = 0.2,
        activation: str = "relu",
        norm_type: str = "batch",
    ):
        """
        Initialize the adaptive MLP.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension (number of classes)
            depth_factor: Multiplier for number of layers
            width_factor: Multiplier for layer widths
            dropout: Dropout probability
            activation: Activation function type
            norm_type: Normalization type
        """
        super(AdaptiveMLP, self).__init__()
        
        # Determine architecture based on input dimension
        if in_dim <= 64:
            # Small input: simple architecture
            base_hidden_dims = [128, 64]
        elif in_dim <= 256:
            # Medium input: moderate architecture
            base_hidden_dims = [512, 256, 128]
        elif in_dim <= 1024:
            # Large input: deeper architecture
            base_hidden_dims = [1024, 512, 256, 128]
        else:
            # Very large input: very deep architecture
            base_hidden_dims = [2048, 1024, 512, 256, 128]
        
        # Apply scaling factors
        num_layers = max(2, int(len(base_hidden_dims) * depth_factor))
        hidden_dims = [max(32, int(dim * width_factor)) for dim in base_hidden_dims[:num_layers]]
        
        logger.info(f"AdaptiveMLP auto-configured: {in_dim} -> {hidden_dims} -> {out_dim}")
        
        # Use the enhanced MLP with determined architecture
        self.mlp = EnhancedMLP(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            dropout=dropout,
            activation=activation,
            norm_type=norm_type,
            use_residual=True
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.mlp(x)


class ResidualBlock(nn.Module):
    """
    Residual block for deeper MLPs.
    """
    
    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: str = "batch"
    ):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        
        if norm_type == "batch":
            self.norm1 = nn.BatchNorm1d(dim)
            self.norm2 = nn.BatchNorm1d(dim)
        elif norm_type == "layer":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        self.dropout = dropout
        self.activation_fn = EnhancedMLP._get_activation(None, activation)
    
    def forward(self, x):
        """Forward pass with residual connection."""
        residual = x
        
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.linear2(x)
        x = self.norm2(x)
        
        # Add residual
        x = x + residual
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class DeepResidualMLP(nn.Module):
    """
    Deep MLP with residual blocks for very complex classification tasks.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        out_dim: int = 2,
        dropout: float = 0.2,
        activation: str = "relu",
        norm_type: str = "batch"
    ):
        """
        Initialize deep residual MLP.
        
        Args:
            in_dim: Input dimension
            hidden_dim: Hidden dimension for residual blocks
            num_blocks: Number of residual blocks
            out_dim: Output dimension
            dropout: Dropout probability
            activation: Activation function
            norm_type: Normalization type
        """
        super(DeepResidualMLP, self).__init__()
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout, activation, norm_type)
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, out_dim)
        
        logger.info(f"DeepResidualMLP: {in_dim} -> {hidden_dim} (x{num_blocks} blocks) -> {out_dim}")
    
    def forward(self, x):
        """Forward pass."""
        # Input projection
        x = self.input_proj(x)
        
        # Apply residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits


def create_classifier(
    classifier_type: str = "enhanced",
    **kwargs
) -> nn.Module:
    """
    Factory function to create classification heads.
    
    Args:
        classifier_type: Type of classifier ("simple", "enhanced", "adaptive", "deep_residual")
        **kwargs: Additional arguments for the classifier
        
    Returns:
        Classifier instance
    """
    if classifier_type == "simple":
        from barygnn.models.classification.mlp import MLP
        simple_kwargs = {
            'in_dim': kwargs.get('in_dim'),
            'hidden_dim': kwargs.get('hidden_dim'),
            'out_dim': kwargs.get('out_dim'),
            'num_layers': kwargs.get('num_layers', 2),
            'dropout': kwargs.get('dropout', 0.2)
        }
        return MLP(**simple_kwargs)
    elif classifier_type == "enhanced":
        return EnhancedMLP(**kwargs)
    elif classifier_type == "adaptive":
        return AdaptiveMLP(**kwargs)
    elif classifier_type == "deep_residual":
        return DeepResidualMLP(**kwargs)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}. "
                        f"Choose from ['simple', 'enhanced', 'adaptive', 'deep_residual']") 