# BaryGNN Pooling

This directory contains the implementation of the barycentric pooling mechanism used in BaryGNN.

## Barycentric Pooling

The `BarycentricPooling` class implements a Wasserstein barycenter-based pooling mechanism with two key features:

1. **Learnable Codebook Prior**: Instead of using a uniform prior for the codebook atoms, this implementation uses a learnable prior distribution, allowing the model to adjust the importance of different codebook atoms.

2. **True Wasserstein Barycenter**: Computes the actual Wasserstein barycenter by:
   - Computing OT between each node's distribution and the codebook separately
   - Averaging the resulting histograms across all nodes in each graph

## Usage

To use the barycentric pooling in your model, update your configuration file:

```yaml
model:
  pooling:
    backend: "barycenter"  # Currently the only supported backend
    codebook_size: 32      # Number of codebook atoms
    epsilon: 0.2           # Sinkhorn regularization parameter
    p: 2                   # Order of Wasserstein distance
    scaling: 0.9           # Scaling factor for Sinkhorn algorithm
```

## Implementation Details

### Learnable Codebook Prior

The prior distribution over codebook atoms is parameterized in log-space and normalized using softmax to ensure it's a valid probability distribution:

```python
# Learnable codebook prior (initialized as uniform)
self.log_codebook_prior = nn.Parameter(torch.zeros(codebook_size))

@property
def codebook_prior(self):
    """Get the normalized codebook prior distribution."""
    return F.softmax(self.log_codebook_prior, dim=0)
```

### Wasserstein Barycenter Computation

The implementation computes the OT histogram for each node separately and then averages them per graph:

```python
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
```

## Mathematical Foundation

The Wasserstein barycenter is defined as the minimizer of the sum of squared Wasserstein distances:

$$\mu^* = \arg\min_{\mu} \sum_{i=1}^{n} W_2^2(\mu_i, \mu)$$

where $\mu_i$ are the node distributions and $\mu$ is the barycenter.

In our implementation, we compute the optimal transport plan between each node distribution and the codebook, then average the resulting histograms to approximate the barycenter.