from .pot_pooling import POTBarycentricPooling
from .geomloss_pooling import GeomLossBarycentricPooling
from .hier_sinkhorn import HierarchicalPooling

def create_barycentric_pooling(backend="pot", **kwargs):
    backend = backend.lower()
    if backend == "geomloss":
        return GeomLossBarycentricPooling(**kwargs)
    elif backend == "pot":
        return POTBarycentricPooling(**kwargs)
    elif backend == "hierarchical":
        return HierarchicalPooling(**kwargs)
    else:
        raise ValueError(f"Unknown OT backend '{backend}'. Choose from: 'pot', 'geomloss', 'hierarchical'")

__all__ = [
    "POTBarycentricPooling",
    "GeomLossBarycentricPooling",
    "HierarchicalPooling",
    "create_barycentric_pooling",
]
