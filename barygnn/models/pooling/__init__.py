from .geomloss_pooling import GeomLossBarycentricPooling
from .hier_sinkhorn import HierarchicalPooling

def create_barycentric_pooling(backend="geomloss", **kwargs):
    backend = backend.lower()
    if backend == "geomloss":
        return GeomLossBarycentricPooling(**kwargs)
    elif backend == "hierarchical":
        return HierarchicalPooling(**kwargs)
    else:
        raise ValueError(f"Unknown OT backend '{backend}'. Choose from: 'geomloss', 'hierarchical'")

__all__ = [
    "GeomLossBarycentricPooling",
    "HierarchicalPooling",
    "create_barycentric_pooling",
]
