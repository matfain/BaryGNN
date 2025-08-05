from .geomloss_barycenter import BarycentricPooling
from .regular_pooling import RegularPooling

def create_barycentric_pooling(backend="barycenter", **kwargs):
    backend = backend.lower()
    if backend == "barycenter":
        return BarycentricPooling(**kwargs)
    elif backend == "regular_pooling":
        return RegularPooling(**kwargs)
    else:
        raise ValueError(f"Unknown pooling backend '{backend}'. Choose from: 'barycenter', 'regular_pooling'")

__all__ = [
    "BarycentricPooling",
    "RegularPooling",
    "create_barycentric_pooling",
]
