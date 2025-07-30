from .pot_pooling import POTBarycentricPooling
from .geomloss_pooling import GeomLossBarycentricPooling

def create_barycentric_pooling(backend="pot", **kwargs):
    backend = backend.lower()
    if backend == "geomloss":
        return GeomLossBarycentricPooling(**kwargs)
    elif backend == "pot":
        return POTBarycentricPooling(**kwargs)
    else:
        raise ValueError(f"Unknown OT backend '{backend}'")

__all__ = [
    "POTBarycentricPooling",
    "GeomLossBarycentricPooling",
    "create_barycentric_pooling",
]
