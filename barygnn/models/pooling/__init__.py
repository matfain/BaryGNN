from .geomloss_barycenter import BarycentricPooling

def create_barycentric_pooling(backend="barycenter", **kwargs):
    backend = backend.lower()
    if backend == "barycenter":
        return BarycentricPooling(**kwargs)
    else:
        raise ValueError(f"Unknown OT backend '{backend}'. Currently only 'barycenter' is supported.")

__all__ = [
    "BarycentricPooling",
    "create_barycentric_pooling",
]
