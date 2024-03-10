import jax.numpy as jnp
from scipy.spatial.distance import pdist, cdist, squareform


class RBF:
    def __init__(self, sigma) -> None:
        self.sigma = sigma

    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray = None) -> jnp.ndarray:
        if Y is None:
            dist2 = pdist(X, metric="sqeuclidean")
            dist2 = squareform(dist2)
        else:
            dist2 = cdist(X, Y, metric="sqeuclidean")

        return jnp.exp(-dist2 / (2 * (self.sigma**2)))


class PolyKernel:
    def __init__(self, degree: float = 1.0) -> None:
        self.degree = degree

    def __call__(self, X: jnp.ndarray, Y: jnp.ndarray = None):
        if Y is None:
            Y = X

        return jnp.dot(X, Y.T) ** self.degree
