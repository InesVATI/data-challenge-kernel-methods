import jax
import jax.numpy as jnp
from jax.experimental import host_callback


def check_objective_value(arg, transforms):
    iteration, objective = arg
    if (iteration + 1) % 10 == 0:
        print(
            "Spherical K means at iter %d :  mean cosine similarity %.4f"
            % (iteration + 1, objective)
        )


@jax.jit
def choose_randomly(X, mask):
    key = jax.random.PRNGKey(0)
    key, _ = jax.random.split(key)
    return jax.random.choice(key, X, axis=0)


@jax.jit
def compute_sum(X, mask):
    return jnp.sum(X * mask[:, None], axis=0)


@jax.jit
def cluster_update(cluster_id: int, assignments: jnp.ndarray, X: jnp.ndarray):
    mask = assignments == cluster_id
    mask = mask.astype(jnp.int32)
    # nb_elem = jnp.sum(mask)

    # centroid = jax.lax.cond(nb_elem==0, choose_randomly, compute_sum, X, mask)
    centroid = jnp.sum(X * mask[:, None], axis=0)

    norm_centroid = jnp.linalg.norm(centroid, axis=-1)
    return centroid / norm_centroid


@jax.jit
def step(X, centroids, prev_objective: float):
    """X should have normalized row"""
    cluster_update_vmap = jax.vmap(cluster_update, in_axes=(0, None, None))

    nb_clusters = centroids.shape[0]

    cos_sim = jnp.dot(X, centroids.T)
    assignments = jnp.argmax(cos_sim, axis=1)

    cluster_indices = jnp.arange(nb_clusters)
    new_centroids = cluster_update_vmap(cluster_indices, assignments, X)

    objective = jnp.take_along_axis(
        cos_sim, jnp.expand_dims(assignments, axis=-1), axis=-1
    ).mean()
    stop_criteria = jnp.abs(prev_objective - objective) / (jnp.abs(objective) + 1e-20)

    return new_centroids, assignments, objective, stop_criteria


class SphericalKMeans:
    def __init__(
        self, nb_clusters: int, max_iter: int = 500, stop_cond: float = 1e-6
    ) -> None:

        self.nb_clusters = nb_clusters
        self.max_iter = max_iter
        self.stop_cond = stop_cond
        self.key = jax.random.PRNGKey(0)

    def initialize_random_centroids(self, X: jnp.ndarray):
        N = X.shape[0]
        self.key, subkey = jax.random.split(self.key)
        rd_ind = jax.random.choice(self.key, N, (self.nb_clusters,), replace=False)
        return X[rd_ind]

    def _main_loop(self, X, centroids):

        @jax.jit
        def while_step(arg):
            iteration, centroids, assignments, prev_objective, stop_criteria = arg
            new_centroids, assignments, objective, stop_criteria = step(
                X, centroids, prev_objective
            )

            host_callback.id_tap(check_objective_value, (iteration, objective))

            return (iteration + 1, new_centroids, assignments, objective, stop_criteria)

        @jax.jit
        def cond(arg):
            iteration, centroids, assignments, objective, stop_criteria = arg

            return (iteration < self.max_iter) & (stop_criteria > self.stop_cond)

        assignments = jnp.zeros(X.shape[0], dtype=jnp.int32)

        iteration, centroids, assignments, objective, stop_criteria = (
            jax.lax.while_loop(
                cond, while_step, (0, centroids, assignments, jnp.inf, jnp.inf)
            )
        )

        return centroids, assignments

    def fit(self, X, init_centroids: jnp.ndarray = None):
        """X should have normalized row"""

        # normalize X to have data on unit sphere
        X = jnp.asarray(X)

        if init_centroids is None:
            init_centroids = self.initialize_random_centroids(X)

        print("Spherical Kmean main loop starts")
        centroids, assignments = self._main_loop(X, init_centroids)
        print("End of main loop")

        return centroids, assignments
