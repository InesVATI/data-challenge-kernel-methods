from jax import jit, vmap
import jax.numpy as jnp
import numpy as np


def gaus_pooling(X, shape, size):
    pass


def gaus_conv(X: jnp.array, beta: int):
    # expected chape of X:
    # n,m,m,d where n is the number of samples, m is the size of the grid, d the dimension per pixel
    grid_size = X.shape[1]
    new_grid_size = int(grid / beta)
    grid = jnp.zeros(shape=(grid_size, grid_size, 2))
    new_grid = jnp.zeros(shape=(new_grid_size, new_grid_size, 2))
    pass
