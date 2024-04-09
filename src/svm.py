from functools import partial
from typing import Mapping
import jax
import jax.numpy as jnp
from jaxopt import BoxCDQP
from src.utils import timeit



def solve_svm(K: jnp.array, y: jnp.array, C: float):
    qp = BoxCDQP()
    lower_bound = jnp.where(y == 1, 0, -C)
    upper_bound = jnp.where(y == 1, C, 0)
    init = jnp.zeros(y.shape[0])
    sol = qp.run(init, params_obj=(K, -y), params_ineq=(lower_bound, upper_bound))
    return sol.params


vect_solve_svm = jax.jit(jax.vmap(solve_svm, in_axes=(None, 0, None), out_axes=0))


class MultiClassKernelSVM:
    def __init__(
        self,
        num_classes: int,
        kernel_func: Mapping,
        c: float,
        comp_num=None,
        threshold=0,
    ):
        self.num_classes = num_classes
        self.c = c
        self.comp_num = comp_num

        self.kernel = kernel_func

        self.threshold = threshold
        self.full_inference = (threshold == 0) and (comp_num == None)

    @timeit
    def fit(self, X, y):
        y_onehot = jax.nn.one_hot(y, num_classes=self.num_classes, axis=0)
        y_onehot = 2 * y_onehot - 1
        K = self.kernel(X)
        alphas = vect_solve_svm(K, y_onehot, self.c)
        if self.full_inference:
            self.alpha_pred = alphas
            self.ref_points = X
        elif self.threshold > 0:
            self.alpha_pred = []
            self.ref_points = []
            for i in range(self.num_classes):
                mask = alphas[i] > self.threshold
                self.alpha_pred.append(alphas[i][mask])
                self.ref_points.append(X[mask])

        elif self.comp_num is not None:
            sort_index = jnp.argsort(-jnp.abs(alphas), axis=1)[:, : self.comp_num]
            self.alpha_pred = []
            self.ref_points = []
            for i in range(self.num_classes):
                self.alpha_pred.append(alphas[i, sort_index[i]])
                self.ref_points.append(X[sort_index[i], :])

    @timeit
    def predict(self, X: jnp.array):
        if X.ndim < 2:
            X = X[None, :]

        if self.full_inference:
            kern_comp = self.kernel(X, self.ref_points)
            prob = kern_comp @ self.alpha_pred.T
            preds = jnp.argsort(-prob, axis=1)[:, 0]

        else:
            prob = []
            for i in range(self.num_classes):
                kern_comp = self.kernel(X, self.ref_points[i])
                prob.append(kern_comp @ self.alpha_pred[i])
            prob = jnp.array(prob)
            preds = jnp.argsort(-prob, axis=0)[0, :]
        X = jnp.squeeze(X)
        return preds
