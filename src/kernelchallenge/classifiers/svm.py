from functools import partial
from typing import Mapping
import jax
import jax.numpy as jnp
from jax import vmap
from jaxopt import BoxCDQP, ProjectedGradient, projection

from kernelchallenge.utils import timeit


def rbf_kernel(x: jnp.array, x_prime: jnp.array, gamma=1):
    return jnp.exp(-0.5 * gamma * jnp.linalg.norm(x - x_prime) ** 2)


def poly_kernel(x: jnp.array, x_prime: jnp.array, a: float):
    return jnp.dot(x, x_prime) ** a


def get_poly_kernel(a):
    return partial(poly_kernel, a=a)


def old_solve_svm(K: jnp.array, y: jnp.array, C: float):

    def objective_fun(beta, K, y):
        # print(K.shape, beta.shape, y.shape)
        return 0.5 * jnp.dot(jnp.dot(beta, K), beta) - jnp.dot(beta, y)

    # TODO add it if intercept needed
    # w = jnp.zeros(y.shape[0])

    def proj(beta, C):
        box_lower = jnp.where(y == 1, 0, -C)
        box_upper = jnp.where(y == 1, C, 0)
        proj_params = (box_lower, box_upper)
        return projection.projection_box(beta, proj_params)

    # Run solver.
    beta_init = jnp.ones(y.shape[0])
    solver = ProjectedGradient(
        fun=objective_fun, projection=proj, tol=tol, maxiter=500, verbose=verbose
    )
    beta_fit = solver.run(beta_init, hyperparams_proj=C, K=K, y=y).params

    return beta_fit


def solve_svm(K: jnp.array, y: jnp.array, C: float):
    qp = BoxCDQP()
    lower_bound = jnp.where(y == 1, 0, -C)
    upper_bound = jnp.where(y == 1, C, 0)
    init = jnp.zeros(y.shape[0])
    sol = qp.run(init, params_obj=(K, -y), params_ineq=(lower_bound, upper_bound))
    return sol.params


vect_solve_svm = jax.jit(jax.vmap(solve_svm, in_axes=(None, 0, None), out_axes=0))


class BinarySVM:
    def __init__(self, kernel_func: Mapping, c: float, threshold=0):
        self.c = c
        self.threshold = threshold
        self.kernel_vec = vmap(kernel_func, (None, 0), 0)
        self.kernel_mat = lambda X, Y: jnp.array([X[i] - Y for i in range(X.shape[0])])

    def fit(self, X, y):
        alpha = solve_svm(self.kernel_mat(X, X), y, self.c)

        if self.threshold > 0:
            threshold_mask = jnp.abs(alpha) > self.threshold
            x_pred = X[threshold_mask]
            self._alpha_pred = alpha[threshold_mask]
            self._kernel_pred = partial(self.kernel_mat, x_pred)
        else:
            self._alpha_pred = alpha
            self._kernel_pred = partial(self.kernel_mat, X)

    def predict(self, X):
        if X.ndim < 2:
            X = X[None, :]
        preds = self._alpha_pred @ self._kernel_pred(X)
        X = jnp.squeeze(X)
        return preds


class MultiClassSVM:
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

        self.kernel_vec = jax.jit(vmap(kernel_func, (None, 0), 0))
        self.kernel_mat = lambda X, Y: jnp.array(
            [self.kernel_vec(X[i], Y) for i in range(X.shape[0])]
        )

        self.threshold = threshold
        self.full_inference = (threshold == 0) and (comp_num == None)

    @timeit
    def fit(self, X, y):
        y_onehot = jax.nn.one_hot(y, num_classes=self.num_classes, axis=0)
        y_onehot = 2 * y_onehot - 1
        K = self.kernel_mat(X, X)
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
            kern_comp = self.kernel_mat(X, self.ref_points)
            prob = kern_comp @ self.alpha_pred.T
            preds = jnp.argsort(-prob, axis=1)[:, 0]

        else:
            prob = []
            for i in range(self.num_classes):
                kern_comp = self.kernel_mat(X, self.ref_points[i])
                prob.append(kern_comp @ self.alpha_pred[i])
            prob = jnp.array(prob)
            preds = jnp.argsort(-prob, axis=0)[0, :]
        X = jnp.squeeze(X)
        return preds


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
    