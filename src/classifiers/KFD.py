import numpy as np
from typing import Callable

def cosine_similarity(X: np.ndarray, Y: np.ndarray = None) -> np.ndarray:
    """ Return pairwise cosine similarity between X and Y
    Parameters
    ----------
    X : np.ndarray (n_samples, n_features) or (n_features, )
    Y : np.ndarray (n_samples, n_features) or (n_features, )
    Returns
    -------
    <X, Y> / (||X|| * ||Y||) : np.ndarray (n_samples, n_samples) or float
    """

    if X.ndim == 1:
        X = X.reshape(1, -1)
    if Y is not None :
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

    X = X / np.linalg.norm(X, axis=-1)[:, np.newaxis]
    if Y is None:
        Y = X
    else: 
        Y = Y / np.linalg.norm(Y, axis=-1)[:, np.newaxis]
    
    return np.dot(X, Y.T)

class KFD() : 
    """
    Kernel Fisher Discriminant
    """

    def __init__(self, kernel : Callable) -> None:
        self.kernel = kernel    

    def train(self, Y : np.ndarray, X : np.ndarray, eps : float=1e-9) -> None:
        """
        Compute Fisher direction
        """
        self.training_data, self.training_labels = X, Y

        self.labels, label_counts = np.unique(Y, return_counts=True)
        nlabels = len(self.labels)
        N = X.shape[0]

        # compute pairwise kernel matrix
        K = self.kernel(X)  # (N, N)
        total_mean = np.mean( K, axis=-1).reshape(-1, 1) # (N, 1)  

        M = np.zeros((N, N))
        N = np.zeros((N, N))

        for j in range(nlabels):
            # compute class kernel matrix
            class_mask = Y == self.labels[j]
            class_kernel_matrices = K[:, class_mask]
            class_means = np.mean(class_kernel_matrices, axis=-1).reshape(-1, 1) # (N, 1)
            centering_matrices = np.eye(label_counts[j]) - 1/label_counts[j] # (n_j, n_j)

            class_centered = class_means - total_mean
            M += class_centered @ class_centered.T # (N, N)
            N += class_kernel_matrices @ centering_matrices @ class_kernel_matrices.T

        try :
            S = np.linalg.inv(N) @ M
        except np.linalg.LinAlgError as e:
            print('detN', np.linalg.det(N))
            S = np.linalg.inv(N + eps * np.eye(N.shape[0])) @ M

        _, vec = np.linalg.eigh(S)

        # self.weights = vec[:, -1]
        self.weights = vec[:, -1] # eigenvector associated with the largest eigenvalue

        self.compute_projected_means(Y, K)
        # self.training_projections = self.weights.T @ K # if we use training points projections for classification

    def compute_projected_means(self, Y : np.ndarray, K : np.ndarray) -> None:
        nlabels = len(self.labels)
        if self.weights.ndim == 1: # if only one Fisher direction 
            self.projected_mean_classes = np.zeros(nlabels)
        else:
            self.projected_mean_classes = np.zeros((nlabels, self.weights.shape[1]))
        for l in range(nlabels):
            class_mask = Y == self.labels[l]
            self.projected_mean_classes[l] = np.mean(self.weights.T @ K[:, class_mask], axis=-1)

    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        Predict labels for input data points X
        """
        projections = np.transpose(self.weights.T @ self.kernel(self.training_data, X)) # (n_test_samples,)
        

        # choose the class whose the mean projections belonging to this class is the closest to the projected point
        class_ind = np.argmin( np.abs(projections[:, np.newaxis] - self.projected_mean_classes), axis=-1)
        # class_ind = np.argmin( np.linalg.norm(projections[:, :, np.newaxis] - self.projected_mean_classes.T[np.newaxis, :, :], axis=1), axis=-1) # if using multiple Fisher directions
        return self.labels[class_ind]

        # choose the class of the closest projected point : do not work well
        # class_ind = np.argmin( np.abs(projections[:, np.newaxis] - self.training_projections), axis=-1)
        # mlabels = self.labels.max()
        # dist = np.abs(projections[:, np.newaxis] - self.training_projections)
        # dist = np.linalg.norm(projections[:, :, np.newaxis] - self.training_projections, axis=1)
        # ind = np.argpartition(dist, 10, axis=-1)
        # class_ind = ind[:, :11]
        # print('one axis bincount', np.bincount(self.training_labels[class_ind][0], minlength=mlabels+1))
        # out = np.apply_along_axis(np.bincount, axis=-1, arr=self.training_labels[class_ind], minlength=mlabels+1)
        # return np.argmax(out, axis=-1)
