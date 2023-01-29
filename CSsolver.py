import numpy as np
from spams import omp, lasso

class SpamsOmpSolver(object):
    def __init__(self, sensing_mat, expected_sparsity):
        self._weights = np.sqrt((sensing_mat*sensing_mat).sum(axis=0))
        normalized_mat = sensing_mat / self._weights
        self._sensing_mat = np.asfortranarray(normalized_mat, dtype=np.double)
        self._sparsity = expected_sparsity

    def solve(self, y, sparsity=0):
        """
        :param y: 1D array of measurments
        :return: x: 1D array with D nnz values
        """
        sparsity = self._sparsity if sparsity == 0 else sparsity
        reformated_y = np.asfortranarray(np.transpose(np.array([y])))
        x_sparse = omp(reformated_y, self._sensing_mat, L=sparsity)
        x = x_sparse.toarray()[:, 0] / self._weights
        return x

    def solve_lasso(self, y, lambda1=0.3):
        """
        :param y: 1D array of measurments
        :return: x: 1D array with D nnz values
        """
        reformated_y = np.asfortranarray(np.transpose(np.array([y])))
        x_sparse = lasso(reformated_y, D=self._sensing_mat, lambda1=lambda1)
        x = x_sparse.toarray()[:, 0] / self._weights
        return x

    def solve_many(self, Y):
        """
        :param Y: 2D array of shape MxK Where K is the number of problems to solve
        :return: X: 2D array of size NxK with every column sparse
        """
        reformated_y = np.asfortranarray(Y)
        x_sparse = omp(reformated_y, self._sensing_mat, L=self._sparsity)
        x = x_sparse.toarray()
        x = np.transpose(np.transpose(x) / self._weights)
        return x


