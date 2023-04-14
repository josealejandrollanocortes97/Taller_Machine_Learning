import numpy as np
from .base import UnsupervisedModel

class TSNE(UnsupervisedModel):
    def __init__(self, n_components=2, perplexity=30, learning_rate=200.0, n_iter=1000, **kwargs):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.kwargs = kwargs

    def fit(self, X):
        # Compute pairwise distances
        distances = self._pairwise_distances(X)

        # Compute similarity matrix
        P = self._similarity_matrix(distances)

        # Initialize embedding
        Y = np.random.randn(X.shape[0], self.n_components)

        # Perform gradient descent
        for i in range(self.n_iter):
            # Compute Q-matrix and gradient
            Q, dY = self._gradient(Y, P)

            # Update embedding
            Y = Y - self.learning_rate * dY + self.kwargs.get('momentum', 0.0) * (Y - self.kwargs.get('Y_prev', Y))

            # Compute loss
            if i % 100 == 0:
                loss = np.sum(P * np.log(P / Q))
                print("Iteration %d, loss: %.3f" % (i, loss))

        self.embedding_ = Y
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_

    def transform(self, X):
        return NotImplementedError

    def _pairwise_distances(self, X):
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        distances = X_squared + X_squared.T - 2 * X @ X.T
        distances = np.maximum(distances, 0)
        return distances

    def _similarity_matrix(self, distances):
        P = np.zeros_like(distances)
        for i in range(distances.shape[0]):
            # Compute conditional probabilities using binary search
            p, idx = self._binary_search(distances[i], self.perplexity)
            P[i, idx] = p
            P[i, idx[0]] = np.sum(p) - p.sum()
        P = (P + P.T) / (2 * P.shape[0])
        P = np.maximum(P, 1e-12)
        return P

    def _gradient(self, Y, P):
        # Compute pairwise distances in the embedded space
        distances = self._pairwise_distances(Y)

        # Compute Q-matrix and gradient
        Q = 1 / (1 + distances)
        Q[range(Q.shape[0]), range(Q.shape[0])] = 0
        Q = Q / np.sum(Q)
        PQ = P - Q
        dY = 4 * (PQ @ Y)
        return Q, dY