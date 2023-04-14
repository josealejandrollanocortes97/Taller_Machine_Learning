import numpy as np
from .base import UnsupervisedModel

class PCA(UnsupervisedModel):
    def fit(self, X):
        # Center data
        X = X - np.mean(X, axis=0)

        # Compute covariance matrix
        cov = np.cov(X, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov, **self.kwargs)

        # Sort eigenvectors by eigenvalues in descending order
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]

        # Save components and explained variance
        self.components = eigenvectors
        self.explained_variance = eigenvalues[order] / np.sum(eigenvalues)

        return self

    def fit_transform(self, X):
        self.fit(X)
        return X @ self.components

    def transform(self, X):
        return X @ self.components