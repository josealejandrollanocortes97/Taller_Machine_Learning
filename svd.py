import numpy as np
from .base import UnsupervisedModel

class SVD(UnsupervisedModel):
    def fit(self, X):
        U, s, Vt = np.linalg.svd(X, **self.kwargs)
        self.U = U
        self.s = s
        self.Vt = Vt
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.U @ np.diag(self.s)

    def transform(self, X):
        return X @ self.Vt.T