from abc import ABC, abstractmethod

class UnsupervisedModel(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def fit_transform(self, X):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def __init__(self, **kwargs):
        self.kwargs = kwargs