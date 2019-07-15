from sklearn.base import BaseEstimator
import numpy as np

class Random(BaseEstimator):
    """
    Wrapper for the random embedding baseline
    """
    def __init__(self, d=128):
        self.d = d

    def fit(self, **kwargs):
        pass

    def predict(self, coords):
        embeddings = np.random.rand(len(coords), self.d)
        return embeddings
