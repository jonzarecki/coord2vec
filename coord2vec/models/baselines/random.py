from sklearn.base import BaseEstimator
import numpy as np

class RandomGeo(BaseEstimator):
    """
    Wrapper for the random embedding baseline
    """
    def __init__(self, dim=9, **kwargs):
        self.dim = dim

    def fit(self, **kwargs):
        pass

    def transform(self, coords):
        embeddings = np.random.rand(len(coords), self.dim)
        return embeddings
