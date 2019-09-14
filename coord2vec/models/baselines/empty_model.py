from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class EmptyModel(BaseEstimator, TransformerMixin):
    """
    Just an empty model to be used as a baseline
    """
    def fit(self):
        return self

    def transform(self, coords):
        return np.zeros((len(coords), 1))