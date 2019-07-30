from sklearn.base import BaseEstimator
import numpy as np

class EmptyModel(BaseEstimator):
    def fit(self):
        pass

    def load_trained_model(self, path):
        return self

    def predict(self, coords):
        return np.zeros((len(coords), 1))