from sklearn.base import BaseEstimator
import numpy as np

class EmptyModel(BaseEstimator):
    """
    Just an empty model to be used as a baseline
    """
    def fit(self):
        self.feature_builder = ['empty_feature']

    def load_trained_model(self, path):
        return self

    def predict(self, coords):
        return np.zeros((len(coords), 1))