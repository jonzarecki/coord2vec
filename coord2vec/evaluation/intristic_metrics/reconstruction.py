import numpy as np
import torch
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
from ignite.metrics import Metric


class DistanceCorrelation(Metric):
    """
        Calculates the root mean squared error for multi-head outputs

        - `update` must receive output of the form `(y_pred, y)`.
        """

    def reset(self):
        self.full_embedding = None
        self.full_features = None

    # TODO: test
    def update(self, output):
        embedding, loss, multi_losses, y_pred_tensor, y_tensor = output
        y_tensor = y_tensor.transpose(0, 1)
        if self.full_embedding is None:
            self.full_embedding = embedding
            self.full_features = y_tensor

        else:
            self.full_embedding = torch.cat((self.full_embedding, embedding))
            self.full_features = torch.cat((self.full_features, y_tensor))

    def compute(self):
        x_distance_matrix = pairwise_distances(self.full_embedding.detach().to('cpu'))
        y_distance_matrix = pairwise_distances(self.full_features.detach().to('cpu'))
        corr_coefficient, p_value = pearsonr(x_distance_matrix.flatten(), y_distance_matrix.flatten())
        return corr_coefficient
