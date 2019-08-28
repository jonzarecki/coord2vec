from abc import ABC, abstractmethod
from typing import Callable, Tuple, Any

import h2o
from h2o import H2OFrame
from h2o.automl import H2OAutoML

from coord2vec import config
import pandas as pd

class TaskHandler(ABC):
    """
    Abstract class for evaluation tasks
    """
    def __init__(self):
        h2o.init()#port = config.h20_port)
        self.aml = H2OAutoML(max_models=10, seed=1)

    @abstractmethod
    def get_data(self) -> Tuple[Tuple[float, float], pd.DataFrame, Any]:
        """
        get the data for the task
        Returns:
            (coords, additional_features, y)
            coords: List of tuples like (Lat, Long) if size n_samples
            additional_features: a pandas dataframe of shape [n_samples, n_features]
            y: labels of length n_samples
        """
        pass

    def fit(self, X, y):
        training_frame = H2OFrame(X)
        training_frame['y'] = H2OFrame(y)

        self.aml.train(y='y', training_frame=training_frame)

    def scores(self):
        lb = self.aml.leaderboard
        return lb.head(rows=lb.nrows)
