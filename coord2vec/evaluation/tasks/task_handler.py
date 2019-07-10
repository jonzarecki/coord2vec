from abc import ABC, abstractmethod

import h2o
from h2o import H2OFrame
from h2o.automl import H2OAutoML


class TaskHandler(ABC):
    """
    Abstract class for evaluation tasks
    """
    def __init__(self):
        h2o.init()
        self.aml = H2OAutoML(max_models=20, seed=1)

    @abstractmethod
    def get_data(self):
        pass

    def fit(self, X, y):
        training_frame = H2OFrame(X)
        training_frame['y'] = H2OFrame(y)

        self.aml.train(y='y', training_frame=training_frame)

    def scores(self):
        lb = self.aml.leaderboard
        return lb.head(rows=lb.nrows)
