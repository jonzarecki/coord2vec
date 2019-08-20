import math
import abc
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


class MtlMetric(Metric):
    """
    Abstract Metric class for MTL metrics
    """

    @abc.abstractmethod
    def update_mtl(self, data, embedding, loss, multi_losses, y_pred_tensor, y_tensor):
        pass

    def update(self, output):
        data, embedding, loss, multi_losses, y_pred_tensor, y_tensor = output
        self.update_mtl(data, embedding, loss, multi_losses, y_pred_tensor, y_tensor)
