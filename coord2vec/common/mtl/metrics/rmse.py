import torch
from ignite.exceptions import NotComputableError

from coord2vec.common.mtl.metrics.mtl_metric import MtlMetric


class RootMeanSquaredError(MtlMetric):
    """
    Calculates the root mean squared error for multi-head outputs

    - `update` must receive output of the form `(y_pred, y)`.
    """
    def reset(self):
        self._sum_of_squared_errors = None
        self._num_examples = 0

    # no test
    def update_mtl(self, data, embedding, loss, multi_losses, y_pred_tensor, y_tensor):
        squared_errors = torch.pow(y_pred_tensor - y_tensor.view_as(y_pred_tensor), 2)
        if self._sum_of_squared_errors is None:
            self._sum_of_squared_errors = torch.sum(squared_errors, 0)
        else:
            self._sum_of_squared_errors += torch.sum(squared_errors, 0)
        self._num_examples += y_tensor.shape[1]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('RootMeanSquaredError must have at least one example before it can be computed.')
        mse = self._sum_of_squared_errors / self._num_examples

        return torch.sqrt(mse).cpu().numpy()
