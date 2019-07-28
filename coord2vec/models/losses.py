from typing import List

import torch
from torch.nn import Parameter, L1Loss
from torch.nn.modules.loss import _Loss
import numpy as np


class MultiheadLoss(_Loss):
    """
    multihead loss based on "https://arxiv.org/pdf/1705.07115.pdf"
    """

    def __init__(self, losses: List[_Loss], weights: List[float] = None, use_log=False):
        super().__init__()

        self.use_log = use_log
        self.losses = losses
        self.n_heads = len(losses)
        self.weights = weights if weights is not None else np.ones(len(losses))
        self.log_vars = Parameter(torch.zeros(self.n_heads, requires_grad=True, dtype=torch.float32))

        assert (len(losses) == len(self.weights))

    def forward(self, input, target):
        """
        Returns: A tuple of these objects:
            * a Scalar of the weighted loss
            * a list of the seperated the losses' scalars
        """
        assert (len(input) == self.n_heads)
        assert (len(target) == self.n_heads)

        losses = [self.losses[i](input[i], target[i]) for i in range(self.n_heads)]
        log_losses = [torch.log(loss) for loss in losses] if self.use_log else losses
        homosced_losses = [self.weights[i] * torch.exp(-self.log_vars[i]) * log_losses[i] + self.log_vars[i]
                           for i in range(self.n_heads)]
        return sum(homosced_losses), losses
