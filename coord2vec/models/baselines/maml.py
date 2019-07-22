import datetime
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.loss import _Loss, L1Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from coord2vec.config import TENSORBOARD_DIR
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.models.architectures import resnet18, dual_fc_head
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset


class MAML:
    def __init__(self,
                 feature_builder: FeaturesBuilder,
                 n_channels: int,
                 losses: List[_Loss] = None,
                 embedding_dim: int = 128,
                 tb_dir: str = 'default'):
        self.tb_dir = tb_dir
        self.embedding_dim = embedding_dim
        self.n_channels = n_channels
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.feature_builder = feature_builder
        self.n_features = len(feature_builder.features)

        # create L1 losses if not supplied
        self.losses = [L1Loss() for i in range(self.n_features)] if losses is None else losses
        assert len(self.losses) == self.n_features, "Number of losses must be equal to number of features"

        # create the model
        self.common_model = resnet18(n_channels, self.embedding_dim)
        self.head_models = [dual_fc_head(self.embedding_dim) for i in range(self.n_features)]

        self.optimizer = torch.optim.SGD(self.common_model.parameters(), lr=1e-3)

    def fit(self, dataset: TileFeaturesDataset,
            n_epochs: int = 10,
            batch_size: int = 64,
            num_workers: int = 4,
            alpha: float = 1e-5,
            beta: float = 1e-5):
        # create a DataLoader
        data_loaders = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers)

        # create tensorboard
        tb_path = os.path.join(TENSORBOARD_DIR, self.tb_dir) if self.tb_dir == 'test' \
            else os.path.join(TENSORBOARD_DIR, self.tb_dir, str(datetime.datetime.now()))
        writer = SummaryWriter(tb_path)

        params = self.common_model.parameters()
        for epoch in tqdm(range(n_epochs), desc='Epochs', unit='epoch'):
            for image_batch, features_batch in data_loaders:
                meta_state = self.common_model.state_dict()
                grads_list = []
                for task_ind in range(self.n_features):
                    self.common_model.load_state_dict(meta_state)
                    self.optimizer.zero_grad()
                    task_model = nn.Sequential(self.common_model, self.head_models[task_ind])
                    params = task_model.parameters()

                    # forward pass
                    output = task_model(image_batch)
                    loss = self.losses[task_ind](output, features_batch[task_ind:task_ind + 1])

                    # backward pass
                    grads = torch.autograd.grad(loss, task_model.parameters())
                    new_params = [(param - alpha * grads[i]) for i, param in enumerate(params)]

                    # save the  new gradient
                    new_output = task_model.bla()
                    params.append(new_params)

                # preform the meta learning
                self.common_model.load_state_dict(meta_state)

