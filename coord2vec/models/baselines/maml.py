import datetime
import os
from collections import OrderedDict
from typing import List

import torch
from torch import nn
from torch.nn.modules.loss import _Loss, L1Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from coord2vec.config import TENSORBOARD_DIR
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.models.architectures import dual_fc_head
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset
from coord2vec.models.resnet import wide_resnet50_2


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
        self.common_model = wide_resnet50_2(n_channels, self.embedding_dim)
        self.head_models = [dual_fc_head(self.embedding_dim) for i in range(self.n_features)]

        self.optimizer = torch.optim.SGD(self.common_model.parameters(), lr=1e-3)

    def fit(self, dataset: TileFeaturesDataset,
            n_epochs: int = 10,
            batch_size: int = 64,
            num_workers: int = 4,
            alpha_lr: float = 1e-5,
            beta: float = 1e-5):

        # create a DataLoader
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)
        val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers)

        # create tensorboard
        tb_path = os.path.join(TENSORBOARD_DIR, self.tb_dir) if self.tb_dir == 'test' \
            else os.path.join(TENSORBOARD_DIR, self.tb_dir, str(datetime.datetime.now()))
        writer = SummaryWriter(tb_path)

        for epoch in tqdm(range(n_epochs), desc='Epochs', unit='epoch'):

            # create a new model using the meta model
            task_gradients = []
            for task_ind in range(self.n_features):
                fast_weights = OrderedDict(self.common_model.named_parameters())
                task_model = nn.Sequential(self.common_model, self.head_models[task_ind])
                for image_batch, features_batch in train_data_loader:
                    # forward pass
                    output = task_model(image_batch)
                    loss = self.losses[task_ind](output, features_batch[task_ind:task_ind + 1])

                    # backward pass
                    gradient = torch.autograd.grad(loss, task_model.parameters())

                    # Update weights manually
                    fast_weights = OrderedDict(
                        (name, param - alpha_lr * grad)
                        for ((name, param), grad) in zip(fast_weights.items(), gradient)
                    )

                # accumulate gradients from all the tasks
                for image_batch, features_batch in val_data_loader:
                    output = task_model(image_batch, fast_weights)
                    loss = self.losses[task_ind](output, features_batch[task_ind:task_ind + 1])
                    loss.backward(retain_graph=True)

                    gradients = torch.autograd.grad(loss, fast_weights.values())
                    named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
                    task_gradients.append(named_grads)

                # meta step
                sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                      for k in task_gradients[0].keys()}
                hooks = []
                for name, param in model.named_parameters():
                    hooks.append(
                        param.register_hook(replace_grad(sum_task_gradients, name))
                    )

                model.train()
                optimiser.zero_grad()
                # Dummy pass in order to create `loss` variable
                # Replace dummy gradients with mean task gradients using hooks
                logits = model(torch.zeros((k_way,) + data_shape).to(device, dtype=torch.double))
                loss = loss_fn(logits, create_nshot_task_label(k_way, 1).to(device))
                loss.backward()
                optimiser.step()

                for h in hooks:
                    h.remove()
                # preform the meta learning
                self.common_model.load_state_dict(meta_state)
