from typing import List, Tuple

import torch
from sklearn.base import BaseEstimator
from torch import optim
from torch.nn.modules.loss import _Loss, L1Loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from coord2vec import config
from coord2vec.image_extraction.tile_image import generate_static_maps, render_multi_channel
from coord2vec.image_extraction.tile_utils import build_tile_extent
from coord2vec.models.architectures import resnet18, dual_fc_head, multihead_model
from coord2vec.models.data_loading.create_dataset_script import save_sampled_dataset
from coord2vec.models.losses import multihead_loss
from coord2vec.models.model_utils import get_data_loader, get_pytorch_dataset


class Coord2Vec(BaseEstimator):
    """
    Wrapper for the coord2vec algorithm
    """

    def __init__(self, losses: List[_Loss] = None, embedding_dim: int = 128):
        """

        Args:
            losses: a list of losses to use. must be same length of the number of features
            embedding_dim: dimension of the embedding to create
        """

        self.embedding_dim = embedding_dim
        self.losses = losses

    def fit(self, cache_dir: str,
            epochs: int = 10,
            batch_size: int = 10,
            num_workers: int = 4,

            sample: bool = False,
            coord_range: List[float, float, float, float] = config.israel_range,
            entropy_threshold: float = 0.1,
            sample_num: int = 50000):
        """
        Args:
            cache_dir: directory path of the data
            epochs: number of epochs to train the network
            batch_size: batch size for the network
            num_workers: number of workers for the network

            # If you want to create data from scratch use these parameters:
            sample: assign True if you want to create new data
            coord_range: coordinates bounding box where to sample coords from
            entropy_threshold: entropy threshold to filter images by
            sample_num: number of coordinates to sample

        Returns:
            a trained pytorch model
        """

        # create data loader
        if sample:
            save_sampled_dataset(cache_dir, entropy_threshold=entropy_threshold, coord_range=coord_range,
                                 sample_num=sample_num)
        data_loader = DataLoader(get_pytorch_dataset(cache_dir), batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers)

        # extract some parameters from the data loader
        n_channels = data_loader.dataset[0][0].shape[0]
        n_features = data_loader.dataset[0][1].shape[0]

        # create losses of it was None
        self.losses = [L1Loss for i in range(n_features)] if self.losses is None else self.losses
        assert len(self.losses) == n_features, "Number of losses must be equal to number of features"

        # create the model
        model = self._build_model(n_channels, n_features)
        criterion = multihead_loss(self.losses)
        optimizer = optim.Adam(model.parameters())

        # train the model
        for epoch in tqdm(range(epochs), desc='Epochs', unit='epoch'):
            # Training
            for images_batch, features_batch in data_loader:
                # split the features into the multi_heads:
                split_features_batch = torch.split(features_batch, 1, dim=1)

                optimizer.zero_grad()
                print(len(split_features_batch))
                output = model.forward(images_batch)[1]
                loss = criterion(output, split_features_batch)
                loss.backward()
                optimizer.step()
        self.model = model
        return self.model

    def load_trained_model(self):
        #################### brus ######################

        ################################################
        pass

    def predict(self, coords: List[Tuple[float, float]]):
        """
        get the embedding of coordinates
        Args:
            coords: a list of tuple like (34.123123,32.23423) to predict on

        Returns:
            A tensor of shape [n_coords, embedding_dim]
        """

        # create tiles using the coords
        s = generate_static_maps(config.tile_server_dns_noport, [8080, 8081])

        images = []
        for coord in coords:
            ext = build_tile_extent(coord, radius_in_meters=50)
            image = render_multi_channel(s, ext)
            images.append(image)
        images = torch.tensor(images).float()

        # predict the embedding
        embeddings = self.model(images)[0]
        return embeddings

    def _build_model(self, n_channels, n_heads):
        model = resnet18(n_channels, self.embedding_dim)
        heads = [dual_fc_head(self.embedding_dim) for i in range(n_heads)]
        model = multihead_model(model, heads)
        return model
