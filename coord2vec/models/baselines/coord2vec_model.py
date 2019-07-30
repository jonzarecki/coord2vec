import datetime
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import random
import torch
from sklearn.base import BaseEstimator
from torch import nn
from torch import optim
from torch.nn.modules.loss import _Loss, L1Loss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from coord2vec import config
from coord2vec.config import HALF_TILE_LENGTH, TENSORBOARD_DIR
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.image_extraction.tile_image import generate_static_maps, render_multi_channel
from coord2vec.image_extraction.tile_utils import build_tile_extent
from coord2vec.models.architectures import resnet18, dual_fc_head, multihead_model
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset
from coord2vec.models.losses import MultiheadLoss

IMG_RADIUS_IN_METERS = 50


class Coord2Vec(BaseEstimator):
    """
    Wrapper for the coord2vec algorithm
    """

    def __init__(self, feature_builder: FeaturesBuilder,
                 n_channels: int,
                 losses: List[_Loss] = None,
                 losses_weights = None,
                 log_loss: bool = False,
                 embedding_dim: int = 128,
                 tb_dir: str = 'default',
                 cuda_device: int = 0,
                 multi_gpu: bool = True):
        """

        Args:
            feature_builder: FeatureBuilder to create features with \ features were created with
            n_channels: the number of channels in the input images
            tb_dir: the directory to use in tensorboard
            log_loss: weather to use the log function on the loss before back propagation
            losses: a list of losses to use. must be same length of the number of features
            embedding_dim: dimension of the embedding to create
        """

        self.losses_weights = losses_weights
        self.log_loss = log_loss
        self.tb_dir = tb_dir
        self.embedding_dim = embedding_dim
        self.n_channels = n_channels
        self.multi_gpu = multi_gpu
        if not multi_gpu:
            self.device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

        self.feature_builder = feature_builder
        self.n_features = len(feature_builder.features)

        # create L1 losses if not supplied
        self.losses = [L1Loss() for i in range(self.n_features)] if losses is None else losses
        assert len(self.losses) == self.n_features, "Number of losses must be equal to number of features"

        # create the model

        self.model = self._build_model(self.n_channels, self.n_features)
        if multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def fit(self, dataset: TileFeaturesDataset,
            epochs: int = 10,
            batch_size: int = 10,
            num_workers: int = 4):
        """
        Args:
            cache_dir: directory path of the data
            epochs: number of epochs to train the network
            batch_size: batch size for the network
            num_workers: number of workers for the network

        Returns:
            a trained pytorch model
        """

        # create data loader
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers)

        # create the model
        criterion = MultiheadLoss(self.losses, use_log=self.log_loss, weights=self.losses_weights).to(self.device)

        # create tensorboard
        tb_path = os.path.join(TENSORBOARD_DIR, self.tb_dir) if self.tb_dir == 'test' \
            else os.path.join(TENSORBOARD_DIR, self.tb_dir, str(datetime.datetime.now()))
        writer = SummaryWriter(tb_path)

        # train the model
        global_step = 0
        for epoch in tqdm(range(epochs), desc='Epochs', unit='epoch'):
            self.epoch = epoch
            # Training
            for images_batch, features_batch in data_loader:
                images_batch = images_batch.to(self.device)
                features_batch = features_batch.to(self.device)
                # split the features into the multi_heads:
                split_features_batch = torch.split(features_batch, 1, dim=1)

                self.optimizer.zero_grad()
                output = self.model.forward(images_batch)[1]
                loss, multi_losses = criterion(output, split_features_batch)
                loss.backward()
                self.optimizer.step()

                # tensorboard
                writer.add_scalar('Loss', loss, global_step=global_step)
                for i in range(self.n_features):
                    writer.add_scalar(f'Multiple Losses/{self.feature_builder.features[i].name}', multi_losses[i],
                                      global_step=global_step)
                global_step += 1

            fig = plt.figure(epoch, figsize=(3, 1.5), dpi=500)
            r = random.randint(0, len(images_batch) - 1)

            im = images_batch[r].cpu().numpy().swapaxes(0, 1).swapaxes(1, 2).astype('int')
            batch_output = self.model.forward(images_batch)[1]

            plt.axis("off")
            plt.imshow(im[:, :, 0])

            title_font = {'size': '4', 'color': 'black', 'weight': 'normal',
                          'verticalalignment': 'bottom', 'wrap': True,
                          'ha': 'left'}  # Bottom vertical alignment for more space
            fig.text(0.15, 0.75, f"actual: [{', '.join([str(a) for a in features_batch[r].cpu().numpy()])}]",
                     **title_font)
            fig.text(0.15, 0.8, f"predicted: [{', '.join([str(head_out[r].item()) for head_out in batch_output])}]",
                     **title_font)

            plt.subplot(1, 3, 2)
            plt.axis("off")
            plt.imshow(im[:, :, 1])

            plt.subplot(1, 3, 3)
            plt.axis("off")
            plt.imshow(im[:, :, 2])

            writer.add_figure(tag="Image epoch example", figure=fig, global_step=global_step)

            self.save_trained_model(config.COORD2VEC_DIR_PATH + "/models/saved_models/trained_model.pkl")
        return self.model

    def load_trained_model(self, path: str):
        """
        load a trained model
        Args:
            path: path of the saved torch NN

        Returns:
            the trained model in 'path'
        """
        checkpoint = torch.load(path)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.embedding_dim = checkpoint['embedding_dim']
        self.losses = checkpoint['losses']

        self.model = self.model.to(self.device)
        return self

    def save_trained_model(self, path: str):
        """
        save a trained model
        Args:
            path: path of the saved torch NN
        """
        self.model = self.model.to('cpu')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'embedding_dim': self.embedding_dim,
            'losses': self.losses,
        }, path)

        self.model = self.model.to(self.device)

    def predict(self, coords: List[Tuple[float, float]]):
        """
        get the embedding of coordinates
        Args:
            coords: a list of tuple like (34.123123,32.23423) to predict on

        Returns:
            A tensor of shape [n_coords, embedding_dim]
        """

        # create tiles using the coords
        s = generate_static_maps(config.tile_server_dns_noport, config.tile_server_ports)

        images = []
        for coord in coords:
            ext = build_tile_extent(coord, radius_in_meters=HALF_TILE_LENGTH)
            image = render_multi_channel(s, ext)
            images.append(image)
        images = torch.tensor(images).float().to(self.device)

        # predict the embedding
        embeddings = self.model(images)[0]
        return embeddings.to('cpu')

    def _build_model(self, n_channels, n_heads):
        model = resnet18(n_channels, self.embedding_dim)
        heads = [dual_fc_head(self.embedding_dim) for i in range(n_heads)]
        model = multihead_model(model, heads)
        return model


if __name__ == '__main__':
    losses = [nn.L1Loss() for i in range(12)]
    coord2vec = Coord2Vec(losses=losses, embedding_dim=128)
    coord2vec.fit(f"../../train_cache")
