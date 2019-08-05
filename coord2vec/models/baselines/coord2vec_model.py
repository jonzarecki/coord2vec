import os
from typing import List, Tuple
import torch
from ignite.contrib.handlers import ProgressBar
from sklearn.base import BaseEstimator
from torch import nn
from torch import optim
from torch.nn.modules.loss import _Loss, L1Loss
from torch.utils.data import DataLoader

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from coord2vec.common.mtl.metrics.rmse import RootMeanSquaredError

from coord2vec import config
from coord2vec.config import HALF_TILE_LENGTH, TENSORBOARD_DIR
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.image_extraction.tile_image import generate_static_maps, render_multi_channel
from coord2vec.image_extraction.tile_utils import build_tile_extent
from coord2vec.models.architectures import resnet18, dual_fc_head, multihead_model
from coord2vec.models.baselines.tensorboard_utils import build_example_image_figure, TrainExample, \
    create_summary_writer, add_rmse_to_tensorboard
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset
from coord2vec.models.losses import MultiheadLoss


class Coord2Vec(BaseEstimator):
    """
    Wrapper for the coord2vec algorithm
    """

    def __init__(self, feature_builder: FeaturesBuilder,
                 n_channels: int,
                 losses: List[_Loss] = None,
                 losses_weights=None,
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
        self.feature_names = [feature_builder.features[i].name for i in range(self.n_features)]

        # create L1 losses if not supplied
        self.losses = [L1Loss() for i in range(self.n_features)] if losses is None else losses
        assert len(self.losses) == self.n_features, "Number of losses must be equal to number of features"

        # create the model

        self.model = self._build_model(self.n_channels, self.n_features)
        if multi_gpu:
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def fit(self, train_dataset: TileFeaturesDataset,
            val_dataset: TileFeaturesDataset = None,
            epochs: int = 10,
            batch_size: int = 10,
            num_workers: int = 4):
        """
        Args:
            train_dataset: The dataset object for training data
            val_dataset: The dataset object for validation data, optional
            epochs: number of epochs to train the network
            batch_size: batch size for the network
            num_workers: number of workers for the network

        Returns:
            a trained pytorch model
        """

        # create data loader
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers)
        if val_dataset is not None:
            val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers)
        else:
            val_data_loader = None

        # create the model
        criterion = MultiheadLoss(self.losses, use_log=self.log_loss, weights=self.losses_weights).to(self.device)

        # create tensorboard
        writer = create_summary_writer(self.model, train_data_loader, log_dir=TENSORBOARD_DIR, expr_name=self.tb_dir)

        def multihead_loss_func(y_pred, y):
            return criterion(y_pred[1], torch.split(y, 1, dim=1))[0]

        def multihead_output_transform(x, y, y_pred, *args):
            output = y_pred[1]
            y_pred_tensor = torch.stack(output).squeeze(2)
            y_tensor = y.transpose(0, 1)

            loss, multi_losses = criterion(output, torch.split(y, 1, dim=1))
            return loss, multi_losses, y_pred_tensor, y_tensor

        metrics = {'rmse': RootMeanSquaredError()}
        trainer = create_supervised_trainer(self.model, self.optimizer, multihead_loss_func, device=self.device,
                                            output_transform=multihead_output_transform)
        for name, metric in metrics.items():  # Calculate metrics also on trainer
            metric.attach(trainer, name)

        evaluator = create_supervised_evaluator(self.model,
                                                metrics=metrics,
                                                device=self.device,
                                                output_transform=multihead_output_transform)
        ProgressBar(persist=True, bar_format="").attach(trainer)

        @trainer.on(Events.EPOCH_STARTED)
        def init_state_params(engine):
            engine.state.plusplus_ex, engine.state.plusminus_ex = [None] * self.n_features, [None] * self.n_features
            engine.state.minusminus_ex, engine.state.minusplus_ex = [None] * self.n_features, [None] * self.n_features

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            loss, multi_losses, y_pred_tensor, y_tensor = engine.state.output
            images_batch, features_batch = engine.state.batch
            plusplus_ex, plusminus_ex = engine.state.plusplus_ex, engine.state.plusminus_ex
            minusminus_ex, minusplus_ex = engine.state.minusminus_ex, engine.state.minusplus_ex

            writer.add_scalar('Loss', loss, global_step=engine.state.iteration)

            feat_diff = y_pred_tensor - y_tensor
            feat_sum = y_pred_tensor + y_tensor
            for j in range(self.n_features):
                writer.add_scalar(f'Multiple Losses/{self.feature_names[j]}', multi_losses[j],
                                  global_step=engine.state.iteration)

                for i in range(len(images_batch)):
                    itm_diff, itm_sum = feat_diff[j][i].item(), feat_sum[j][i].item()
                    itm_pred, itm_actual = y_pred_tensor[j][i].item(), y_tensor[j][i].item()
                    ex = TrainExample(images_batch[i], predicted=itm_pred, actual=itm_actual, sum=itm_sum,
                                      diff=itm_diff)
                    if minusminus_ex[j] is None or minusminus_ex[j].sum > itm_sum:
                        engine.state.minusminus_ex[j] = ex
                    elif plusminus_ex[j] is None or plusminus_ex[j].diff < itm_diff:
                        engine.state.plusminus_ex[j] = ex
                    elif minusplus_ex[j] is None or minusplus_ex[j].diff > itm_diff:
                        engine.state.minusplus_ex[j] = ex
                    elif plusplus_ex[j] is None or plusplus_ex[j].sum < itm_sum:
                        engine.state.plusplus_ex[j] = ex

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            global_step = engine.state.iteration
            # evaluator.run(train_data_loader)
            metrics = engine.state.metrics  # already attached to the trainer engine to save
            # can add more metrics here
            add_rmse_to_tensorboard(metrics, writer, self.feature_names, global_step, log_str="train")

            # plot min-max examples
            plusplus_ex, plusminus_ex = engine.state.plusplus_ex, engine.state.plusminus_ex
            minusminus_ex, minusplus_ex = engine.state.minusminus_ex, engine.state.minusplus_ex

            for j in range(self.n_features):
                writer.add_figure(tag=f"{self.feature_names[j]}/plusplus",
                                  figure=build_example_image_figure(plusplus_ex[j]), global_step=global_step)

                writer.add_figure(tag=f"{self.feature_names[j]}/plusminus",
                                  figure=build_example_image_figure(plusminus_ex[j]), global_step=global_step)

                writer.add_figure(tag=f"{self.feature_names[j]}/minusminus",
                                  figure=build_example_image_figure(minusminus_ex[j]), global_step=global_step)

                writer.add_figure(tag=f"{self.feature_names[j]}/minusplus",
                                  figure=build_example_image_figure(minusplus_ex[j]), global_step=global_step)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            global_step = engine.state.iteration
            evaluator.run(val_data_loader)
            metrics = evaluator.state.metrics
            # can add more metrics here
            add_rmse_to_tensorboard(metrics, writer, self.feature_names, global_step, log_str="validation")

        trainer.run(train_data_loader, max_epochs=epochs)

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
