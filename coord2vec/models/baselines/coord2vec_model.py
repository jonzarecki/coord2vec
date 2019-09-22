import os
import random
from typing import List, Tuple
import torch
from ignite.contrib.handlers import ProgressBar
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
from torch import optim
from torch.nn.modules.loss import _Loss, L1Loss
from torch.utils.data import DataLoader
from ignite.metrics import Metric
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

from coord2vec.common.mtl.metrics import EmbeddingData, DistanceCorrelation, RootMeanSquaredError

from coord2vec import config
from coord2vec.config import HALF_TILE_LENGTH, TENSORBOARD_DIR
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.image_extraction.tile_image import generate_static_maps, render_multi_channel
from coord2vec.image_extraction.tile_utils import build_tile_extent
from coord2vec.models.architectures import resnet18, dual_fc_head, multihead_model, simple_cnn
from coord2vec.models.baselines.tensorboard_utils import TrainExample, \
    create_summary_writer, add_metrics_to_tensorboard, add_embedding_visualization, build_example_image_figure
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset
from coord2vec.models.losses import MultiheadLoss


class Coord2Vec(BaseEstimator, TransformerMixin):
    """
    Wrapper for the coord2vec algorithm
    """

    def __init__(self, feature_builder: FeaturesBuilder,
                 n_channels: int,
                 losses: List[_Loss] = None,
                 losses_weights: List[float] = None,
                 log_loss: bool = False,
                 exponent_heads:bool=False,
                 embedding_dim: int = 128,
                 tb_dir: str = 'default',
                 multi_gpu: bool = True,
                 cuda_device: int = 0,
                 lr: float = 1e-4):
        """

        Args:
            feature_builder: FeatureBuilder to create features with \ features were created with
            n_channels: the number of channels in the input images
            losses: a list of losses to use. must be same length of the number of features
            losses_weights: weights to give the different losses. if None then equals weights of 1
            log_loss: whether to use the log function on the loss before back propagation
            embedding_dim: dimension of the embedding to create
            tb_dir: the directory to use in tensorboard
            multi_gpu: whether to use more than one GPU or not
            cuda_device: if multi_gpu==False, choose the GPU to work on
            lr: learning rate for the Adam optimizer
        """

        self.losses_weights = losses_weights
        self.log_loss = log_loss
        self.exponent_head = exponent_heads
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, train_dataset: TileFeaturesDataset,
            val_dataset: TileFeaturesDataset = None,
            epochs: int = 10,
            batch_size: int = 10,
            num_workers: int = 4,
            evaluate_every: int = 5):
        """
        Args:
            train_dataset: The dataset object for training data
            val_dataset: The dataset object for validation data, optional
            epochs: number of epochs to train the network
            batch_size: batch size for the network
            num_workers: number of workers for the network
            evaluate_every: every how many batches to run evaluation

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
            embedding, output = y_pred
            y_pred_tensor = torch.stack(output).squeeze(2).transpose(0, 1)
            y_tensor = y
            data = x
            # print("\n\nY_pred: ", y_pred[1][0])
            # print("Y_true: ", y[0])
            with torch.no_grad():
                loss, multi_losses = criterion(output, torch.split(y, 1, dim=1))
            # print("LOSS: ", loss)
            # print("Multi LOSS: ", multi_losses[0])
            return data, embedding, loss, multi_losses, y_pred_tensor, y_tensor

        eval_metrics = {'rmse': RootMeanSquaredError(), 'corr': DistanceCorrelation(),
                        'embedding_data': EmbeddingData()}
        train_metrics = {'rmse': RootMeanSquaredError(), 'corr': DistanceCorrelation()}
        trainer = create_supervised_trainer(self.model, self.optimizer, multihead_loss_func, device=self.device,
                                            output_transform=multihead_output_transform)
        for name, metric in train_metrics.items():  # Calculate metrics also on trainer
            metric.attach(trainer, name)

        evaluator = create_supervised_evaluator(self.model,
                                                metrics=eval_metrics,
                                                device=self.device,
                                                output_transform=multihead_output_transform)

        pbar = ProgressBar()
        # RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
        # pbar.attach(trainer, ['loss'])
        pbar.attach(trainer)

        @trainer.on(Events.EPOCH_STARTED)
        def init_state_params(engine):
            engine.state.plusplus_ex, engine.state.plusminus_ex = [None] * self.n_features, [None] * self.n_features
            engine.state.minusminus_ex, engine.state.minusplus_ex = [None] * self.n_features, [None] * self.n_features

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            _, embedding, loss, multi_losses, y_pred_tensor, y_tensor = engine.state.output
            images_batch, features_batch = engine.state.batch
            plusplus_ex, plusminus_ex = engine.state.plusplus_ex, engine.state.plusminus_ex
            minusminus_ex, minusplus_ex = engine.state.minusminus_ex, engine.state.minusplus_ex

            writer.add_scalar('General/Train Loss', loss, global_step=engine.state.iteration)

            feat_diff = (y_pred_tensor - y_tensor)  # / y_tensor + 1
            feat_sum = y_pred_tensor + y_tensor
            for j in range(self.n_features):
                writer.add_scalar(f'Multiple Losses/{self.feature_names[j]}', multi_losses[j],
                                  global_step=engine.state.iteration)
                for i in range(len(images_batch)):
                    itm_diff, itm_sum = feat_diff[i][j].item(), feat_sum[i][j].item()
                    itm_pred, itm_actual = y_pred_tensor[i][j].item(), y_tensor[i][j].item()
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
            metrics = engine.state.metrics  # already attached to the trainer engine to save
            # can add more metrics here
            add_metrics_to_tensorboard(metrics, writer, self.feature_names, global_step, log_str="train")

            # plot min-max examples
            plusplus_ex, plusminus_ex = engine.state.plusplus_ex, engine.state.plusminus_ex
            minusminus_ex, minusplus_ex = engine.state.minusminus_ex, engine.state.minusplus_ex

            for j in range(self.n_features):
                if plusplus_ex[j] is None:
                    continue
                writer.add_figure(tag=f"{self.feature_names[j]}/plusplus",
                                  figure=build_example_image_figure(plusplus_ex[j]), global_step=global_step)

                writer.add_figure(tag=f"{self.feature_names[j]}/plusminus",
                                  figure=build_example_image_figure(plusminus_ex[j]), global_step=global_step)

                writer.add_figure(tag=f"{self.feature_names[j]}/minusminus",
                                  figure=build_example_image_figure(minusminus_ex[j]), global_step=global_step)

                writer.add_figure(tag=f"{self.feature_names[j]}/minusplus",
                                  figure=build_example_image_figure(minusplus_ex[j]), global_step=global_step)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_validation_results(engine):
            global_step = engine.state.iteration
            if global_step % evaluate_every == 0:
                evaluator.run(val_data_loader)
                metrics = evaluator.state.metrics
                # can add more metrics here
                add_metrics_to_tensorboard(metrics, writer, self.feature_names, global_step, log_str="validation")
                add_embedding_visualization(writer, metrics, global_step)

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
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.embedding_dim = checkpoint['embedding_dim']
        self.losses = checkpoint['losses']

        self.model = self.model.to(self.device)
        return self

    def _model_to(self):
        self.model = self.model.to(self.device)
        # # from apex import amp
        # if self.amp:
        #     model, optimizer = amp.initialize(model.to('cuda'), optimizer, opt_level="O1")

    def save_trained_model(self, path: str):
        """
        save a trained model
        Args:
            path: path of the saved torch NN
        """
        self.model = self.model.to('cpu')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'embedding_dim': self.embedding_dim,
            'losses': self.losses,
        }, path)

        # do we want to use it ? from Ignite
        # checkpoint_handler = ModelCheckpoint(args.checkpoint_model_dir, 'checkpoint',
        #                                  save_interval=args.checkpoint_interval,
        #                                  n_saved=10, require_empty=False, create_dir=True)

        self.model = self.model.to(self.device)

    def transform(self, coords: List[Tuple[float, float]]) -> torch.tensor:
        """
        get the embedding of coordinates
        Args:
            coords: a list of tuple like (lat, long) to predict on

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
        embeddings, output = self.model(images)
        return embeddings.to('cpu')

    def _build_model(self, n_channels, n_heads):
        # model = resnet18(n_channels, self.embedding_dim)
        model = simple_cnn(n_channels, self.embedding_dim)
        heads = [dual_fc_head(self.embedding_dim, add_exponent=self.exponent_head) for i in range(n_heads)]
        model = multihead_model(model, heads)
        return model
