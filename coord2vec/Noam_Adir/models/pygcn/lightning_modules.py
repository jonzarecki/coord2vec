import os
import pickle
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import networkx as nx
import scipy.sparse as sp
import numpy as np
from argparse import Namespace
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader
from geopandas import GeoSeries
from sklearn.model_selection import train_test_split
from shapely.geometry import Point

from coord2vec.Noam_Adir.Adir.autoencoder.data_preparation import Feature_Dataset
from coord2vec.Noam_Adir.models.geom_graph_builder import GeomGraphBuilder
from coord2vec.Noam_Adir.models.pygcn import GraphConvolution, sparse_mx_to_torch_sparse_tensor
from coord2vec.Noam_Adir.pipeline.base_pipeline import extract_and_filter_csv_data, extract_geographical_features, \
    extract_and_filter_manhattan_data, my_z_score_norm, get_data
from coord2vec.Noam_Adir.pipeline.utils import get_non_repeating_coords

from coord2vec.Noam_Adir.models.pyGAT.layers import GraphAttentionLayer, SpGraphAttentionLayer
from coord2vec.Noam_Adir.models.pyGAT.utils import normalize_adj


class LitGCN(pl.LightningModule):
    def __init__(self, X: np.ndarray, y: np.ndarray, adj, train_idx: np.array, nclass: int, hparams: Namespace,
                 loss_fn=F.mse_loss, optimizer=torch.optim.Adam):
        """GCN in lightning."""
        super(LitGCN, self).__init__()
        self.hparams = hparams
        self.loss_fn = loss_fn
        self.task = "class" if nclass > 1 else "reg"
        self.dtype = torch.float64 if self.hparams.use_double_precision else torch.float32

        X, self.X_normalizer = my_z_score_norm(X, return_scalers=True)
        self.adj = self.process_adj(adj)
        self.X = torch.tensor(X, requires_grad=False, dtype=self.dtype)
        if self.task == "reg":
            self.y = torch.tensor(y, requires_grad=False, dtype=self.dtype)  # TODO genetalize to calssification
        else:  # self.task == "class":
            self.y = torch.LongTensor(np.where(y)[1])

        idx_train, idx_val = train_test_split(train_idx)
        self.idx_train = torch.LongTensor(idx_train)
        self.idx_val = torch.LongTensor(idx_val)

        if self.hparams.cuda:
            self.cuda()
            self.X = self.X.cuda()
            self.adj = self.adj.cuda()
            self.y = self.y.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()

        self.dropout = self.hparams.dropout
        nfeat = self.X.shape[1]
        self.epoch_debug_counter = 0

        self.gc1 = GraphConvolution(nfeat, self.hparams.hidden)
        self.gc2 = GraphConvolution(self.hparams.hidden, self.hparams.hidden)
        self.gc3 = GraphConvolution(self.hparams.hidden, self.hparams.hidden)
        self.gc4 = GraphConvolution(self.hparams.hidden, nclass)

        self.optimizer = optimizer(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def forward(self, x, adj):
        self.epoch_debug_counter += 1

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        if self.task == "class":
            return F.log_softmax(x, dim=1)
        else:
            return x


    def train_dataloader(self):
        feature_dataset = Feature_Dataset(self.X, self.y)
        dataloader = DataLoader(feature_dataset, batch_size=len(self.y), shuffle=False)
        return dataloader

    def val_dataloader(self):
        feature_dataset = Feature_Dataset(self.X, self.y)
        dataloader = DataLoader(feature_dataset, batch_size=len(self.y), shuffle=False)
        return dataloader

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        self.train()
        X, y = batch
        y_pred = self.forward(X, self.adj).squeeze()
        loss = self.loss_fn(y_pred[self.idx_train], y[self.idx_train])
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        self.eval()
        X, y = batch
        y_pred = self.forward(X, self.adj).squeeze()
        y_pred_val, y_true_val = y_pred[self.idx_val].detach(), y[self.idx_val].detach()
        loss = F.mse_loss(y_pred_val, y_true_val)
        val_r2 = r2_score(y_pred=y_pred_val, y_true=y_true_val)
        val_mae = mean_absolute_error(y_pred=y_pred_val, y_true=y_true_val)
        val_rmse = np.sqrt(loss)
        tensorboard_logs = {'val_loss': loss,
                            'val_r2': val_r2,
                            'val_mae': val_mae,
                            'val_rmse': val_rmse}
        return tensorboard_logs

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_r2 = np.mean([x['val_r2'] for x in outputs])  # TODO
        avg_mae = np.mean([x['val_mae'] for x in outputs])
        avg_rmse = torch.stack([x['val_rmse'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss,
                            'val_r2': avg_r2,
                            'val_mae': avg_mae,
                            'val_rmse': avg_rmse}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def process_adj(self, adj):
        adj = normalize_adj(adj + self.hparams.self_attention * sp.eye(adj.shape[0]))
        return sparse_mx_to_torch_sparse_tensor(adj)


    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parser = ArgumentParser(parents=[parent_parser])

        # MODEL specific
        # parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
        parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
        parser.add_argument('--self-attention', type=float, default=1, help='control the relative weight of the node represntation')

        # training specific (for this model)
        # parser.add_argument('--epochs', type=int, default=3000, help='Number of epochs to train.')
        parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate.')

        return parser

