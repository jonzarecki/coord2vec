import os
import pickle
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import networkx as nx
import scipy.sparse as sp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader
from geopandas import GeoSeries
from sklearn.model_selection import train_test_split
from shapely.geometry import Point

from coord2vec.Noam_Adir.Adir.autoencoder.data_preparation import Feature_Dataset
from coord2vec.Noam_Adir.models.geom_graph_builder import GeomGraphBuilder
from coord2vec.Noam_Adir.pipeline.base_pipeline import extract_and_filter_csv_data, extract_geographical_features, \
    extract_and_filter_manhattan_data
from coord2vec.Noam_Adir.pipeline.utils import get_non_repeating_coords

from coord2vec.Noam_Adir.models.pyGAT.layers import GraphAttentionLayer, SpGraphAttentionLayer
from coord2vec.Noam_Adir.models.pyGAT.utils import normalize_adj


class LitGAT(pl.LightningModule):
    def __init__(self, nclass, hparams, loss_fn=F.mse_loss, use_all_data=False, optimizer=torch.optim.Adam):
        """Dense version of GAT in lightning."""
        super(LitGAT, self).__init__()
        self.hparams = hparams
        self.loss_fn = loss_fn
        self.use_all_data = use_all_data
        self.task = "classification" if nclass > 1 else "regression"

        self.adj, self.X, self.y, self.idx_train, self.idx_val, self.idx_test = self.load_data()
        if self.hparams.cuda:
            self.cuda()
            self.X = self.X.cuda()
            self.adj = self.adj.cuda()
            self.y = self.y.cuda()
            self.idx_train = self.idx_train.cuda()
            self.idx_val = self.idx_val.cuda()
            self.idx_test = self.idx_test.cuda()

        self.dropout = self.hparams.dropout
        self.alpha = self.hparams.alpha
        nfeat = self.X.shape[1]
        self.epoch_debug_counter = 0

        if self.hparams.sparse:
            Layer = SpGraphAttentionLayer
        else:
            Layer = GraphAttentionLayer

        if self.hparams.testFC:
            self.encoder = nn.Sequential(
                nn.Linear(nfeat, self.hparams.hidden),
                nn.ReLU(),
                nn.Linear(self.hparams.hidden, nclass)
            )
        else:
            self.attentions = [Layer(nfeat, self.hparams.hidden, dropout=self.dropout, alpha=self.alpha, concat=True) for _ in
                               range(self.hparams.nb_heads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)

            self.out_att = Layer(self.hparams.hidden * self.hparams.nb_heads, nclass, dropout=self.dropout, alpha=self.alpha, concat=False)

        self.optimizer = optimizer(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def forward(self, x, adj):
        self.epoch_debug_counter += 1
        if self.hparams.testFC:
            return self.encoder(x)
        else:
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.out_att(x, adj)
            if self.task == "classification":
                return F.log_softmax(F.elu(x), dim=1)
            else:
                return x

    def load_data(self):
        dataset = "manhattan"
        if dataset == "manhattan":
            if not self.use_all_data:
                debug_cache_filename = "small_data_manhattan.pickle"
            else:
                debug_cache_filename = "full_data_manhattan.pickle"
            if os.path.isfile(debug_cache_filename):
                features, y_col = pickle.load(open(debug_cache_filename, "rb"))
            else:
                features, y_col = extract_and_filter_manhattan_data(use_full_dataset=self.use_all_data)
                features = extract_geographical_features(features)
                pickle.dump((features, y_col), open(debug_cache_filename, "wb"))
        else:  # dataset == "beijing":
            all_features, y_col = extract_and_filter_csv_data(use_full_dataset=self.use_all_data)
            all_features = extract_geographical_features(all_features)
            features = get_non_repeating_coords(all_features)

        features = features.set_index("coord_id").sort_index()
        X = features.drop(columns=["coord", y_col]).values.astype(float)
        y = features[y_col].values.astype(float)[:, None]

        all_idx = range(len(features))
        idx_train, idx_test = train_test_split(all_idx)
        idx_train, idx_val = train_test_split(idx_train)
        # adj
        all_geometries = GeoSeries([Point(coord[0], coord[1]) for coord in features["coord"]])
        graph_builder = GeomGraphBuilder(geometries=all_geometries, method="DT")
        graph_builder.construct_vertices()
        adj = nx.to_scipy_sparse_matrix(graph_builder.graph)
        # normalize
        self.X_normalizer = StandardScaler()
        self.X_normalizer.fit(X)
        X = self.X_normalizer.transform(X)
        # self.y_normalizer = StandardScaler()
        # self.y_normalizer.fit(y)
        # y = self.y_normalizer.transform(y)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))

        adj = torch.FloatTensor(np.array(adj.todense()))
        X = torch.FloatTensor(np.array(X))
        # if self.task == "regression":
        y = torch.FloatTensor(np.array(y))  # TODO genetalize to calssification
        # if self.task == "classification":
        #     y = torch.LongTensor(np.where(y)[1])

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        return adj, X, y, idx_train, idx_val, idx_test

    def train_dataloader(self):
        feature_dataset = Feature_Dataset(self.X, self.y)
        dataloader = DataLoader(feature_dataset, batch_size=len(self.y), shuffle=False)
        return dataloader

    # def val_dataloader(self):
    #     feature_dataset = Feature_Dataset(self.X, self.y)
    #     dataloader = DataLoader(feature_dataset, batch_size=len(self.y), shuffle=False)
    #     return dataloader

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        self.train()
        X, y = batch
        y_pred = self.forward(X, self.adj)[self.idx_train]
        loss = self.loss_fn(y_pred, y.float()[self.idx_train])
        y_pred_val = self.forward(X, self.adj)[self.idx_val]
        val_loss = self.loss_fn(y_pred_val, y.float()[self.idx_val])
        tensorboard_logs = {'train_loss': loss, 'val_loss': val_loss}
        # tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    # def validation_step(self, batch, batch_idx):
    #     self.eval()
    #     X, y = batch
    #     y_pred = self.forward(X, self.adj)[self.idx_val]
    #     loss = self.loss_fn(y_pred, y.float()[self.idx_val])
    #     return loss
    #
    # def validation_epoch_end(self, outputs: list):
    #     avg_val_loss = outputs[0]
    #     tensorboard_logs = {'val_loss': avg_val_loss}
    #     return {'avg_val_loss': avg_val_loss, 'log': tensorboard_logs}

    def evaluate(self):
        self.eval()
        X, y = self.X, self.y.detach()
        y_pred = self.forward(X, self.adj)[self.idx_test].detach()
        mse_test = mean_squared_error(y_pred=y_pred, y_true=y[self.idx_test])
        r2_test = r2_score(y_true=y[self.idx_test], y_pred=y_pred)
        rmse_test = np.sqrt(mse_test)
        return mse_test, rmse_test, r2_test

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parser = ArgumentParser(parents=[parent_parser])

        # MODEL specific
        parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
        parser.add_argument('--sparse', action='store_true', default=True, help='GAT with sparse version or not.')
        parser.add_argument('--testFC', action='store_true', default=False, help='test pipeline with simple 2 layer FC')
        parser.add_argument('--seed', type=int, default=70, help='Random seed.')
        parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
        parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
        parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
        parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

        # training specific (for this model)
        parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
        parser.add_argument('--patience', type=int, default=100, help='Patience')

        return parser

