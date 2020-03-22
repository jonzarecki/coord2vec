import inspect
import os

from sklearn.preprocessing import StandardScaler
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pipeline_adir import *
import pickle
from feature_dataset import Feature_Dataset
from pytorch_lightning.loggers import TensorBoardLogger
import torch.nn.functional as F


def save_to_pickle_features_and_coords():
    coords, features = get_csv_data()
    pickle_out_coords = open("coords.pickle", "wb")
    pickle_out_features = open("features.pickle", "wb")
    pickle.dump(coords, pickle_out_coords)
    pickle.dump(features, pickle_out_features)
    pickle_out_coords.close()
    pickle_out_features.close()


# save_to_pickle_features_and_coords()


def load_from_pickle_features_and_coords():
    pickle_in_coords = open("coords.pickle", "rb")
    coords = pickle.load(pickle_in_coords)
    pickle_in_coords.close()
    pickle_in_features = open("features.pickle", "rb")
    features = pickle.load(pickle_in_features)
    # print(features)
    pickle_in_features.close()
    return coords, features


class Autoencoder(pl.LightningModule):
    def __init__(self, optimizer=torch.optim.Adam, loss_func=F.mse_loss,
                 learning_rate=1e-1, weight_decay=0, num_train=10, num_features=10, batch_size=20,
                 embedding_dim=15):
        super(Autoencoder, self).__init__()
        self.loss_fn = loss_func
        self.num_train = num_train
        self.num_features = num_features
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.init_layers()
        self.optimizer = optimizer(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.prepare_data()

    def init_layers(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.num_features, self.embedding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.num_features)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def prepare_data(self):
        coords, features = load_from_pickle_features_and_coords()  # features contain also x and y
        features['coord'] = coords

        clean_funcs = [clean_floor_col, clean_constructionTime_col]  # can add function if needed
        cleaned_features = generic_clean_col(features, clean_funcs)
        self.coords = cleaned_features['coord']
        self.coords = self.coords[:self.num_train]

        X = cleaned_features.drop(columns=['totalPrice', 'coord']).values.astype(float)
        # todo delete next line to get all the data
        X = X[: self.num_train, :self.num_features]

        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        y = X

        # y = cleaned_features['totalPrice'].values
        self.X_train, self.X_test, self.y_train, self.y_test, self.coords_train, self.coords_test = train_test_split(X,
                                                                                                                     y,
                                                                                                                     self.coords)
        self.X_train, self.X_val, self.y_train, self.y_val, self.coords_train, self.coords_val = train_test_split(
            self.X_train, self.y_train, self.coords_train)
        print(self.coords_train.shape)

    def train_dataloader(self):
        feature_dataset = Feature_Dataset(self.X_train, self.y_train)
        dataloader = DataLoader(feature_dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def val_dataloader(self):
        feature_dataset = Feature_Dataset(self.X_val, self.y_val)
        dataloader = DataLoader(feature_dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def test_dataloader(self):
        feature_dataset = Feature_Dataset(self.X_test, self.y_test)
        dataloader = DataLoader(feature_dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x.float())
        loss = self.loss_fn(output, y.float())
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x.float())
        loss = self.loss_fn(output, y.float())
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def get_data(self, x_or_y, train_val_or_test):
        if x_or_y == 'X':
            if train_val_or_test == 'train':
                return self.X_train
            elif train_val_or_test == 'val':
                return self.X_val
            else:
                return self.X_test
        else:
            if train_val_or_test == 'train':
                return self.y_train
            elif train_val_or_test == 'val':
                return self.y_val
            else:
                return self.y_test
