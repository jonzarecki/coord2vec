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

num_epochs = 100
num_train = 10
batch_size = 20
learning_rate = 1e-3
num_workers = 4
batch_size_val = 20


def save_to_pickle_features_and_coords():
    coords, features = get_csv_data()
    pickle_out_coords = open("coords.pickle", "wb")
    pickle_out_features = open("features.pickle", "wb")
    pickle.dump(coords, pickle_out_coords)
    pickle.dump(features, pickle_out_features)
    pickle_out_coords.close()
    pickle_out_features.close()


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
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(18, 10),
            nn.ReLU(True),
            nn.Linear(10, 5),
            nn.ReLU(True),
            nn.Linear(5, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(True),
            nn.Linear(5, 10),
            nn.ReLU(True),
            nn.Linear(10, 18),
        )
        self.prepare_data()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def prepare_data(self):
        coords, features = load_from_pickle_features_and_coords()  # features contain also x and y
        clean_funcs = [clean_floor_col, clean_constructionTime_col]  # can add function if needed
        cleaned_features = generic_clean_col(features, clean_funcs)
        # todo delete next line to get all the data
        cleaned_features = cleaned_features[:num_train]
        X = cleaned_features.drop(columns='totalPrice').values.astype(float)
        # print(X.mean(axis=0))
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        # print(X.mean(axis=0))
        y = X
        # y = cleaned_features['totalPrice'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train)

    def train_dataloader(self):
        feature_dataset = Feature_Dataset(self.X_train, self.y_train)
        dataloader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return dataloader

    def val_dataloader(self):
        feature_dataset = Feature_Dataset(self.X_val, self.y_val)
        dataloader = DataLoader(feature_dataset, batch_size=batch_size_val, shuffle=True, num_workers=num_workers)
        return dataloader

    def test_dataloader(self):
        feature_dataset = Feature_Dataset(self.X_test, self.y_test)
        dataloader = DataLoader(feature_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        return dataloader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x.float())
        loss = self.criterion(output, y.float())
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x.float())
        loss = self.criterion(output, y.float())
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}


model = Autoencoder()
trainer = Trainer(max_epochs=num_epochs)
trainer.fit(model)
