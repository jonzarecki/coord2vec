import pickle

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from base_pipeline import *
from feature_dataset import Feature_Dataset
from preprocess import *


def save_to_pickle_features():
    cleaned_csv_features = extract_and_filter_csv_data([clean_floor_col, clean_constructionTime_col])
    all_features = extract_geographical_features(cleaned_csv_features)
    pickle_out_features = open("features.pickle", "wb")
    pickle.dump(all_features, pickle_out_features)
    pickle_out_features.close()


# save_to_pickle_features()


def load_from_pickle_features():
    pickle_in_features = open("features.pickle", "rb")
    features = pickle.load(pickle_in_features)
    # print(features)
    pickle_in_features.close()
    return features


class Autoencoder(pl.LightningModule):
    def __init__(self, optimizer=torch.optim.Adam, loss_func=F.mse_loss,
                 learning_rate=1e-1, weight_decay=0, num_train=10, batch_size=20,
                 embedding_dim=15, use_all_data=False):
        super(Autoencoder, self).__init__()
        self.loss_fn = loss_func
        self.num_train = num_train
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.use_all_data = use_all_data
        self.prepare_data()
        self.num_features = self.X_train.shape[1]
        self.init_layers()
        self.optimizer = optimizer(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.emb_train_list = []

    def init_layers(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.num_features, self.embedding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.num_features)
        )

    def forward(self, x, return_emb=False):
        emb = self.encoder(x)
        output = self.decoder(emb)
        return output, emb if return_emb else output

    def prepare_data(self):
        features = load_from_pickle_features()
        if not self.use_all_data:
            features = features[:self.num_train]
        X = features.drop(columns=["coord", "coord_id", "totalPrice"]).values.astype(float)

        # z-score normalization
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        # targets in auto encoder are the input
        y = X

        # train_val_test split
        self.X_train, self.X_test, self.y_train, self.y_test, self.coords_train, self.coords_test = train_test_split(
            X, y, features['coord'])
        self.X_train, self.X_val, self.y_train, self.y_val, self.coords_train, self.coords_val = train_test_split(
            self.X_train, self.y_train, self.coords_train)

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
        y_hat, embedding_code = self.forward(x.float(), return_emb=True)
        self.emb_train_list.append(embedding_code.cpu())
        loss = self.loss_fn(y_hat, y.float())
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, embedding_code = self.forward(x.float(), return_emb=True)
        loss = self.loss_fn(y_hat, y.float())
        return {'val_loss': loss,
                'val_emb': embedding_code.cpu()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        train_emb = torch.stack(self.emb_train_list)
        self.emb_train_list = []
        val_emb = torch.stack([x['val_emb'] for x in outputs])
        print(f'train_emb = {train_emb.shape}')
        print(f'val_emb = {val_emb.shape}')
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
