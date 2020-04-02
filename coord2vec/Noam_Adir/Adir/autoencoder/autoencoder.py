import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
import sys

# sys.path.append('/data/home/morpheus/coord2vec_Adir/coord2vec/Noam_Adir/Adir')
from coord2vec.Noam_Adir.Adir.autoencoder.data_preparation import *


class Autoencoder(pl.LightningModule):
    def __init__(self, hparams, num_train, use_all_data=False, optimizer=torch.optim.Adam, loss_fn=F.mse_loss):
        super(Autoencoder, self).__init__()
        self.hparams = hparams
        self.loss_fn = loss_fn
        self.num_train = num_train
        self.use_all_data = use_all_data
        self.prepare_data()
        self.num_features = self.X_train.shape[1]
        self.init_layers()
        self.optimizer = optimizer(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)
        self.emb_train_list = []

    def init_layers(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.num_features, self.hparams.emb_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.emb_dim, self.num_features)
        )

    def forward(self, x, return_emb=False):
        emb = self.encoder(x)
        output = self.decoder(emb)
        return output, emb if return_emb else output

    def prepare_data(self):
        features = load_from_pickle_features("beijing_features.pickle")
        if not self.use_all_data:
            features = features[:self.num_train]
        X = features.drop(columns=["coord", "coord_id", "totalPrice"]).values.astype(float)
        total_price = features['totalPrice'].values[:, None]

        # z-score normalization
        X, total_price = self.z_score_norm_on_X_and_total_price(X, total_price)

        # targets in auto encoder are the input
        y = X

        # train_val_test split
        self.split_train_val_test(X, y, features['coord'], total_price)

    def train_dataloader(self):
        feature_dataset = Feature_Dataset(self.X_train, self.y_train)
        dataloader = DataLoader(feature_dataset, batch_size=self.hparams.bsize, shuffle=True)
        return dataloader

    def val_dataloader(self):
        feature_dataset = Feature_Dataset(self.X_val, self.y_val)
        dataloader = DataLoader(feature_dataset, batch_size=self.hparams.bsize, shuffle=True)
        return dataloader

    def test_dataloader(self):
        feature_dataset = Feature_Dataset(self.X_test, self.y_test)
        dataloader = DataLoader(feature_dataset, batch_size=self.hparams.bsize, shuffle=True)
        return dataloader

    def configure_optimizers(self):
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(self.on_gpu)
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
        if len(self.emb_train_list) > 0:
            train_emb = torch.cat(self.emb_train_list)
            self.emb_train_list = []
            val_emb = torch.cat([x['val_emb'] for x in outputs])
            mse_catboost = train_models_from_splitted_data(
                [CatBoostRegressor(verbose=False)],
                train_emb.data.numpy(), self.tot_price_train,
                val_emb.data.numpy(), self.tot_price_val
            )[1][0]  # train_models return tuple of lists
        else:
            mse_catboost = 0
        self.last_mse_catboost = mse_catboost
        tensorboard_logs = {'val_loss': avg_loss, 'mse_catboost': mse_catboost}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def z_score_norm_on_X_and_total_price(self, X, totalPrice):
        """
        make a z-score normalization on the X features by column and on the targets totalPrice
        Args:
            X: features
            totalPrice: target

        Returns: X_norm, totalPrice_norm

        """
        self.X_normalizer = StandardScaler()
        self.X_normalizer.fit(X)
        X_norm = self.X_normalizer.transform(X)
        self.tot_price_normalizer = StandardScaler()
        self.tot_price_normalizer.fit(totalPrice)
        total_price_norm = self.tot_price_normalizer.transform(totalPrice)
        return X_norm, total_price_norm

    def split_train_val_test(self, X, y, coords, total_price):
        """
        split every arg to train, validation and test sets
        Args:
            X: features
            y: same as X
            coords: coordinates feature
            total_price: target feature

        """
        self.X_train, self.X_test, self.y_train, self.y_test, \
        self.coords_train, self.coords_test, self.tot_price_train, self.tot_price_test = train_test_split(
            X, y, coords, total_price)
        self.X_train, self.X_val, self.y_train, self.y_val, \
        self.coords_train, self.coords_val, self.tot_price_train, self.tot_price_val = train_test_split(
            self.X_train, self.y_train, self.coords_train, self.tot_price_train)
