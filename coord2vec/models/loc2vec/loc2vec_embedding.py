import os
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from coord2vec.common.parallel.multiproc_util import parmap
from coord2vec.models.loc2vec.geo_tile_dataset import GeoTileDataSet
from coord2vec.models.loc2vec.loc2vec_config import EMB_SIZE, NUM_WORKERS, BATCH_SIZE, MARGIN, N_EPOCHS, LOG_INTERVAL, \
    HP_ZOOM, HP_STEP, HP_WD, HP_LR
from coord2vec.models.loc2vec.networks import Loc2Vec
from coord2vec.models.loc2vec.train_utils import fit
from coord2vec.models.loc2vec.triplet_loss import TripletLoss
from coord2vec.models.loc2vec.utils import HardestNegativeTripletSelector, SemihardNegativeTripletSelector, \
    RandomNegativeTripletSelector


class Loc2vecEmbedding:

    def __init__(self, zoom: int = 17, lr: float = 1e-4, wd: float = 0.1, step: int = 0.1,
                 cuda=torch.cuda.is_available(), skip_model_check=False):
        self.zoom = zoom
        self.lr = lr
        self.wd = wd
        self.step = step
        self.hparams_str = Loc2vecEmbedding.create_str_from_hparams(zoom=zoom, lr=lr, wd=wd, step=step)

        self.feature_names = [f'loc2vec_{self.hparams_str}_{i}' for i in range(EMB_SIZE)]

        self.model_path = os.path.join(PRETRAINED_MODELS_DIR, 'loc2vec', self.hparams_str)
        self.cuda = cuda
        self.model = self.load_model_from_path(self.model_path) if os.path.exists(self.model_path) else None

        if (not skip_model_check) and self.model is None:
            raise Exception(f'The loc2vec model with hparams {self.hparams_str} is not saved. '
                            f'fit should be called before calling transform')

    def fit(self):
        """
        A fit should be used when no saved model exists. this function fit and save the trained model and then
        no further calls are needed.
        """
        self.model = Loc2Vec()
        if self.cuda:
            self.model.cuda()
        # todo choose poly
        train_tiles = GeoTileDataSet(poly=None, zoom=self.zoom, is_inference=False)

        kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': True} if self.cuda else {}
        train_loader = DataLoader(train_tiles, batch_size=BATCH_SIZE, **kwargs)

        loss_fn = TripletLoss(MARGIN,
                              HardestNegativeTripletSelector(MARGIN),
                              SemihardNegativeTripletSelector(MARGIN),
                              RandomNegativeTripletSelector(MARGIN),
                              reg=self.step
                              )
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = lr_scheduler.stepLR(optimizer, step_size=16, gamma=self.wd)

        log_dir = os.path.join(TENSORBOARD_DIR, 'train_nn_loc2vec')
        print(f'logging in {log_dir}\n{self.hparams_str}\n')
        logger = TensorBoardLogger(save_dir=log_dir, name=self.hparams_str)

        fit(train_loader=train_loader,
            model=self.model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            n_epochs=N_EPOCHS,
            cuda=self.cuda,
            log_interval=LOG_INTERVAL,
            filename=self.model_path)

    def calculate_feature(self, input_gs):
        if self.model is None:
            raise Exception(f'The loc2vec model with hparams {self.hparams_str} is not saved. '
                            f'fit should be called before calling transform')
        input_gs = input_gs if isinstance(input_gs, np.ndarray) else input_gs.values
        coords = [(p.centroid.x, p.centroid.y) for p in input_gs]

        tiles_dataset = GeoTileDataSet(coords=coords, zoom=self.zoom, is_inference=True)
        kwargs = {'num_workers': NUM_WORKERS, 'pin_memory': True} if self.cuda else {}
        tiles_loader = DataLoader(tiles_dataset, batch_size=BATCH_SIZE, **kwargs)

        z_lst = []
        for batch_tiles in tqdm(tiles_loader, desc='Embedding tiles', unit='batch'):
            if self.cuda: batch_tiles = batch_tiles.cuda()
            z = self.model(batch_tiles)
            if self.cuda: z = z.cpu()
            z = z.data.numpy()
            z_lst.append(z)
        embeddings = np.vstack(z_lst).astype(float)
        emb_df = pd.DataFrame(embeddings, index=input_gs, columns=self.feature_names)
        return emb_df

    def load_model_from_path(self, model_path):
        if self.cuda:
            model = torch.load(model_path)
            model.cuda()
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
        return model

    @staticmethod
    def create_str_from_hparams(**kwargs):
        hparams_str = ';'.join([f'{k}:{convert_float_to_sci(v)}' for k, v in kwargs.items()])
        return hparams_str

    def __repr__(self):
        is_pretrained = 'Not yet trained' if self.model is None else 'Pretrained'
        rpr = '\n' + is_pretrained + ' Loc2vec' + self.hparams_str
        return rpr


def fit_all_loc2vec(parallel=True, cuda=torch.cuda.is_available()):
    all_hyper_params = list(product(HP_ZOOM, HP_LR, HP_WD, HP_STEP))
    if parallel and not cuda:
        parmap(lambda hparams: Loc2vecEmbedding(*hparams, cuda=cuda).fit(), all_hyper_params,
               use_tqdm=True, desc='Fit Loc2vecEmbedding', unit='hparams choice', nprocs=32)
    else:
        for hparams in tqdm(all_hyper_params, desc='Fit Loc2vecEmbedding', unit='hparams choice'):
            Loc2vecEmbedding(*hparams, cuda=cuda).fit()


if __name__ == '__main__':
    fit_all_loc2vec(parallel=False, cuda=True)
