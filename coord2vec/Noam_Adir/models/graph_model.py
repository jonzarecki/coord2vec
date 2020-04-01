from sklearn.base import BaseEstimator
from argparse import ArgumentParser
import numpy as np
import torch
import random
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from coord2vec.Noam_Adir.models.pyGAT.lightning_modules import LitGAT
from coord2vec.Noam_Adir.models.pygcn.lightning_modules import LitGCN


class GraphModel:
    def __init__(self, hparams):
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if hparams.cuda:
            torch.cuda.manual_seed(hparams.seed)

        # most basic trainer, uses good defaults
        logs_path = '/data/home/morpheus/coord2vec_noam/coord2vec/Noam_Adir/tb_logs'
        logger = TensorBoardLogger(save_dir=logs_path, name="LitGat_tb")
        if hparams.gpus != 0:
            self.trainer = Trainer(max_epochs=hparams.epochs, logger=logger, gpus=hparams.gpus)
        else:
            self.trainer = Trainer(max_epochs=hparams.epochs, logger=logger)

    def fit(self, X, y):
        # init module
        pass
        # if hparams.attention:
        #     self.model = LitGAT(nclass=hparams.nclass, hparams=hparams, use_all_data=True, X, y)
        # else:
        #     self.model = LitGCN(nclass=hparams.nclass, hparams=hparams, use_all_data=True, X, y)
        # self.trainer.fit(self.model)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    # add hear non model specific parameters
    parser.add_argument('--gpus', type=int, default=0, help='Number of gpus.')
    parser.add_argument('--use-double-precision', action='store_true', default=False, help='use-double-precision.')
    parser.add_argument('--save-path', default='manhattan_litGat.', help='the save path to the model save dict')
    parser.add_argument('--attention', action='store_true', default=False, help='use GAT layers')
    parser.add_argument('--nclass', type=int, default=1, help='Number of classes if 1 the regression.')

    # good practice to define LightningModule speficic params in the module
    parser = LitGAT.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()
    hparams.cuda = (hparams.gpus!=0) and torch.cuda.is_available()

    # task_handler = ManhattanHousePricing()
    # X, y = task_handler.get_dataset()
    # graph_model = GraphModel(hparams)
    # graph_model.fit(X,y)