from argparse import Namespace
from itertools import product

import parmap
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from autoencoder.autoencoder import Autoencoder
from tensorboard_vis import *


def one_run_of_hparams_tuning(lr: float, weight_decay: float, emb_dim: float, batch_size: float):
    hparams_dict = {'lr': lr,
                    'wd': weight_decay,
                    'emb_dim': emb_dim,
                    'bsize': batch_size
                    }
    comment = ", ".join([f"{k}={v}" for k, v in hparams_dict.items()])
    print(comment)
    logger = TensorBoardLogger(save_dir=logs_path, name=comment)
    hparams = Namespace(**hparams_dict)
    model = Autoencoder(hparams=hparams, num_train=num_train, use_all_data=False)
    trainer = Trainer(max_epochs=num_epochs, logger=logger)  # , gpus="0")
    trainer.fit(model)
    add_embedding_visualization_tensor_board(model, logger)
    logger.log_metrics({'hparam/mse_catboost': model.last_mse_catboost})


if __name__ == '__main__':
    num_epochs = 150
    num_train = 10000
    logs_path = '/mnt/adir_logs'
    parameters = {
        'lr': [1e-1, 1e-2],
        'weight_decay': [1e-8, 1e-7],
        'emb_dim': [30, 45],
        'batch_size': [1000],
    }
    parameters_values = list(parameters.values())
    parmap.starmap(one_run_of_hparams_tuning, product(*parameters_values), pm_processes=8)
