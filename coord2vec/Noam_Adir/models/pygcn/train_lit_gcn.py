from argparse import ArgumentParser
import numpy as np
import torch
import random
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from coord2vec.Noam_Adir.models.pygcn.lightning_modules import LitGCN


def main(hparams):
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    if hparams.cuda:
        torch.cuda.manual_seed(hparams.seed)

    # init module
    model = LitGCN(nclass=1, hparams=hparams, use_all_data=True)

    # most basic trainer, uses good defaults
    logs_path = hparams.log_path
    logger = TensorBoardLogger(save_dir=logs_path, name="LitGCN_tb")
    if hparams.gpus != 0:
        trainer = Trainer(max_epochs=hparams.epochs, logger=logger, gpus=hparams.gpus)
    else:
        trainer = Trainer(max_epochs=hparams.epochs, logger=logger)
    trainer.fit(model)
    torch.save(model.state_dict(), hparams.save_path)
    mse_test, rmse_test, mae_test, r2_test = model.evaluate()
    print(f"mse_test: {mse_test}\nrmse_test: {rmse_test}\nr2_test: {r2_test}\nmae_test: {mae_test}")


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    # add hear non model specific parameters
    parser.add_argument('--gpus', type=int, default=0, help='number of gpus to use, 0 means cpu.')
    parser.add_argument('--use-double-precision', action='store_true', default=False, help='use-double-precision.')
    parser.add_argument('--save-path', default='manhattan_litGCN.', help='the save path to the model save dict')
    parser.add_argument('--log-path', default='/data/home/morpheus/coord2vec_noam/coord2vec/Noam_Adir/tb_logs', help='tensorbord log path')

    # good practice to define LightningModule speficic params in the module
    parser = LitGCN.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()
    hparams.cuda = (hparams.gpus != 0) and torch.cuda.is_available()

    main(hparams)