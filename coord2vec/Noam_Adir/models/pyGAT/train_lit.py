from argparse import ArgumentParser
import numpy as np
import torch
import random
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from coord2vec.Noam_Adir.models.pyGAT.lightning_modules import LitGAT


def main(hparams):
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    if hparams.cuda:
        torch.cuda.manual_seed(hparams.seed)

    # init module
    model = LitGAT(nclass=1, hparams=hparams, use_all_data=True)

    # most basic trainer, uses good defaults
    logs_path = '/data/home/morpheus/coord2vec_noam/coord2vec/Noam_Adir/tb_logs'
    logger = TensorBoardLogger(save_dir=logs_path, name="LitGat_tb")
    if hparams.gpus != 0:
        trainer = Trainer(max_epochs=hparams.epochs, logger=logger, gpus=hparams.gpus)
    else:
        trainer = Trainer(max_epochs=hparams.epochs, logger=logger)
    trainer.fit(model)
    torch.save(model.state_dict(), hparams.save_path)
    test_loss, mse_test, rmse_test, r2_test, r2_test_transform = model.evaluate()
    print(f"test_loss: {test_loss}\nmse_test: {mse_test}\nrmse_test: {rmse_test}\nr2_test: {r2_test}\nr2_test_transform: {r2_test_transform}")


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    # add hear non model specific parameters
    parser.add_argument('--gpus', type=int, default=0, help='Number of gpus.')
    parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--save-path', default='manhattan_litGat.', help='the save path to the model save dict')

    # good practice to define LightningModule speficic params in the module
    parser = LitGAT.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()
    hparams.cuda = not hparams.no_cuda and torch.cuda.is_available()

    main(hparams)
