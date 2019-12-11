import argparse as argparse
import numpy as np
import random
import torch
from torch.backends import cudnn

from coord2vec.common.expr_util import save_all_py_files
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset
from coord2vec.config import VAL_CACHE_DIR, TRAIN_CACHE_DIR, get_builder, EXPR_NAME
from coord2vec.models.baselines import Coord2Vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')  # jupyter notebook compatibility
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--n_feats', type=int, default=9)
    parser.add_argument('--start_lr', type=float, default=1e-4)
    parser.add_argument('--lr_steps', nargs='+', default=[20_000, 40_000, 60_000],
                        help='List of steps in which we apply an LR step')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='gamma to multiply each lr step')

    # parser.add_argument('--hidden_size', type=int, default=32)
    # parser.add_argument('--exp_dir', type=str, default='experiments/2019_11_30_hidden_32')
    parser.add_argument('--nepoch', type=int, default=200)
    parser.add_argument('--manualSeed', type=int, default=42)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--val_interval', type=int, default=500)
    opt = parser.parse_args()
    print(opt)
    print(TRAIN_CACHE_DIR, VAL_CACHE_DIR)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    builder = get_builder()

    train_dataset = TileFeaturesDataset(TRAIN_CACHE_DIR, builder)
    val_dataset = TileFeaturesDataset(VAL_CACHE_DIR, builder)
    save_all_py_files()

    coord2vec = Coord2Vec(get_builder(), n_channels=3, embedding_dim=opt.embedding_size, lr=opt.start_lr,
                          lr_steps=opt.lr_steps, lr_gamma=opt.lr_gamma)
    coord2vec.fit(train_dataset, val_dataset, epochs=opt.nepoch, batch_size=opt.batch_size,
                  evaluate_every=opt.val_interval, save_every=opt.save_interval)


if __name__ == '__main__':
    main()
