import numpy as np
import pandas as pd
import random
import torch
from torch.backends import cudnn

from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset
from coord2vec.feature_extraction.features_builders import house_price_builder, only_build_area_builder

from coord2vec.config import VAL_CACHE_DIR, TRAIN_CACHE_DIR, get_builder

from coord2vec.models.baselines import Coord2Vec

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = True

train_dataset = TileFeaturesDataset(TRAIN_CACHE_DIR)
val_dataset = TileFeaturesDataset(VAL_CACHE_DIR)

coord2vec = Coord2Vec(get_builder(), n_channels=3, tb_dir='build_multi_50_000_faster')
coord2vec.fit(train_dataset, val_dataset, epochs=200, batch_size=64, evaluate_every=300)
