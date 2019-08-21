import numpy as np
import pandas as pd

from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset
from coord2vec.feature_extraction.features_builders import house_price_builder

from coord2vec.config import VAL_CACHE_DIR, TRAIN_CACHE_DIR

from coord2vec.models.baselines import Coord2Vec


val_dataset = TileFeaturesDataset(VAL_CACHE_DIR)
train_dataset = TileFeaturesDataset(VAL_CACHE_DIR)

coord2vec = Coord2Vec(house_price_builder, n_channels=3 , tb_dir='test')
coord2vec.fit(train_dataset, val_dataset, epochs=20, batch_size=64)