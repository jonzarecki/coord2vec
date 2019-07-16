from typing import Tuple, List

import numpy as np
from torch.utils.data import DataLoader

from coord2vec.config import CACHE_DIR
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset


def extract_embeddings(model, coordinates: List[Tuple[float, float]]) -> np.array:
    # TODO: implement
    pass

def get_pytorch_dataset():
    return TileFeaturesDataset(root_dir=CACHE_DIR)


def get_data_loader():
    return DataLoader(get_pytorch_dataset(), batch_size=4, shuffle=True, num_workers=4)