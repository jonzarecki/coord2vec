from __future__ import print_function, division

import pickle
from pathlib import Path
from typing import List
import torch
import numpy as np
from torch.utils.data import Dataset


def get_files_from_path(pathstring) -> List[str]:
    """
    Retrives file names from the folder and returns all pickle paths

    Args:
        pathstring: The folder path

    Returns:
        The all pickle paths
    """

    pkl_paths = []
    for file in Path(pathstring).glob("**/*.pkl"):
        pkl_paths.append(str(file))

    return pkl_paths


class TileFeaturesDataset(Dataset):
    """Tile Features Dataset """

    def __init__(self, root_dir, transform=None, inf2value: float = 1e3):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            inf2value : number to replace all the inf's with
        """
        self.pkl_paths = get_files_from_path(root_dir)
        self.transform = transform
        self.inf2value = inf2value

    def __len__(self):
        return len(self.pkl_paths)

    def __getitem__(self, idx):
        with open(self.pkl_paths[idx], 'rb') as f:
            image_arr, features = pickle.load(f)

        for c in features.columns:
            features_c = features[c]
            features.loc[features_c == float('inf'), [c]] = self.inf2value
        sample = {'image': image_arr, 'features': features}

        if self.transform:
            sample = self.transform(sample)

        image_torch = torch.tensor(sample['image']).float()
        features_torch = torch.tensor(sample['features'].values.astype(np.float)[0]).float()

        return image_torch, features_torch


class SingleTileFeaturesDataset(TileFeaturesDataset):
    def __init__(self, root_dir, feature_index: int = None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            feature_index: the index of the feature to be used
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, transform)
        self.feature_index = feature_index

    def __getitem__(self, idx):
        image_torch, features_torch = super().__getitem__(idx)
        return image_torch, features_torch[self.feature_index:self.feature_index + 1]
