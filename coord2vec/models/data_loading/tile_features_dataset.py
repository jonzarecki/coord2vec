from __future__ import print_function, division

import pickle
from pathlib import Path
from typing import List

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
        pkl_paths.append(file.as_uri())

    return pkl_paths


class TileFeaturesDataset(Dataset):
    """Tile Features Dataset """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pkl_paths = get_files_from_path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.pkl_paths)

    def __getitem__(self, idx):
        with open(self.pkl_paths[idx]) as f:
            image_arr, features = pickle.load(f)

        sample = {'image': image_arr, 'features': features}

        if self.transform:
            sample = self.transform(sample)

        return sample