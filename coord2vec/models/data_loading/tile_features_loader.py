from __future__ import print_function, division

import os
import pickle
from pathlib import Path
from typing import List, Tuple
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from coord2vec.feature_extraction.features_builders import FeaturesBuilder


def get_files_from_path(pathstring) -> List[Tuple[str, str]]:
    """
    Retrives file names from the folder and returns dual file names (in tuple pairs)

    Args:
        pathstring: The folder path

    Returns:
        The all pair file paths
    """

    files_paths = []
    for file in Path(pathstring).glob("**/*_img.npy"):
        fname = os.path.basename(file)
        file_number = fname[:-8]  # minus _img.npy
        features_file = f"{pathstring}/{file_number}_features.npy"
        if os.path.exists(features_file):
            files_paths.append((str(file), features_file))

    return files_paths


class TileFeaturesDataset(Dataset):
    """Tile Features Dataset """

    def __init__(self, root_dir, feature_builder, image_transforms=None, inf2value: float = 1e3):
        """
        Args:
            feature_builder:
            root_dir (string): Directory with all the images.
            image_transforms (callable, optional): Optional transform to be applied
                on an image.
            inf2value : number to replace all the inf's with
        """
        self.files_paths = get_files_from_path(root_dir)
        if image_transforms is None:
            image_transforms = torchvision.transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.image_transforms = image_transforms
        self.feature_builder = feature_builder
        self.inf2value = inf2value

    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, idx):
        img_path, feats_paths = self.files_paths[idx]
        image_arr = np.load(img_path)
        features_arr = np.load(feats_paths)
        features_arr = features_arr[0] if len(features_arr.shape) > 1 else features_arr

        if len(features_arr) > len(self.feature_builder.features_names):  # read more from cache (both norm and not)
            features_arr = features_arr[self.feature_builder.relevant_feature_idxs]

        features_arr[np.isnan(features_arr)] = self.inf2value

        image_torch = self.image_transforms(image_arr.swapaxes(0, 2))  # make it (X, X, 3)
        features_torch = torch.tensor(features_arr).float()

        return image_torch, features_torch


class SingleTileFeaturesDataset(TileFeaturesDataset):
    def __init__(self, root_dir, feature_builder: FeaturesBuilder, feature_index: int = None, image_transforms=None):
        """
        Args:
            feature_builder:
            root_dir (string): Directory with all the images.
            feature_index: the index of the feature to be used
            image_transforms (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, feature_builder, image_transforms)
        self.feature_index = feature_index

    def __getitem__(self, idx):
        image_torch, features_torch = super().__getitem__(idx)
        return image_torch, features_torch[self.feature_index:self.feature_index + 1]
