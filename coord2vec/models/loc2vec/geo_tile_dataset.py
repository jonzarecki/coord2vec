from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

from coord2vec.image_extraction.tile_utils import sample_coordinates_in_poly
from coord2vec.models.loc2vec.loc2vec_config import SAMPLE_NUM, TILE_SIZE


class GeoTileDataSet(Dataset):
    """
    A custom dataset to provide a btach of geo_tiles.
    """

    def __init__(self, poly: Polygon = None, coords: List = None, zoom: int = 17, is_inference: bool = False,
                 sample_num: int = SAMPLE_NUM):

        assert_message = 'GeoTile Dataset should get a polygon or a list of coords and only one of them'
        assert ((poly is not None) and (coords is None)) or ((poly is None) and (coords is not None)), assert_message

        if poly is not None:
            # todo create sample_coordinates_in_poly
            self.coords = sample_coordinates_in_poly(poly=poly, num_samples=sample_num, seed=42)
        else:  # coords is not None
            self.coords = coords
        self.zoom = zoom
        self.is_inference = is_inference
        self.sample_num = sample_num

        # self.ten_crop = transforms.Compose([transforms.TenCrop(TILE_SIZE)])
        self.five_crop = transforms.Compose([transforms.FiveCrop(TILE_SIZE)])

        self.basic_transform = transforms.Compose([
            transforms.Resize(TILE_SIZE)
            , transforms.ToTensor()
            , transforms.Normalize([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])
        ])

        self.center_transform = transforms.Compose([
            transforms.CenterCrop(TILE_SIZE)
            , self.basic_transform
        ])

        self.anchor_transform = transforms.Compose([
            transforms.RandomAffine(degrees=90, translate=(0.25, 0.25))
            , transforms.RandomHorizontalFlip()
            , transforms.RandomVerticalFlip()
            , self.center_transform
        ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lon, lat = self.coords[index]
        data = load_tile_from_coord(lon, lat, self.zoom, use_cache=True)
        data = data if isinstance(data, Image.Image) else Image.fromarray(data)
        # new
        data = transforms.Compose([transforms.CenterCrop(300)])(data)

        if self.is_inference:
            return self.center_transform(data)
        else:
            # 5 corresponds to five_crop
            center_data_tensor = torch.stack([self.anchor_transform(data) for _ in range(5)], 0)
            cropped_data = self.five_crop(data)
            five_data = torch.stack([self.basic_transform(x) for x in cropped_data], 0)
            ten_data = torch.cat([center_data_tensor, five_data], 0)
            tile_ids = torch.from_numpy(index * np.ones([ten_data.shape[0], 1]))
            tile_ids = tile_ids.type(torch.long)
            return ten_data, tile_ids

    def __len__(self) -> int:
        return self.sample_num if not self.is_inference else len(self.coords)
