from typing import Tuple

from PIL import Image

import numpy as np

def is_tile_empty(im: Image) -> bool:
    pass


def build_tile_extent(center: Tuple[float, float], radius_in_meters: float) -> list:
    return [center[0], center[1], center[0] + 0.001, center[1] + 0.001]



def sample_coordinate_in_range(min_lat: float, min_lon: float, max_lat: float, max_lon: float, seed=None) -> Tuple[float, float]:
    np.random.seed(seed)
    lat = np.random.uniform(min_lat, max_lat)
    lon = np.random.uniform(min_lon, max_lon)

    return lat,lon

