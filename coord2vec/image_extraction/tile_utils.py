from typing import Tuple, List

from PIL import Image
import geopy
import geopy.distance
import numpy as np


def is_tile_empty(im: Image) -> bool:
    pass


def build_tile_extent(center: List[float], radius_in_meters: float) -> list:
    start = geopy.Point(*center)
    d = geopy.distance.geodesic(kilometers=radius_in_meters / 1000)
    topleft = d.destination(point=start, bearing=135)
    bottomright = d.destination(point=start, bearing=315)
    return [topleft.latitude, topleft.longitude, bottomright.latitude, bottomright.longitude]


def sample_coordinate_in_range(min_lat: float, min_lon: float, max_lat: float, max_lon: float, seed=None) -> Tuple[
    float, float]:
    np.random.seed(seed)
    lat = np.random.uniform(min_lat, max_lat)
    lon = np.random.uniform(min_lon, max_lon)

    return lat, lon
