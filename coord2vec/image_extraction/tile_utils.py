from typing import Tuple, List

from PIL import Image
import geopy
import geopy.distance
import numpy as np


def is_tile_empty(im: Image) -> bool:
    pass


def build_tile_extent(center: Tuple[float, float], radius_in_meters: float) -> list:
    start = geopy.Point(*center)
    d = geopy.distance.geodesic(kilometers=radius_in_meters / 1000)
    ext_points = list(map(lambda bearing: d.destination(point=start, bearing=bearing), [0, 90, 180, 270]))
    return [ext_points[0].latitude, ext_points[3].longitude,
            ext_points[2].latitude, ext_points[1].longitude]
    # return [topleft.latitude, topleft.longitude, bottomright.latitude, bottomright.longitude]

def sample_coordinate_in_range(min_lat: float, min_lon: float, max_lat: float, max_lon: float, seed=None) -> Tuple[
    float, float]:
    np.random.seed(seed)
    lat = np.random.uniform(min_lat, max_lat)
    lon = np.random.uniform(min_lon, max_lon)

    return lat, lon

def sample_grid_in_range(min_lat: float, min_lon: float, max_lat: float, max_lon: float, step: float = 0.01):
    x = np.arange(min_lon, max_lon, step=step)
    y = np.arange(min_lat, max_lat, step=step)
    coords = np.stack(np.meshgrid(x, y), -1)
    return coords
