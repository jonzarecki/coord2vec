from typing import Tuple, List

from PIL import Image
import geopy
import geopy.distance
import numpy as np
from shapely.geometry import Polygon, Point
from tqdm.auto import tqdm


def is_tile_empty(im: Image) -> bool:
    pass


def build_tile_extent(center: Tuple[float, float], radius_in_meters: float) -> list:
    start = geopy.Point(*center)
    d = geopy.distance.geodesic(kilometers=radius_in_meters / 1000)
    ext_points = list(map(lambda bearing: d.destination(point=start, bearing=bearing), [0, 90, 180, 270]))
    # return [ext_points[0].latitude, ext_points[3].longitude,
    #         ext_points[2].latitude, ext_points[1].longitude]
    return [ext_points[3].longitude,
            ext_points[2].latitude, ext_points[1].longitude, ext_points[0].latitude]
    # return [topleft.latitude, topleft.longitude, bottomright.latitude, bottomright.longitude]

def sample_coordinate_in_range(min_lon: float, min_lat: float, max_lon: float, max_lat: float, seed=None) -> Tuple[
    float, float]:
    if seed is not None:
        np.random.seed(seed)
    lon = np.random.uniform(min_lon, max_lon)
    lat = np.random.uniform(min_lat, max_lat)

    return lon, lat

def sample_coordinate_in_poly(poly: Polygon, seed=None) -> Tuple[float, float]:
    np.random.seed(seed)
    coord = sample_coordinate_in_range(*poly.bounds)
    while not Point(coord).within(poly):
        coord = sample_coordinate_in_range(*poly.bounds)

    return coord

def sample_grid_in_range(min_lon: float, min_lat: float, max_lon: float, max_lat: float, step: float = 0.01) -> np.array:
    x = np.arange(min_lon, max_lon, step=step)
    y = np.arange(min_lat, max_lat, step=step)
    coords = np.stack(np.meshgrid(x, y), -1)
    return coords

def sample_grid_in_poly(poly: Polygon, step: float = 0.01) -> List[Tuple]:
    coords = sample_grid_in_range(*poly.bounds, step=step).reshape(-1, 2)
    return [coord for coord in coords if Point(coord).within(poly)]


def sample_coordinates_in_poly(poly: Polygon, num_samples=1, seed=None) -> List[Tuple]:
    np.random.seed(seed)
    coords = [sample_coordinate_in_poly(poly) for _ in tqdm(range(num_samples))]

    return coords

def poly_outer_intersection(poly: Polygon, delete_polys: List[Polygon]) -> Polygon:
    for delete_poly in delete_polys:
        if poly.intersects(delete_poly) == True:
            # If they intersect, create a new polygon that is
            # essentially pol minus the intersection
            poly = poly.symmetric_difference(delete_poly).difference(delete_poly)
    return poly
