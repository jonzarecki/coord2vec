from typing import Tuple, List

import numpy as np
from shapely.geometry import Polygon
from coord2vec.image_extraction.tile_utils import sample_grid_in_poly, sample_coordinates_in_poly, \
    poly_outer_intersection


def polys2dataset(true_polys: List[Polygon], full_poly: Polygon, step=0.01) -> Tuple[List, List]:
    """
    Given polygons representing the true areas, generate a dataset of true and false coordinates (in true areas and not).
    Args:
        true_polys: list of polygons representing the true areas
        full_poly: polygon representing the entire area of interest
        step: angle size of steps in the grid sampling

    Returns:
        coords: list of all coordinates, true and false
        in_poly_label: list of labels if in true areas (1) or not (0).
    """
    true_coords = []
    for true_poly in true_polys:
        true_coords.append(sample_grid_in_poly(true_poly, step=step))
    true_coords = np.hstack(true_coords)

    false_poly = poly_outer_intersection(full_poly, true_polys)

    false_coords = sample_coordinates_in_poly(false_poly, num_samples=len(true_coords))

    coords = true_coords + false_coords
    in_poly_label = [1] * len(true_coords) + [0] * len(false_coords)

    return coords, in_poly_label


def coords2dataset(true_coords: List, full_poly: Polygon, step=0.01) -> Tuple[np.array, np.array]:
    """
    Given coordinates representing the trues, generate a dataset of true and false coordinates.
    Args:
        true_coords: list of coordinates representing the trues
        full_poly: polygon representing the entire area of interest
        step: angle size of steps in the grid sampling

    Returns:
        coords: list of all coordinates, true and false
        in_poly_label: list of labels if in true areas (1) or not (0).
    """
    true_polys = [coord2rect(*true_coord, step, step) for true_coord in true_coords]
    return polys2dataset(true_polys, full_poly, step)


def coord2rect(lat: float, lon: float, w: float, h: float) -> Polygon:
    """
    Transform a coordinate into a Polygon in a shape of a rectangular
    Args:
        lat: of the coordinate
        lon: of the coordinate
        w: width of the rectangular
        h: height of the rectangular

    Returns:
        Polygon representing the rectangular around the coordinate
    """
    min_lat = float(lat) - float(w) / 2.0
    max_lat = float(lat) + float(w) / 2.0
    min_lon = float(lon) - float(h) / 2.0
    max_lon = float(lon) + float(h) / 2.0
    return Polygon([[min_lat, min_lon], [min_lat, max_lon], [max_lat, max_lon], [max_lat, min_lon]])
