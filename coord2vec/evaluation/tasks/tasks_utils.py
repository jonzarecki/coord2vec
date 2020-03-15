import datetime
import hashlib
from typing import Tuple, List

import geopandas as gpd
import numpy as np
import pandas as pd
from geoalchemy2 import WKTElement
from shapely.geometry import Polygon

from coord2vec.common.db.connectors import get_connection
from coord2vec.common.db.sqlalchemy_utils import merge_to_table
from coord2vec.common.geographic.geo_utils import sample_grid_in_poly, sample_points_in_poly, \
    poly_outer_intersection
from coord2vec.evaluation.tasks.scores_table import GEOM, MODIFICATION_DATE, SCORES, GEOM_WKT, GEOM_WKT_HASH, \
    EXPERIMENT, DTYPES, MODEL, TRAIN_HASH


def polys2dataset(true_polys: List[Polygon], full_poly: Polygon, step=0.01) -> Tuple[
    List[Tuple[float, float]], List[float]]:
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
    true_points = []
    for true_poly in true_polys:
        true_points.append(sample_grid_in_poly(true_poly, step=step))
    true_points = np.hstack(true_points)

    false_poly = poly_outer_intersection(full_poly, true_polys)

    false_points = sample_points_in_poly(false_poly, num_samples=len(true_points))

    coords = [p.coords[0] for p in true_points + false_points]
    in_poly_label = [1] * len(true_points) + [0] * len(false_points)

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


def coord2rect(lon: float, lat: float, w: float, h: float) -> Polygon:
    """
    Transform a coordinate into a Polygon in a shape of a rectangular
    Args:
        lon: of the coordinate
        lat: of the coordinate
        w: width of the rectangular
        h: height of the rectangular

    Returns:
        Polygon representing the rectangular around the coordinate
    """
    min_lat = float(lat) - float(w) / 2.0
    max_lat = float(lat) + float(w) / 2.0
    min_lon = float(lon) - float(h) / 2.0
    max_lon = float(lon) + float(h) / 2.0
    return Polygon([[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat]])


def save_scores_to_db(scores_df: pd.DataFrame, table_name: str, experiment: str, train_hash: str):
    """
    Save Building probas to a db
    Args:
        scores_df: the scores data, with geometries as indices, and models as columns
        table_name: name of the table to insert the data into
        experiment: name of the experiment resulted in those probas
        train_hash: hash of the train geometries - used for kfolds

    Returns:
        None
    """
    table_name = table_name.lower()
    eng = get_connection(db_name='POSTGRES')

    for model in scores_df.columns:
        insert_df = pd.DataFrame(data={MODIFICATION_DATE: datetime.datetime.now(),
                                       GEOM: scores_df.index,
                                       SCORES: scores_df[model],
                                       MODEL: model,
                                       EXPERIMENT: experiment,
                                       TRAIN_HASH: train_hash}).reset_index(drop=True)

        insert_df[GEOM_WKT] = insert_df[GEOM].apply(lambda g: g.wkt)
        # add hash column for the GEOM_WKT
        # use md5, consistently
        insert_df[GEOM_WKT_HASH] = [str(h) for h in pd.util.hash_pandas_object(insert_df[GEOM_WKT])]

        insert_df[GEOM] = insert_df[GEOM].apply(lambda x: WKTElement(x.wkt, srid=4326))
        merge_to_table(eng, insert_df, table_name, compare_columns=[GEOM_WKT_HASH, EXPERIMENT, MODEL, TRAIN_HASH],
                       update_columns=[MODIFICATION_DATE, SCORES, GEOM, GEOM_WKT], dtypes=DTYPES)
    eng.dispose()


# TODO: add all consistent hash functions to one file, and order them. Consistent hashing is something we do a-lot
def hash_geoseries(no_order_gs: gpd.GeoSeries) -> str:
    """
    Persistent hash for a geoseries with no order
    Args:
        no_order_gs: geoseries

    Returns:
        str of md5 hash
    """
    # encode sorted gs as bytes
    consistent_str = pd.Series(no_order_gs.apply(lambda a: a.wkt).unique()).sort_values()\
        .reset_index(drop=True).to_csv(header=False).encode()
    gs_hash = hashlib.md5(consistent_str)  # build consistent hash with md5
    return gs_hash.hexdigest()
