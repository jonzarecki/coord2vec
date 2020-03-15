from typing import Tuple, Union

from shapely import wkt
from shapely.geometry import Polygon
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List

from geopandas import GeoSeries
from coord2vec.common.db.connectors import get_connection
from coord2vec.common.db.postgres import get_df, save_geo_series_to_tmp_table
from coord2vec.feature_extraction.osm.osm_tag_filters import OSM_POLYGON_TABLE, BUILDING


def get_buildings_from_polygon(poly: Polygon, is_intersects=True) -> gpd.GeoSeries:
    """
    Get all buildings from inside a polygon
    Args:
        poly: polygon to intersect the buildings in
        is_intersects: if True returns all buildings that intersects poly, else uses within poly

    Returns:
        A Geoseries with all the polygons of the buildings
    """
    eng = get_connection('POSTGRES')  # TODO: doesn't use spatial index
    query = f"""SELECT DISTINCT st_astext(way) as geom
                FROM {OSM_POLYGON_TABLE}
                WHERE {BUILDING}
                AND ST_{"INTERSECTS" if is_intersects else "WITHIN"}(way, ST_GEOMFROMTEXT('{poly.wkt}', 4326))
                """
    df = get_df(query, eng)
    eng.dispose()
    gs = gpd.GeoSeries(wkt.loads(geom) for geom in df['geom'])
    return gs


def extract_buildings_from_polygons(polys: Union[GeoSeries, List[Polygon]], return_source=False) -> \
        Union[
            gpd.GeoSeries, Tuple[gpd.GeoSeries, np.array]]:
    """
    convert polygons to it's buildings
    Args:
        polys: polygons
        return_source: True if want to return the source indices of 'buildings_gs' in 'polys', False otherwise

    Returns:
        buildings_gs: A series of all the buildings
        source_indices: (optional) if return_source==True return np.array in len of buildings_gs mapping to
                        source_index in 'polys'
    """
    # to buildings
    poly_buildings = [get_buildings_from_polygon(poly, is_intersects=True) for poly in polys]
    source_indices = np.array([i for i, buildings in enumerate(poly_buildings) for _ in buildings])

    # distinct
    buildings_wkt = pd.concat(poly_buildings).apply(lambda p: p.wkt)
    unique_buildings_wkt, unique_indices = np.unique(buildings_wkt, return_index=True)
    source_indices = source_indices[unique_indices]

    buildings_gs = gpd.GeoSeries(pd.Series(unique_buildings_wkt, index=range(len(unique_buildings_wkt))).map(wkt.loads))

    if return_source:
        return buildings_gs, source_indices
    else:
        return buildings_gs


def get_k_nearest_buildings(poly: Polygon, k: int, excluded_poly: Polygon = None) -> List[Polygon]:
    """
    Get k nearest buildings to a polygon
    Args:
        poly: polygon to intersect the buildings in
        k: number of buildings to return
        excluded_poly: optional polygon to exclude from the nearest

    Returns:
        A Geoseries with all the polygons of the buildings
    """
    eng = get_connection('POSTGRES')
    if excluded_poly is not None:
        excluded_sql = f"AND ST_INTERSECTS(way, ST_GEOMFROMTEXT('{excluded_poly.wkt}', 4326))=FALSE"
    else:
        excluded_sql = ""
    query = f"""
        SELECT st_astext(way) as geom FROM (
            SELECT way
            FROM {OSM_POLYGON_TABLE}
            WHERE {BUILDING}
                  {excluded_sql}
            ORDER BY way <#> ST_GEOMFROMTEXT('{poly.wkt}', 4326)
            ) t
        LIMIT {int(max(1.1*k, k+10))}
        """
    df = get_df(query, eng)
    eng.dispose()
    gs = [wkt.loads(geom) for geom in df['geom'].unique()][:k]
    return gs


def get_buildings_in_radius(poly: Polygon, radius: float, excluded_poly: Polygon=None) -> List[Polygon]:
    """
    Get all buildings within $radius
    Args:
        poly: polygon to intersect the buildings in
        radius: radius from within we will retrieve all buildings
        excluded_poly: optional polygon to exclude from the nearest

    Returns:
        A Geoseries with all the polygons of the buildings
    """
    eng = get_connection('POSTGRES')
    if excluded_poly is not None:
        excluded_sql = f"AND ST_INTERSECTS(way, ST_GEOMFROMTEXT('{excluded_poly.wkt}', 4326))=FALSE"
    else:
        excluded_sql = ""
    # TODO: doesn't use spatial index
    query = f"""
        SELECT st_astext(way) as geom FROM (
            SELECT way
            FROM {OSM_POLYGON_TABLE}
            WHERE {BUILDING}
                  {excluded_sql}
                  AND ST_DWithin(way, ST_GEOMFROMTEXT('{poly.wkt}', 4326), {radius}, true)
            ) t
        """
    df = get_df(query, eng)
    eng.dispose()
    gs = [wkt.loads(geom) for geom in df['geom'].unique()]
    return gs