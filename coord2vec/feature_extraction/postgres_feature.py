from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, Callable

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from coord2vec.common.db.postgres import get_df, connect_to_db, connection

# general feature types
NEAREST_NEIGHBOUR_all = 'nearest_neighbour'
NUMBER_OF_all = 'number_of'

# polygon feature types
AREA_OF_poly = 'area_of'

# line feature types
LENGTH_OF_line = 'length_of'


def geo2sql(geo: BaseGeometry, to_geography: bool = False) -> str:
    """
    Transforms $geo to the correct srid geometry sql statement
    Args:
        geo: The geometry we want to transform
        to_geography: whether to transform geo into a geometry or geography object

    Returns:
        The query as a str
    """
    sql = f"ST_GeomFromText('{wkt.dumps(geo)}', 4326)"
    sql = f"geography({sql})" if to_geography else sql
    return sql

#TODO: postgres feature. Does not work for ORS for example
class PostgresFeature(ABC):
    def __init__(self, apply_type: str, name: str = 'anonymos_feature', **kwargs):
        #  Classes that add apply functions should add them to the dictionary
        self.name = name
        self.apply_functions = {
            NEAREST_NEIGHBOUR_all: partial(self.apply_nearest_neighbour, **kwargs),
            NUMBER_OF_all: partial(self.apply_number_of, **kwargs)
        }
        self.apply_type = apply_type

    @staticmethod
    def apply_nearest_neighbour(base_query: str, geo: BaseGeometry, conn: connection, max_radius_meter, **kwargs) -> float:
        q = f"""
        SELECT COALESCE (
           (SELECT ST_DistanceSpheroid(t.geom, {geo2sql(geo)}, 'SPHEROID["WGS 84",6378137,298.257223563]') as dist
            FROM ({PostgresFeature._intersect_circle_query(base_query, geo, max_radius_meter)}) t
            ORDER BY dist ASC
            LIMIT 1), 
        FLOAT '+infinity') as dist;
        """

        df = get_df(q, conn)

        return df['dist'].iloc[0]

    @staticmethod
    def apply_number_of(base_query: str, geo: BaseGeometry, conn: connection, max_radius_meter: float, **kwargs) -> int:
        q = f"""
        SELECT count(*) as cnt
            FROM ({PostgresFeature._intersect_circle_query(base_query, geo, max_radius_meter)}) t
            WHERE ST_DWithin(t.geom, {geo2sql(geo)}, {max_radius_meter}, true);
        """

        df = get_df(q, conn)

        return df['cnt'].iloc[0]

    @staticmethod
    def _intersect_circle_query(base_query: str, geo: BaseGeometry, max_radius_meter: float) -> str:
        """
        Transform a normal base_query into a query after only elements within the radius from geo remain
        Args:
            base_query: the postgres base query to get geo elements
            geo: the geometry object
            max_radius_meter: the radius of the circle to intersect with

        Returns:
            a Postgres query the return only the data on max radius from geo
        """
        query = f"""
                select t.geom 
                from(
                    select ST_Intersection(t.geom, geometry(ST_Buffer({geo2sql(geo, to_geography=True)}, {max_radius_meter}))) as geom
                    from ({base_query}) t) t
                where ST_IsEmpty(geom) = FALSE
                """
        return query

    @abstractmethod
    def get_postgis_connection(self) -> connection:
        """
        Retrieves the correct connection object in the correct db scheme
        Returns:
            The connection object
        """
        pass

    @abstractmethod
    def _build_postgres_query(self) -> str:
        """
        Returns the postgres query filtering the geometries object relevant in the column 'geom'

        For example: filter all buildings and retrieve their geometry in 'geom'
        :return: The filtering query as a string
        """
        pass

    def extract(self, gdf: GeoDataFrame) -> pd.Series:
        """
        Applies the feature on the gdf, returns the series after the apply
        Args:
            gdf: The gdf we want to apply the feature on

        Returns:
            The return values as a Series
        """
        assert self.apply_type in self.apply_functions, "apply_type does not match a function"

        func = self.apply_functions[self.apply_type]
        conn = connect_to_db()
        res = gdf.geometry.apply(lambda x: func(base_query=self._build_postgres_query(), geo=x, conn=conn))
        conn.close()
        return res

    def extract_single_coord(self, coordinate: Tuple[float, float]) -> float:
        """
        Applies the feature on the gdf, returns the series after the apply
        Args:
            coordinate: (lat, lon) the coordinate to extract the feature on

        Returns:
            The return value
        """
        # TODO: test
        assert self.apply_type in self.apply_functions, "apply_type does not match a function"
        p = wkt.loads(f'POINT ({coordinate[1]} {coordinate[0]})')
        gdf = GeoDataFrame(pd.DataFrame({'geom': [p]}), geometry='geom')
        return self.extract(gdf).iloc[0]
