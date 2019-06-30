from abc import ABC, abstractmethod
from functools import partial

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt
from shapely.geometry.base import BaseGeometry

from common.db.postgres import get_df, connect_to_db, connection


def geo2sql(geo: BaseGeometry) -> str:
    """
    Transforms $geo to the correct srid geometry sql statement
    Args:
        geo: The geometry we want to transform

    Returns:
        The query as a str
    """
    return f"ST_Transform(ST_GeomFromText('{wkt.dumps(geo)}', 4326), 3857)"


class Feature(ABC):
    def __init__(self, apply_type: str, **kwargs):
        #  Classes that add apply functions should add them to the dictionary
        self.apply_functions = {
            'nearest_neighbour': partial(self.apply_nearest_neighbour, **kwargs),
            'number_of': partial(self.apply_number_of, **kwargs)
        }
        self.apply_type = apply_type


    @staticmethod
    def apply_nearest_neighbour(base_query: str, geo: BaseGeometry, conn: connection, **kwargs) -> float:
        q = f"""
        SELECT ST_Distance(t.geom, {geo2sql(geo)}) as dist
            FROM ({base_query}) t
            ORDER BY dist ASC
            LIMIT 1;
        """

        df = get_df(q, conn)

        return df['dist'].iloc[0]

    @staticmethod
    def apply_number_of(base_query: str, geo: BaseGeometry, conn: connection, max_radius_meter: float, **kwargs) -> int:
        q = f"""
        SELECT count(*) as cnt
            FROM ({base_query}) t
            WHERE ST_DWithin(t.geom, {geo2sql(geo)}, {max_radius_meter});
        """

        df = get_df(q, conn)

        return df['cnt'].iloc[0]

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
        assert self.apply_type in self.apply_functions, "apply_type does not match a function"

        func = self.apply_functions[self.apply_type]
        conn = connect_to_db()
        res = gdf.geometry.apply(lambda x: func(base_query=self._build_postgres_query(), geo=x, conn=conn))
        conn.close()
        return res
