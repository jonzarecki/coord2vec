from abc import ABC, abstractmethod
from functools import partial

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt
from shapely.geometry.base import BaseGeometry

from common.db.postgres import get_df, connect_to_db, connection


class Feature(ABC):
    def __init__(self, apply_type: str, **kwargs):
        if apply_type == 'nearest_neighbour':
            self.apply_func = partial(self.apply_nearest_neighbour, **kwargs)
        elif apply_type == 'number_of':
            self.apply_func = partial(self.apply_number_of, **kwargs)
        else:
            raise AssertionError("apply_type does not match a function")

        self.apply_type = apply_type

    def apply_nearest_neighbour(self, geo: BaseGeometry, conn: connection) -> float:
        q = f"""
        SELECT ST_Distance(t.geom, ST_GeomFromText('{wkt.dumps(geo)}', 4326)) as min_distance
            FROM ({self.get_postgis_connection()}) t
            ORDER BY ST_Distance(t.geom, ST_GeomFromText('{wkt.dumps(geo)}')) ASC
            LIMIT 1;
        """

        df = get_df(q, conn, dispose_conn=True)

        return df['min_distance'].iloc[0]

    def apply_number_of(self, geo: BaseGeometry, conn: connection, max_radius_meter: float) -> int:
        q = f"""
        SELECT count(*) as cnt
            FROM ({self.get_postgis_connection()}) t
            WHERE ST_Distance(t.geom, ST_GeomFromText('{wkt.dumps(geo)}')) <= {max_radius_meter};
        """

        df = get_df(q, conn, dispose_conn=True)

        return df['cnt'].iloc[0]

    @abstractmethod
    def get_postgis_connection(self) -> connection:
        """
        Retrieves the correct connection ojbect in the correct db scheme
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
        with connect_to_db() as conn:
            res = gdf.geometry.apply(lambda x: self.apply_func(geo=x, conn=conn))

        return res
