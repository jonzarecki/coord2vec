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
        geo_wkt = f"ST_Transform(ST_GeomFromText('{wkt.dumps(geo)}', 4326), 3857)"
        q = f"""
        SELECT ST_Distance_Sphere(t.geom, {geo_wkt}) as dist
            FROM ({self._build_postgres_query()}) t
            ORDER BY dist ASC
            LIMIT 1;
        """

        df = get_df(q, conn, dispose_conn=True)

        return df['dist'].iloc[0]

    def apply_number_of(self, geo: BaseGeometry, conn: connection, max_radius_meter: float) -> int:
        geo_wkt = f"ST_Transform(ST_GeomFromText('{wkt.dumps(geo)}', 4326), 3857)"

        q = f"""
        SELECT count(*) as cnt
            FROM ({self._build_postgres_query()}) t
            WHERE ST_DWithin(t.geom, {geo_wkt}, {max_radius_meter});
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
