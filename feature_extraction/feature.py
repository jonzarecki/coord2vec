from abc import ABC, abstractmethod
from pandas import DataFrame
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry.base import BaseGeometry
from shapely import wkt


class Feature(ABC):

    def apply_nearest_neighbour(self, geo: BaseGeometry, eng: object):
        s = f"""
        SELECT t.geom
            FROM ({self.get_postgis_engine()}) t
            ORDER BY ST_Distance(t.geom, ST_GeomFromText('{wkt.dumps(geo)}',4326)) ASC
            LIMIT 1;
        """

    def apply_number_of(self, geo: BaseGeometry, eng: object):
        s = f"""
        SELECT count(*)
            FROM ({self.get_postgis_engine()}) t
            ORDER BY ST_Distance(t.geom, ST_GeomFromText('{wkt.dumps(geo)}',4326)) ASC
            LIMIT 1;
        """

    @abstractmethod
    def get_postgis_engine(self) -> object:
        """
        Retrieves the correct engine in the correct db scheme
        Returns:
            The engine object
        """
        pass

    @abstractmethod
    def _build_postgres_query(self) -> str:
        pass


    def extract(self, gdf: GeoDataFrame) -> pd.Series:
        base_query = self._build_postgres_query()

        return gdf.geometry.apply()
