import hashlib
import time
import time
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point


class Feature(ABC):
    def __init__(self, table_filter_dict: Dict[str, Dict[str, str]], radius: float, feature_names: List[str], **kwargs):
        #  Classes that add apply functions should add them to the dictionary
        self.feature_names = feature_names
        self.radius = None
        self.intersect_tbl_name_dict, self.input_geom_table = None, None
        self.default_value = None
        self.input_gs = None
        self.cache = kwargs.get('cache', None)
        self.table_filter_dict = table_filter_dict
        assert radius is not None, "Radius is now in feature, update your code"
        self.set_radius(radius)

    def extract(self, geom_gs: gpd.GeoSeries = None) -> pd.DataFrame:
        """
        Applies the feature on the gdf, returns the series after the apply
        Args:
            geom_gs: The gdf we want to apply the feature on it's geom column

        Returns:
            The return values as a DataFrame with feature_names as columns
        """
        geom_gs = geom_gs if geom_gs is not None else self.input_gs
        if geom_gs is None:
            raise Exception("Must supply a geo-series, either directly or by factory")
        calculated_gdf = self._calculate_feature(geom_gs)
        calculated_df = calculated_gdf.drop('geom', axis=1, errors='ignore')
        return calculated_df

    def _calculate_feature(self, input_gs: gpd.GeoSeries):
        """
        Calculate a the feature, with the use of a temp table in the db

        Args:
            input_gs: a gs with geometry column of the geometries to query about

        Returns:
            a gs with the same geometry column, and the feature columns
        """
        raise NotImplementedError()

    def extract_single_coord(self, coordinate: Tuple[float, float]) -> pd.DataFrame:
        """
        Applies the feature on the gdf, returns the series after the apply
        Args:
            coordinate: (lat, lon) the coordinate to extract the feature on

        Returns:
            The return value
        """
        return self.extract_coordinates([coordinate])

    def extract_coordinates(self, coords: List[Tuple[float, float]]) -> pd.DataFrame:
        """
        extract the desired features on desired points
        Args:
            coords: list of coordinates

        Returns:
            a pandas dataframe, with columns as features, and rows as the points in gdf
        """
        wkt_points = [Point(coord) for coord in coords]
        gdf = GeoDataFrame(pd.DataFrame({'geom': wkt_points}), geometry='geom')
        return self.extract(gdf)

    def set_radius(self, radius: float) -> None:
        """
        set the radius of the feature
        Args:
            radius: the radius in meters

        Returns:
            None
        """
        self.radius = radius
        self.feature_names = [f"{name}_{radius}m" for name in self.feature_names]
        self.set_default_value(radius)

    def set_intersection_table_names(self, tbl_name_to_intersect_tbl_name: Dict[str, str]) -> None:
        """
        Set the temporary intersection table name for the feature calculation
        Args:
            tbl_name_to_intersect_tbl_name: name of the temporary intersection table

        Returns:
            None
        """
        self.intersect_tbl_name_dict = tbl_name_to_intersect_tbl_name

    def set_input_geom_table(self, table_name: str) -> None:
        """
        Set the temporary input geom table name for the feature calculation
        Args:
            table_name: name of the temporary input geom table

        Returns:
            None
        """
        self.input_geom_table = table_name

    def set_input_gs(self, input_gs: gpd.GeoSeries) -> None:
        """
        Set the input geo series to be calculated
        Args:
            table_name: geo series with all the geometries to calculate features on

        Returns:
            None
        """
        self.input_gs = input_gs

    @abstractmethod
    def set_default_value(self, radius) -> float:
        """
        Set the default value of the feature, you can use the radius for that
        Args:
            radius: the radius of the feature

        Returns:
            The default value to set in the feature
        """
        raise NotImplementedError

    @abstractmethod
    def _build_postgres_query(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return '; '.join(self.feature_names)
