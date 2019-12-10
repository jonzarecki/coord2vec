from abc import ABC, abstractmethod
from typing import Tuple, List

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt
from shapely.geometry import Point


class Feature(ABC):
    def __init__(self, feature_names=None, max_radius=50, normed=False, **kwargs):
        #  Classes that add apply functions should add them to the dictionary
        self.feature_names = feature_names
        self.max_radius = max_radius  # in meters
        self.normed = normed

    @abstractmethod
    def extract(self, gdf: GeoDataFrame, only_relevant=False) -> pd.DataFrame:
        """
        Applies the feature on the gdf, returns the series after the apply
        Args:
            gdf: The gdf we want to apply the feature on
            only_relevant: extract only relevant features, (normed or not normed)

        Returns:
            The return values as a Series
        """
        pass

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
