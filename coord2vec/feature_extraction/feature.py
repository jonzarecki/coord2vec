from abc import ABC, abstractmethod
from typing import Tuple, List

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt
from shapely.geometry import Point


class Feature(ABC):
    def __init__(self, name: str = 'anonymos_feature', **kwargs):
        #  Classes that add apply functions should add them to the dictionary
        self.name = name

    @abstractmethod
    def extract(self, gdf: GeoDataFrame) -> pd.DataFrame:
        """
        Applies the feature on the gdf, returns the series after the apply
        Args:
            gdf: The gdf we want to apply the feature on

        Returns:
            The return values as a Series
        """
        pass

    def extract_single_coord(self, coordinate: Tuple[float, float]) -> float:
        """
        Applies the feature on the gdf, returns the series after the apply
        Args:
            coordinate: (lat, lon) the coordinate to extract the feature on

        Returns:
            The return value
        """
        return self.extract_coordinates([coordinate]).iloc[0]

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
