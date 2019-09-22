from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt


class Feature(ABC):
    def __init__(self, name: str = 'anonymos_feature', **kwargs):
        #  Classes that add apply functions should add them to the dictionary
        self.name = name

    @abstractmethod
    def extract(self, gdf: GeoDataFrame) -> pd.Series:
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
        # TODO: test
        p = wkt.loads(f'POINT ({coordinate[1]} {coordinate[0]})')
        gdf = GeoDataFrame(pd.DataFrame({'geom': [p]}), geometry='geom')
        return self.extract(gdf).iloc[0]
