from abc import ABC, abstractmethod
from pandas import DataFrame
import pandas as pd
from geopandas import GeoDataFrame


class Feature(ABC):
    @abstractmethod
    def extract(self, gdf: GeoDataFrame) -> pd.Series:
        pass
