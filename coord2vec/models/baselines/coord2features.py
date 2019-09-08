from typing import Tuple, List
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from coord2vec.feature_extraction.features_builders import FeaturesBuilder


class Coord2Features(BaseEstimator, TransformerMixin):
    """
    Wrapper for the coord2features baseline. Given a coord transforms it to map features
    """

    def __init__(self, feature_builder: FeaturesBuilder):
        self.feature_builder = feature_builder

    def fit(self):
        return self

    def transform(self, coords: List[Tuple[float, float]]) -> pd.DataFrame:
        """
        Tranform the coordinates into the geographic features from self.feature builder
        Args:
            coords: list of coordinates tuples like (lat, long)

        Returns:
            a pandas DataFrame with columns as geo-features, and rows for each coordinate
        """
        features = self.feature_builder.extract_coordinates(coords)
        return features
