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
        # # create the features
        # cache_dir = os.path.join(TEST_CACHE_DIR, 'seattle_prices')
        #
        # def get_features(i):
        #     feature_vec = self.feature_builder.extract_coordinates([coords[i]])
        #     with open(f"{cache_dir}/{i}.pkl", 'wb') as f:
        #         pickle.dump(feature_vec, f)
        #
        # parmap(get_features, range(len(coords)), use_tqdm=True, desc='building_dataset')
        #
        # # read the features and return them
        # pkl_paths = get_files_from_path(cache_dir)
        # features = []
        # for pkl_path in pkl_paths:
        #     with open(pkl_path, 'rb') as f:
        #         feature_vec = pickle.load(f)
        #     features.append(feature_vec)
        # features_matrix = np.concatenate(features, axis=0)
        # return features_matrix
