from sklearn.base import BaseEstimator
import pickle
import numpy as np

from coord2vec import config
from coord2vec.feature_extraction.features_builders import example_features_builder, FeaturesBuilder
from coord2vec.models.data_loading.tile_features_loader import get_files_from_path
from coord2vec.models.data_loading.create_dataset_script import sample_and_save_dataset


class Coord2Features(BaseEstimator):
    """
    Wrapper for the coord2features baseline. Given a coord transforms it to map features
    """

    def __init__(self, feature_builder: FeaturesBuilder):
        self.feature_builder = feature_builder

    def fit(self, cache_dir, sample=False, entropy_threshold=0.1, coord_range=config.israel_range, sample_num=50000):
        if sample:
            sample_and_save_dataset(cache_dir, entropy_threshold=entropy_threshold, coord_range=coord_range,
                                    sample_num=sample_num)

        features = []
        pkl_paths = get_files_from_path(cache_dir)
        for pkl_path in pkl_paths:
            with open(pkl_path, 'rb') as f:
                _, feature_vec = pickle.load(f)
            features.append(feature_vec)

        #################### autoencoder ######################

        #######################################################

    def load_trained_model(self, path):
        return self

    def predict(self, coords):
        feature_vec = self.feature_builder.extract_coordinates(coords)
        return np.asarray(feature_vec)
