from sklearn.base import BaseEstimator
import pickle

from coord2vec import config
from coord2vec.feature_extraction.features_builders import example_features_builder
from coord2vec.models.data_loading.tile_features_loader import get_files_from_path
from coord2vec.models.data_loading.create_dataset_script import sample_and_save_dataset


class Coord2Featrues(BaseEstimator):
    """
    Wrapper for the coord2features baseline. Given a coord transforms it to map features
    """
    def __init__(self):
        pass

    def fit(self, cache_dir, sample=False, entropy_threshold=0.1, coord_range=config.israel_range, sample_num=50000):
        if sample:
            sample_and_save_dataset(cache_dir, entropy_threshold=entropy_threshold, coord_range=coord_range,
                                    sample_num=sample_num)

        features = []
        pkl_paths = get_files_from_path(cache_dir)
        for pkl_path in pkl_paths:
            with open(pkl_path, 'wb') as f:
                _, feature_vec = pickle.load(f)
            features.append(feature_vec)

        #################### autoencoder ######################

        #######################################################


    def predict(self, coords):
        features = []
        for coord in coords:
            feature_vec = example_features_builder.extract_coordinate(coord)
            features.append(feature_vec)

        return features
