from typing import Dict, List

import geopandas as gpd

from coord2vec.config import BUILDINGS_FEATURES_TABLE
from coord2vec.feature_extraction.feature import Feature
from coord2vec.feature_extraction.features.osm_features.number_of import NumberOf
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.feature_extraction.osm.postgres_feature_factory import PostgresFeatureFactory
import pandas as pd
import numpy as np


def _polys_to_sparse_objects(polys: gpd.GeoSeries, features: List[Feature]) -> pd.DataFrame:
    """
    find the number of objects inside each polygon, for each feature
    Args:
        polys: geometries to count object inside of them
        features: features that consist objects

    Returns:
        A DataFrame of shape [n_polys, n_features]
    """
    number_of_features = [NumberOf(feature.table_filter_dict[feature.table][feature.object_name],
                                   feature.table,
                                   feature.object_name,
                                   radius=0) for feature in features]
    for i, feature in enumerate(number_of_features):  # prevent same name
        feature.feature_names = [str(i)]
        feature.set_default_value(0)
    feature_builder = FeaturesBuilder(number_of_features, cache_table=BUILDINGS_FEATURES_TABLE)
    object_count = feature_builder.transform(polys)
    return object_count


class SparseFilter:
    def __init__(self, precentile: float = None, min_count: int = None):
        self.min_count = min_count
        self.precentile = precentile

    def __call__(self, feature_column):
        assert isinstance(feature_column, pd.Series)

        count_mask = feature_column >= self.min_count if self.min_count is not None else np.ones_like(feature_column)
        precentile_mask = feature_column >= feature_column.quantile(
            self.precentile) if self.precentile is not None else np.ones_like(feature_column)

        mask = count_mask * precentile_mask
        return mask


def _dense_features_mask(object_count_df: pd.DataFrame, filters: List[SparseFilter]):
    mask_df = pd.DataFrame(index=object_count_df.index, columns=object_count_df.columns, dtype=int)
    for i, feature_object_count in enumerate(object_count_df.columns):
        mask_df[feature_object_count] = filters[i](object_count_df[feature_object_count])
    return mask_df


def build_dense_feature_mask(polys: gpd.GeoSeries, features: List[Feature], filters: List[SparseFilter]) -> pd.DataFrame:
    """
    apply a dense / sparse filter for each of the feautres, for each of the polys
    Args:
        polys: polygons to check on them if the feature is sparse in them or not
        features: list of features to check their sparsity
        filters: SpraseFilter objects that filters only the dense features

    Returns:
        A DataFrame of shape [n_polys, n_features] with binary values. True means dense
    """
    object_count_df = _polys_to_sparse_objects(polys, features)
    is_dense_df = _dense_features_mask(object_count_df, filters)
    return is_dense_df
