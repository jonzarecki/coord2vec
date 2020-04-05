import logging
from typing import List, Union

import pandas as pd
from geopandas import GeoSeries
from sklearn.base import TransformerMixin
from tqdm import tqdm

from coord2vec.common.itertools import flatten
from coord2vec.common.parallel.multiproc_util import parmap
from coord2vec.feature_extraction.feature import Feature
from coord2vec.feature_extraction.feature_table import GEOM_WKT
from coord2vec.feature_extraction.feature_utils import load_features_using_geoms, save_features_to_db
from coord2vec.feature_extraction.osm.postgres_feature_factory import PostgresFeatureFactory


class FeaturesBuilder(TransformerMixin):
    """
    A data class for choosing the desired features
    """

    def __init__(self, features: List[Union[Feature, List[Feature]]], cache_table: str = None):
        """
        features to be used in this builder
        Args:
            features: a list of features
            cache_table: Optional, if specified will look/save calculated features in the cache
        """
        self.features = flatten(features)

        self.cache_table = cache_table

    @property
    def all_feat_names(self) -> List[str]:
        return flatten([feat.feature_names for feat in self.features])

    def transform_batch(self, input_gs: GeoSeries, use_cache: bool = True) -> pd.DataFrame:
        """
        extract the desired features on desired geometries on a batch
        Args:
            input_gs: a batch of GeoSeries with the desired geometries
            use_cache: if set and self.cache_table is filled will load/save the features to the cache

        Returns:
            a pandas dataframe, with columns as features, and rows as the geometries in input_gs

        """
        assert len(input_gs.apply(lambda p: p.wkt).unique()) == len(
            input_gs), "Shouldn't have duplicates when transform"
        required_feats, loaded_feats_dfs = self.features, []

        if use_cache:
            logging.debug(f"Starting load from cache for {len(input_gs)} objects")
            required_feats, loaded_feats_dfs = self.load_from_cache(self.features, input_gs)

            if len(required_feats) == 0:
                logging.debug("loaded all from cache!")
                return pd.concat(loaded_feats_dfs, axis=1)  # append by column
            else:
                logging.debug(f"loaded from cache {len(loaded_feats_dfs)}/{len(self.features)}")
        else:
            logging.debug(f"Don't load from cache")
        feature_factory = PostgresFeatureFactory(required_feats, input_gs=input_gs)
        with feature_factory:
            features_gs_list = parmap(lambda feature: feature.extract(input_gs), feature_factory.features,
                                      use_tqdm=True, desc=f"Calculating Features for {len(input_gs)} geoms",
                                      unit='feature', leave=False)
            # TODO: if want, extract_object_set

        all_features_df = pd.concat(features_gs_list + loaded_feats_dfs, axis=1)[self.all_feat_names]

        if self.cache_table and use_cache:
            calculated_features_df = pd.concat(features_gs_list, axis=1)
            save_features_to_db(input_gs, calculated_features_df, self.cache_table)

        return all_features_df

    def transform(self, input_gs: GeoSeries, use_cache: bool = True, calc_geo_features_in_batches: bool = True,
                  batch_size=10000) -> pd.DataFrame:
        """
        extract the desired features on desired geometries
        Args:
            input_gs: a GeoSeries with the desired geometries
            use_cache: if set and self.cache_table is filled will load/save the features to the cache
            calc_geo_features_in_batches: should the features be calculated in batches
            batch_size: a batch size for the geographical features

        Returns:
            a pandas dataframe, with columns as features, and rows as the geo-features

        """
        n_samples = len(input_gs)

        if not calc_geo_features_in_batches:
            geo_results = self.transform_batch(input_gs, use_cache)
        else:
            geo_results_list = []
            for batch_start_ind in range(0, n_samples, batch_size):
                batch_end_ind = min(batch_start_ind + batch_size, n_samples)
                input_gs_batch = input_gs[batch_start_ind:batch_end_ind]
                geo_results_list.append(self.transform_batch(input_gs_batch, use_cache))
            geo_results = (pd.concat(geo_results_list)).reset_index(drop=True)
        return geo_results

    def load_from_cache(self, all_features: List[Feature], input_gs) \
            -> (List[Feature], List[pd.DataFrame]):
        """
        extract the desired features on desired geometries
        Args:
            input_gs: a GeoSeries with the desired geometries
            all_features: All the features which we want to read.

        Returns:
            - required_feats: List containing the still relevant features to calculate
            - loaded_feats_dfs: List of DataFrames containing all the loaded features (which completely loaded)

        """
        if self.cache_table is None:
            logging.debug("Cannot use cache without a cache table. calculating all features")
            return all_features, []

        loaded_feats_dfs, required_feats = [], []

        loaded_gdf = load_features_using_geoms(input_gs, self.cache_table, feature_names=self.all_feat_names)
        if len(loaded_gdf) == 0:
            return all_features, []

        have_na_dict = loaded_gdf.isna().any(axis=0).to_dict()
        relevant_columns = [fname for fname in self.all_feat_names if not have_na_dict.get(fname, True)] + [GEOM_WKT]
        wkt_ordered_loaded_gdf = \
            loaded_gdf[relevant_columns].assign(**{GEOM_WKT: loaded_gdf[GEOM_WKT].apply(lambda p: p.wkt)}) \
                .set_index(GEOM_WKT).loc[[p.wkt for p in input_gs]]
        for feat in tqdm(all_features, desc="Checking loaded features", unit="feature", leave=False):
            curr_feat_names = feat.feature_names
            if any((feat_name not in loaded_gdf.columns) for feat_name in curr_feat_names):
                required_feats.append(feat)
                continue

            if any(have_na_dict[feat_name] for feat_name in curr_feat_names):
                required_feats.append(feat)
            else:
                features_df = wkt_ordered_loaded_gdf[curr_feat_names].reset_index(drop=True)
                loaded_feats_dfs.append(features_df)

        return required_feats, loaded_feats_dfs

    def __str__(self):
        return ';'.join([str(f) for f in self.features] + [str(self.cache_table)])
