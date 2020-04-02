import pandas as pd
from geopandas import GeoSeries

from coord2vec.feature_extraction.features_builders import FeaturesBuilder


class Manhattan_Feature_Builder(FeaturesBuilder):

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
            geo_results = super(Manhattan_Feature_Builder, self).transform(input_gs, use_cache)
        else:
            geo_results_list = []
            for batch_start_ind in range(0, n_samples, batch_size):
                batch_end_ind = min(batch_start_ind + batch_size, n_samples)
                input_gs_batch = input_gs[batch_start_ind:batch_end_ind]
                geo_results_list.append(
                    super(Manhattan_Feature_Builder, self).transform(input_gs_batch, use_cache))
            geo_results = (pd.concat(geo_results_list)).reset_index(drop=True)
        return geo_results
