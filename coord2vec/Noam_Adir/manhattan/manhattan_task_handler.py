from typing import Tuple

import numpy as np
import pandas as pd
from geopandas import GeoSeries

from coord2vec.Noam_Adir.manhattan.preprocess import ALL_MANHATTAN_FILTER_FUNCS_LIST
from coord2vec.Noam_Adir.utils import load_from_pickle_features, generic_clean_col
from coord2vec.evaluation.tasks.task_handler import TaskHandler
from files_path import *


class Manhattan_Task_Handler(TaskHandler):


    def get_dataset(self, all_dataset: bool) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        """
        get the non-geo data for the task from manhattan pickle
        Returns:
            coords, features_without_geo, sold
        """
        df = load_from_pickle_features(MANHATTAN_PKL_PATH)
        df = generic_clean_col(df, ALL_MANHATTAN_FILTER_FUNCS_LIST)
        df = df if all_dataset else df[:5]
        coords = df.apply(lambda row: tuple(row[['lon', 'lat']].values), axis=1).values  # np.ndarray, dtype=object
        features_without_geo = df[['numBedrooms', 'numBathrooms', 'sqft']].astype(float)  # pd.DataFrame, dtype=float
        y = df['sold'].values.astype(float)  # np.ndarray, dtype=float

        return coords, features_without_geo, y

    def transform(self, input_qs: GeoSeries, calc_geo_features_in_batches: bool = True,
                  batch_size=10000) -> pd.DataFrame:
        """
        Transform the coordinates into the geographic features with self.embedder
        Args:
            input_qs: geo-series with the desired geometries
            calc_geo_features_in_batches: should the features be calculated in batches
            batch_size: a batch size for the geographical features

        Returns:
            a pandas DataFrame with columns as geo-features, and rows for each coordinate
        """

        features = self.embedder.transform(input_qs, calc_geo_features_in_batches=calc_geo_features_in_batches,
                                           batch_size=batch_size)
        return pd.DataFrame(features.values, index=input_qs, columns=features.columns)
