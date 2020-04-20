import pickle
from itertools import chain
from typing import Tuple, Union, List, Dict, Callable

import numpy as np
import pandas as pd
from geopandas import GeoSeries
from shapely.geometry import Polygon
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from coord2vec.common.parallel.multiproc_util import parmap
from coord2vec.Noam_Adir.manhattan.preprocess import ALL_MANHATTAN_FILTER_FUNCS_LIST
from coord2vec.Noam_Adir.utils import load_from_pickle_features, generic_clean_col
from coord2vec.evaluation.tasks.task_handler import TaskHandler
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from files_path import *


class Manhattan_Task_Handler(TaskHandler):
    def __init__(self, embedder: FeaturesBuilder, models: List[BaseEstimator],
                 bounding_geom: Union[List[Polygon], Polygon] = None,
                 graph_models: Dict = None):
        super().__init__(embedder, models, bounding_geom=bounding_geom)
        self.graph_models_dict = {} if graph_models is None else graph_models

    def get_dataset(self, all_dataset: bool) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        """
        get the non-geo data for the task from manhattan pickle
        Returns:
            coords, features_without_geo, sold
        """
        df = load_from_pickle_features(MANHATTAN_PKL_PATH)
        df = generic_clean_col(df, ALL_MANHATTAN_FILTER_FUNCS_LIST)
        df = df[:16000] if all_dataset else df[:5]
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

    def score_all_model_multi_metrics(self, x: pd.DataFrame, y: Union[List, np.array], use_cache: bool = False,
                                      measure_funcs: Dict[str, Callable] = None) -> dict:
        if measure_funcs is None:
            measure_funcs = {"mse": mean_absolute_error}
        scores = {}

        for model_name, model in tqdm(chain(self.models_dict.items(), self.graph_models_dict),
                                      desc="Scoring Models", unit='model'):
            if use_cache:
                with open(os.path.join(self.models_dir, model_name), 'rb') as f:
                    model = pickle.load(f)
            y_pred = model.predict(x)
            model_scores = {}
            for metric_name, metric in measure_funcs.items():
                model_scores[metric_name] = metric(y_true=y, y_pred=y_pred)
            scores[model_name] = model_scores

        return scores

    def score_all_model_multi_metrics_idx(self, x: pd.DataFrame, y: Union[List, np.array], indexes,
                                          use_cache: bool = False,
                                          measure_funcs: Dict[str, Callable] = None) -> dict:
        if measure_funcs is None:
            measure_funcs = {"mse": mean_absolute_error}
        scores = {}

        for model_name, model in tqdm(self.models_dict.items(), desc="Scoring Models", unit='model'):
            y_pred = model.predict(x.iloc[indexes])
            model_scores = {}
            for metric_name, metric in measure_funcs.items():
                model_scores[metric_name] = metric(y_true=y[indexes], y_pred=y_pred)
            scores[model_name] = model_scores

        for model_name, model in tqdm(self.graph_models_dict.items(), desc="Scoring Graph Models", unit='model'):
            y_pred = model.predict_idx(indexes)
            model_scores = {}
            for metric_name, metric in measure_funcs.items():
                model_scores[metric_name] = metric(y_true=y[indexes], y_pred=y_pred)
            scores[model_name] = model_scores

        return scores

    def fit_all_models_with_idx(self, x: pd.DataFrame, y_true, train_idx):
        x_defualt_idx = x.reset_index(drop=True)
        for model_name, model in tqdm(self.models_dict.items(), desc="fitting Models", unit='model'):
            model.fit(x_defualt_idx.iloc[train_idx], y_true[train_idx])

        for model_name, model in tqdm(self.graph_models_dict.items(), desc="fitting graph Models", unit='model'):
            model.fit(x, y_true, train_idx)

    def add_graph_model(self, model):
        self.graph_models_dict[model.__class__.__name__] = model
