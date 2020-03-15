import os
import pickle
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import sklearn
from geopandas import GeoSeries
from shapely import wkt
from shapely.geometry import Polygon, Point, GeometryCollection
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from coord2vec.common.geographic.geo_utils import sample_points_in_poly
from coord2vec.common.parallel.multiproc_util import parmap
from coord2vec.config import TRUE_POSITIVE_RADIUS, DISTANCE_CACHE_DIR
from coord2vec.evaluation.evaluation_metrics.metrics import soft_precision_recall_fscore, soft_precision_recall_curve
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.feature_extraction.osm.osm_utils import extract_buildings_from_polygons


class TaskHandler(ABC):
    """
    Abstract class for evaluation tasks
    """

    def __init__(self, embedder: FeaturesBuilder, models: Dict[str, BaseEstimator],
                 bounding_geom: Union[List[Polygon], Polygon] = None):
        self.embedder = embedder

        self.bounding_geom = bounding_geom  # TODO: add documentation
        self.true_geos = None

        # self.models_dict = {model_name: Pipeline([('std', StandardScaler()), (model_name, model)]) for model_name, model in models.items()}
        self.models_dict = models
        self.positive_sampling = 1
        self.noise_sampling_to_positive = 0
        self.models_dir = f"/repositories/cached_models/{self.__class__.__name__}"
        os.makedirs(self.models_dir, exist_ok=True)

    @abstractmethod
    def get_dataset(self) -> Tuple[GeoSeries, List[object]]:
        """
        get the data for the task
        Returns:
            geoms, y
        """
        pass

    @staticmethod
    def set_y_true_from_geoms(pred_geoms: GeoSeries, true_geoms: GeoSeries) -> List[bool]:
        """
        Builds the true class list for $pred_geoms, True is defined if the geometry is within an element in $true_geoms
        :param pred_geoms: The geometries we want to build a true label list for
        :param true_geoms: The geometries we define as "True"
        :return: A list
        """
        y_true = []
        for geom in pred_geoms:
            for true_geom in true_geoms:
                if geom.within(true_geom):
                    y_true.append(True)
                    break
            else:
                y_true.append(False)

        return y_true

    def sample_points_from_get_data(self, train_geos: GeoSeries, test_geos: GeoSeries,
                                    y_train: List[object], y_test: List[object], seed=None, samples_per_polygon=5) \
            -> Tuple[GeoSeries, List[float], GeoSeries, List[float]]:
        """
        Samples coordinates from the geoms received from get_data, refer to it on parameters
        """
        sampled_train_points, y_train = self.sample_points_from_geos(train_geos, y_train, seed,
                                                                     samples_per_polygon=samples_per_polygon)
        sampled_test_points, y_test = self.sample_points_from_geos(test_geos, y_test, seed)

        return sampled_train_points, y_train, sampled_test_points, y_test

    @staticmethod
    def sample_points_from_geos(geos: GeoSeries, y: List[object] = None, seed=None, samples_per_polygon=5) -> \
            Union[GeoSeries, Tuple[GeoSeries, List[object]]]:
        """
        Samples a list of coordinates from geoms (point or polygon). WITHOUT RETURNS
        For now, sampling from a point is taking the point, from poly is sampling 2 points.
        Args:
            geos: The list of geometries to sample from
            y: Optional parameter if we want to match y labels to the samples
            seed: Optional parameter to set random sampling seed
            samples_per_polygon: number of points to sample from each polygon in geo

        Returns:
            A list of coords as GeoSeries if y=None, else Tuple[GeoSeries, List]
        """
        assert len(geos.apply(lambda g: g.wkt).unique()) == len(geos), "Shouldn't have duplicates when sampling"

        sampled_geos = []
        sampled_y = []
        for i, geo in enumerate(geos):
            if isinstance(geo, Polygon):
                geo_samples = TaskHandler.sample_points_from_geo(geo, sampled_geos, samples_per_polygon, seed)
            elif isinstance(geo, Point):
                geo_samples = [geo]
            elif isinstance(geo, GeometryCollection):
                geo_samples = []
                for cur_geo in geo:
                    geo_samples += TaskHandler.sample_points_from_geo(cur_geo, sampled_geos, samples_per_polygon, seed)
            else:
                raise AssertionError(f"geo type {type(geo)} is not supported (yet)")
            sampled_geos += geo_samples
            if y is not None:
                sampled_y += [y[i]] * len(geo_samples)

        sampled_gs = GeoSeries(sampled_geos)
        return (sampled_gs, sampled_y) if y is not None else sampled_gs

    @staticmethod
    def sample_points_from_geo(geo, sampled_geos, samples_per_polygon, seed):
        all_distinct = False
        count = 0
        geo_samples = []
        while not all_distinct and count <= 100:
            geo_samples = sample_points_in_poly(geo, num_samples=samples_per_polygon, seed=seed)
            all_distinct = len(set([p.wkt for p in sampled_geos] + [p.wkt for p in geo_samples])) \
                           == len(geo_samples) + len(sampled_geos)  # all unique
            count += 1
        if count == 100:
            raise AssertionError("Can't sample distinct points")
        return geo_samples

    def transform(self, input_qs: GeoSeries) -> pd.DataFrame:
        """
        Transform the coordinates into the geographic features with self.embedder
        Args:
            input_qs: geoseries with the desired geometries

        Returns:
            a pandas DataFrame with columns as geo-features, and rows for each coordinate
        """
        features = self.embedder.transform(input_qs)
        return pd.DataFrame(features.values, index=input_qs, columns=features.columns)

    def fit_all_models(self, x: pd.DataFrame, y_true, cv=None):
        # trained_models_and_scores = parmap(lambda model: model.fit(x, y_true, cv=cv), self.models_dict.items(),
        #                                    use_tqdm=True, desc="Fitting Models", unit="model")
        # y_true_soft = self.get_soft_labels(gpd.GeoSeries(data=x.index.values), radius=TRUE_POSITIVE_RADIUS,
        #                                    cache_dir=DISTANCE_CACHE_DIR)

        # y_true_soft = y_true  # TODO delete this and uncomment last row

        def fit_model(model):
            if cv is not None:  # TODO is this really needed?
                model.fit(x, y_true, cv=cv)
            else:
                model.fit(x, y_true)
            return model

        models = parmap(fit_model, self.models_dict.values(), use_tqdm=True, desc="Fitting Models", unit="model", nprocs=32)
        # for name, model in tqdm(self.models_dict.items(), desc="Fitting Models", unit="model"):
        return models

    def _combine_results(self, model_results):
        num_kfold = len(model_results[0])
        results = []
        for kfold in range(num_kfold):
            kfold_results = {}

            kfold_results['X_train_df'] = model_results[0][kfold]['X_train_df']
            kfold_results['X_test_df'] = model_results[0][kfold]['X_test_df']
            kfold_results['y_train'] = model_results[0][kfold]['y_train']
            kfold_results['y_test'] = model_results[0][kfold]['y_test']

            model_dict = {}
            train_probas = {}
            test_probas = {}
            auc_scores = {}
            for model_result in model_results:
                kfold_model_result = model_result[kfold]
                model_dict.update(kfold_model_result['model_dict'])

                model_name = list(kfold_model_result['model_dict'].keys())[0]
                train_probas[model_name] = kfold_model_result['train_probas_df']
                test_probas[model_name] = kfold_model_result['test_probas_df']
                auc_scores[model_name] = kfold_model_result['auc_score']

            kfold_results['models_dict'] = model_dict
            kfold_results['train_probas_df'] = pd.DataFrame(train_probas)
            print(pd.DataFrame(train_probas).shape, len(kfold_results['X_train_df'].index))
            kfold_results['train_probas_df'].index = kfold_results['X_train_df'].index
            kfold_results['test_probas_df'] = pd.DataFrame(test_probas)
            kfold_results['test_probas_df'].index = kfold_results['X_test_df'].index
            kfold_results['model2score'] = auc_scores

            results.append(kfold_results)
        return results

    def save_all_models(self):
        """
        Save all the models_dict into the cache
        """
        for model_name, model in self.models_dict.items():
            with open(os.path.join(self.models_dir, str(hash(model_name))), 'wb') as f:
                # model.clear_all_hp_variables()
                pickle.dump(model, f)

    def predict_all_models(self, x: pd.DataFrame, use_cache: bool = False) -> pd.DataFrame:
        """
        predict this task using all the different cached trained models_dict
        Args:
            x: a Table of of all the samples features

        Returns:
            A dataframe of shape [n_samples, n_models], has the same index as $x
        """
        results = {}
        for model_name, model in tqdm(self.models_dict.items(), desc="Predicting Models", unit='model'):
            if use_cache:
                with open(os.path.join(self.models_dir, model_name), 'rb') as f:
                    model = pickle.load(f)
            y_pred = model.predict_proba(x)[:, 1]  # only on the True class
            results[model_name] = y_pred
        return pd.DataFrame(results, index=x.index)

    def score_all_models(self, x: pd.DataFrame, y: Union[List, np.array], use_cache: bool = False) -> dict:
        """
        score this task using all the different cached trained models_dict
        Args:
            x: a Table of of all the samples features
            y: list of all the true labels
        Returns:
            scores: dictionary from experiment to model score
        """
        scores = {}
        normalized_x = self.normalize(x)
        for model_name, model in tqdm(self.models_dict.items(), desc="Scoring Models", unit='model'):
            if use_cache:
                with open(os.path.join(self.models_dir, model_name), 'rb') as f:
                    model = pickle.load(f)
            precision, recall, _ = soft_precision_recall_curve(y, model.predict(x))
            scores[model_name] = sklearn.metrics.auc(recall, precision)
        return scores

    def transform_and_predict(self, gs: GeoSeries) -> pd.DataFrame:
        """
        predict this task using all the different cached trained models_dict, using raw Points
        Args:
            x: Geo Series of all the geometries to predict on

        Returns:
            A dataframe of shape [n_samples, n_models]
        """
        x_df = self.transform(gs)
        return self.predict_all_models(x_df)

    def normalize(self, x):
        std = x.std(axis=0).replace(0, 1)
        x = (x - x.mean(axis=0)) / std
        return x

    def extract_buildings_from_polygons(self, polys: GeoSeries, y: Union[List, np.array], return_source=False,
                                        positive_sampling=1., noise_sampling_to_positive=0.) -> \
            Union[Tuple[gpd.GeoSeries, List[bool]],
                  Tuple[gpd.GeoSeries, List[bool], np.array]]:
        """
        convert the GT polygons to it's buildings
        Args:
            polys: polygons
            y: the label (0 or 1) of each polygon in polys
            return_source: True if want to return the source indices of 'buildings_gs' in 'polys', False otherwise
            positive_sampling: ratio of positive samples from a building to keep
            noise_sampling_to_positive:  ratio of negative exmaple to add as noise (with positive labels)

        Returns:
            buildings_gs: A series of all the buildings
            buildings_y: their labels
            source_indices: (optional) if return_source==True return np.array in len of buildings_gs mapping to
                            source_index in 'polys'
        """
        y = np.array(y)

        buildings_gs, source_indices = extract_buildings_from_polygons(polys, return_source=True)
        buildings_y = y[source_indices]

        self.positive_sampling = positive_sampling  # signals that we performed that
        self.noise_sampling_to_positive = noise_sampling_to_positive
        pos_idxs = [i for i in range(len(buildings_y)) if buildings_y[i] == 1]
        if positive_sampling < 1.:
            import math
            chosen_pos_idxs = np.random.choice(pos_idxs, math.ceil(positive_sampling * len(pos_idxs)), replace=False)

            all_idxs = [i for i in range(len(buildings_y)) if buildings_y[i] == 0 or i in chosen_pos_idxs]  # keeps order
            buildings_gs = buildings_gs[all_idxs]
            buildings_y = buildings_y[all_idxs]
            source_indices = source_indices[all_idxs]

        if 0. < noise_sampling_to_positive <= 1.:
            neg_idxs = [i for i in range(len(buildings_y)) if buildings_y[i] == 0]
            chosen_noise_idxs = np.random.choice(neg_idxs, round(noise_sampling_to_positive * len(pos_idxs)), replace=False)

            buildings_y[chosen_noise_idxs] = 1  # change to True

        if return_source:
            return buildings_gs, buildings_y, source_indices
        else:
            return buildings_gs, buildings_y

    def train_test_split(self, geos: gpd.GeoSeries, y: Union[List[int], np.array], test_size: float = 0.25):
        """
        split the data for the task
        Inheriting classes can override this function to i.e split by geographic area
        Args:
            geos: geometries of both positive and negative samples
            y: the label (0 or 1) of each geo in geos
            test_size: the relative size of the test set

        Returns:
            train_polygons, test_polygons, y_train_geos, y_test_geos
        """
        return train_test_split(geos, y, test_size=test_size)

    def kfold_split(self, geos: gpd.GeoSeries, y: Union[List[int], np.array], n_splits: int = 5, **kwargs) -> List[
        Tuple[List[object], List[object]]]:
        """
        split the data for the task in k_folds
        Args:
            geos: geometries of both positive and negative samples
            y: the label (0 or 1) of each geo in geos
            n_splits: number of splits in the k_fold

        Returns:
            list of tuples: (train_index, test_index)
        """
        return list(KFold(n_splits=n_splits).split(geos, y))
