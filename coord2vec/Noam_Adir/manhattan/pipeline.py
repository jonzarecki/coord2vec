import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from typing import Tuple, Any, List, Dict
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point
from sklearn import neighbors
from tqdm import tqdm

from coord2vec.Noam_Adir.utils import norm_for_train_and_test
from coord2vec.common.parallel.multiproc_util import parmap
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from coord2vec.Noam_Adir.manhattan.manhattan_feature_builder import Manhattan_Feature_Builder
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.Noam_Adir.manhattan.manhattan_task_handler import Manhattan_Task_Handler
from coord2vec.config import BUILDINGS_FEATURES_TABLE
from coord2vec.feature_extraction.feature_bundles import karka_bundle_features, create_building_features


# helper function that merge non_geo_features with geo_features
def create_full_df(unique_coords_array, features_without_geo, only_geo_features_unique_coords, coords, unique_idx):
    coord_2_coord_id = {coord: coord_id for coord_id, coord in enumerate(unique_coords_array)}
    copy_features_without_geo = features_without_geo.copy()
    copy_only_geo_features_unique_coords = only_geo_features_unique_coords.copy()
    copy_features_without_geo['coord_id'] = [coord_2_coord_id[coord] for coord in coords]
    copy_only_geo_features_unique_coords['coord_id'] = [coord_2_coord_id[coord] for coord in unique_coords_array]
    all_features = copy_features_without_geo.merge(copy_only_geo_features_unique_coords, left_on='coord_id',
                                                   right_on='coord_id',
                                                   how='left').drop(columns=['coord_id'])
    all_features_unique_coords = all_features.iloc[unique_idx]
    return all_features, all_features_unique_coords


def init_pipeline(models: List[Any]) -> dict:
    """
    example of usage in pipeline:

    pipeline_dict = init_pipeline(models=[LinerRegression()])
    task_handler = pipeline_dict['task_handler']
    features_without_geo = pipeline_dict['features_without_geo']
    price = pipeline_dict['price']
    task_handler.fit_all_models(features_without_geo, price)

    Args:
        models: list of models

    Returns:
        pipeline_dict that contains
        'task_handler' (Manhattan_Task_Handler): Manhattan_Task_Handler
        , 'coords' (ndarray): coords
        , 'unique_coords' (ndarray): unique_coords
        , 'unique_coords_idx' (ndarray): indexes of unique coords (coords[unique_coords_idx] is equivalent to unique_coords)
        , 'price' (ndarray): price (y_true)
        , 'features_without_geo' (DataFrame): features_without_geo
        , 'only_geo_features_unique_coords' (DataFrame): only_geo_features_unique_coords
        , 'all_features' (DataFrame): features_without_geo merge with only_geo_features
        , 'all_features_unique_coords' (DataFrame): all_features_unique_coords
    """
    # models_dict = {model.__class__.__name__: model for model in models}

    # build the embedder (FeatureBuilder)
    building_features = create_building_features(karka_bundle_features)
    embedder = FeaturesBuilder(building_features, cache_table=BUILDINGS_FEATURES_TABLE)

    # create the task_handler
    task_handler = Manhattan_Task_Handler(embedder, models=models)

    coords, features_without_geo, price = task_handler.get_dataset(all_dataset=True)

    # get the geo_features_unique_coords
    unique_coords_array, unique_idx = np.unique(coords, return_index=True)
    coords_geo_series = GeoSeries([Point(coord[0], coord[1]) for coord in unique_coords_array])
    only_geo_features_unique_coords = task_handler.transform(coords_geo_series)

    # get the all_features, all_features_unique_coords
    all_features, all_features_unique_coords = create_full_df(unique_coords_array, features_without_geo,
                                                              only_geo_features_unique_coords, coords, unique_idx)

    pipeline_dict = {
        'task_handler': task_handler
        , 'coords': coords
        , 'unique_coords': unique_coords_array
        , 'unique_coords_idx': unique_idx
        , 'price': price
        , 'features_without_geo': features_without_geo
        , 'only_geo_features_unique_coords': only_geo_features_unique_coords
        , 'all_features': all_features
        , 'all_features_unique_coords': all_features_unique_coords
    }
    return pipeline_dict


def fit_and_score_models_on_datasets(models: List[Any],
                                     data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """
    fit and score each model in models on each dataset in data_dict.values() and check mae-accuracy on y
    each dataset is normalized in this function
    Args:
        models: list of models
        data_dict: (name_of_dataset: (dataset, target)) for each dataset we want to train on

    Returns:
        data frame of mae-scores whose columns are dataset na mes and its index is model names
    """

    # helper function for parallelize (this also normalize the data)
    def fit_and_score(data_name_and_data_tuple):
        data_name, data = data_name_and_data_tuple
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_norm_train, X_norm_test = norm_for_train_and_test(X_train, X_test)
        task_handler = Manhattan_Task_Handler(embedder=None, models=models)
        task_handler.fit_all_models(X_norm_train, y_train)
        scores = task_handler.score_all_models(X_norm_test, y_test, measure_func=mean_absolute_error)
        return {data_name: scores}

    scores_lst = []
    for data_name, data in data_dict.items():
        print(f'handle dataset {data_name}')
        scores_lst.append(fit_and_score((data_name, data)))
    scores_dct = {k: v for d in scores_lst for k, v in d.items()}

    # scores_lst = parmap(fit_and_score, data_dict.items(), use_tqdm=True, desc="Fit and score per dataset",
    #                     unit="dataset", nprocs=32)
    # # union all the dicts in scores_lst to one dict
    # scores_dct = {k: v for d in scores_lst for k, v in d.items()}

    df_scores = pd.DataFrame(scores_dct)
    return df_scores


# print("\n".join([f'{k} = pipeline_dict["{k}"]' for k, v in pipeline_dict.items()]))


def test_init_pipeline():
    models = [svm.SVR(), LinearRegression()]
    pipeline_dict = init_pipeline(models)

    task_handler = pipeline_dict["task_handler"]
    coords = pipeline_dict["coords"]
    unique_coords = pipeline_dict["unique_coords"]
    unique_coords_idx = pipeline_dict["unique_coords_idx"]
    price = pipeline_dict["price"]
    features_without_geo = pipeline_dict["features_without_geo"]
    only_geo_features_unique_coords = pipeline_dict["only_geo_features_unique_coords"]
    all_features = pipeline_dict["all_features"]
    all_features_unique_coords = pipeline_dict["all_features_unique_coords"]

    task_handler.fit_all_models(all_features, price)
    print(task_handler.score_all_models(all_features, price, measure_func=mean_absolute_error))


def test_fit_and_score_models_on_datasets():
    models = [svm.SVR(), LinearRegression(), CatBoostRegressor(verbose=False)]
    pipeline_dict = init_pipeline(models)

    price = pipeline_dict["price"]
    features_without_geo = pipeline_dict["features_without_geo"].values
    all_features = pipeline_dict["all_features"].values

    data_dict = {'features_without_geo': (features_without_geo, price), 'all_features': (all_features, price)}
    print(fit_and_score_models_on_datasets(models, data_dict))


if __name__ == "__main__":
    models = [svm.SVR()
        , neighbors.KNeighborsRegressor(n_neighbors=10)
        , LinearRegression()
        , tree.DecisionTreeRegressor()
        , GradientBoostingRegressor()
        , AdaBoostRegressor()
        , RandomForestRegressor()
        , CatBoostRegressor()
              ]
    test_fit_and_score_models_on_datasets()
