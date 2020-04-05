import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from typing import Tuple, Any, List
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point
from sklearn import neighbors
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


# models = [svm.SVR()
#     , neighbors.KNeighborsRegressor(n_neighbors=10)
#     , LinearRegression()
#     , tree.DecisionTreeRegressor()
#     , GradientBoostingRegressor()
#     , AdaBoostRegressor()
#     , RandomForestRegressor()
#     , CatBoostRegressor()
#           ]


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
        'task_handler': Manhattan_Task_Handler
        , 'coords': coords
        , 'unique_coords': unique_coords
        , 'unique_coords_idx': indexes of unique coords (coords[unique_coords_idx] is equivalent to unique_coords)
        , 'price': price (y_true)
        , 'features_without_geo': features_without_geo
        , 'only_geo_features_unique_coords': only_geo_features_unique_coords
        , 'all_features': features_without_geo merge with only_geo_features
        , 'all_features_unique_coords': all_features_unique_coords
    }
    """
    # models_dict = {model.__class__.__name__: model for model in models}

    # build the embedder (FeatureBuilder)
    building_features = create_building_features(karka_bundle_features)
    embedder = FeaturesBuilder(building_features, cache_table=BUILDINGS_FEATURES_TABLE)

    # create the task_handler
    task_handler = Manhattan_Task_Handler(embedder, models=models)

    coords, features_without_geo, price = task_handler.get_dataset(all_dataset=False)

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


# print("\n".join([f'{k} = pipeline_dict["{k}"]' for k, v in pipeline_dict.items()]))

if __name__ == "__main__":
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
