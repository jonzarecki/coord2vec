from geopandas import GeoDataFrame
import pandas as pd
import numpy as np
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from coord2vec.config import BUILDINGS_FEATURES_TABLE
from coord2vec.feature_extraction.feature_bundles import karka_bundle_features, create_building_features
from coord2vec.feature_extraction.features_builders import FeaturesBuilder

from coord2vec.Noam_Adir.pipeline.preprocess import get_csv_data
from coord2vec.Noam_Adir.pipeline.preprocess import generic_clean_col, ALL_FILTER_FUNCS_LIST


def extract_and_filter_csv_data(clean_funcs=ALL_FILTER_FUNCS_LIST, use_full_dataset=True):
    csv_features = get_csv_data(use_full_dataset=use_full_dataset)
    cleaned_features = generic_clean_col(csv_features, clean_funcs)
    return cleaned_features


def extract_geographical_features(cleaned_features: pd.DataFrame, calculate_feats_in_batchs=True,
                                  batch_size=10000, features_bundle=karka_bundle_features):
    """
    Args:
        features_bundle: geographical features bundle from coord2vec
        cleaned_features: pd.DataFrame with "coord" column of coordinats, each row is data point
        calculate_feats_in_batchs: should the features be calculated in batches
        batch_size: a batch size for the geographical features

    Returns: all_features df the same legth as clearned features, with the following extra columns
                1."coord_id"
                2. all the geographical features related with the coord

    """
    cleaned_features = cleaned_features.copy()
    unique_coords = cleaned_features["coord"].unique()
    shapely_coords_unique = [Point(coord[0], coord[1]) for coord in unique_coords]
    coord2coord_id = {coord: i for i, coord in enumerate(cleaned_features["coord"].unique())}
    cleaned_features["coord_id"] = cleaned_features["coord"].apply(lambda coord: coord2coord_id[coord])

    geo_feats = create_building_features(features_bundle)
    builder = FeaturesBuilder(geo_feats, cache_table=BUILDINGS_FEATURES_TABLE)
    gdf = GeoDataFrame(pd.DataFrame({'geom': shapely_coords_unique}), geometry='geom')

    geo_results_list = []
    n_samples = len(gdf.geometry)
    calculate_geo_features_with_batches = calculate_feats_in_batchs
    if (not calculate_geo_features_with_batches) or (n_samples <= batch_size):
        geo_results = builder.transform(gdf.geometry)
    else:
        for batch_start_ind in range(0, n_samples, batch_size):
            batch_end_ind = batch_start_ind + batch_size if batch_start_ind + batch_size < n_samples else n_samples
            geo_results_list.append(builder.transform(gdf.geometry[batch_start_ind:batch_end_ind]))
        geo_results = pd.concat(geo_results_list)
        geo_results = geo_results.reset_index(drop=True)

    all_features = cleaned_features.merge(geo_results, left_on='coord_id', right_index=True, how='left')
    return all_features


def train_models(models, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
    scores = []
    for model in models:
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        scores.append(mean_squared_error(y_test, y_test_pred))
    return models, scores


def extract_train_test_set_from_features(all_features):
    X = all_features.drop(columns=["coord", "coord_id", "totalPrice"]).values
    y = all_features['totalPrice'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, y_train, X_test, y_test


def plot_scores(training_cache):
    models, scores, y_test = training_cache
    print("mean price - ", np.mean(y_test))
    print(f"MSE: linear regression - {scores[0]}, catboost - {scores[1]}")
    print(f"RMSE: linear regression - {np.sqrt(scores[0])}, catboost - {np.sqrt(scores[1])}")
