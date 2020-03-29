from typing import List, Tuple

import numpy as np
import pandas as pd
import parmap
from geopandas import GeoDataFrame
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from coord2vec.Noam_Adir.pipeline.preprocess import generic_clean_col, ALL_FILTER_FUNCS_LIST
from coord2vec.Noam_Adir.pipeline.preprocess import get_csv_data
from coord2vec.config import BUILDINGS_FEATURES_TABLE
from coord2vec.feature_extraction.feature_bundles import karka_bundle_features, create_building_features
from coord2vec.feature_extraction.features_builders import FeaturesBuilder


def extract_and_filter_csv_data(clean_funcs=ALL_FILTER_FUNCS_LIST, use_full_dataset=True) -> pd.DataFrame:
    """
    load anf filter data from csv, at the moment uses get_csv_data specific for Beijing dataset
    Args:
        clean_funcs: the filter functions to use on the csv raw data
        use_full_dataset: parameter for the csv loading function

    Returns: filtered DataFrame

    """
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
                1."coord_id" - if the columns exist may change it
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


def extract_train_test_set_from_features(all_features: pd.DataFrame, drop_cols: List[str], y_col: str) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    simple method to split DataFrame into train test splits, allow dropping of columns
    Args:
        all_features: the DataFrame of all the features, contains the label in y_col column
        drop_cols: list o columns from the DataFrame to drop
        y_col: the column in the features DataFrame of the label

    Returns: tuple of 4 np.ndarry - (X_train, y_train, X_test, y_test)

    """
    X = all_features.drop(columns=drop_cols).drop(columns=[y_col]).values
    y = all_features[y_col].values
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    return X_train, y_train, X_test, y_test


def train_models(models, all_features: pd.DataFrame, train_test_extraction_func=extract_train_test_set_from_features,
                 drop_cols=["coord", "coord_id"], y_col="totalPrice"):
    """
    a function to train models on df with features and labels, defaults for Beijing dataset for backward compatibility
    Args:
        models: list if models with fit(X,y) and predict(X) methods
        all_features: the data frame off the features and the labels
        train_test_extraction_func: a function to extract from the data frame train and test sets
        drop_cols: list of columns to drop from the dataframe
        y_col: the name of the label column

    Returns:

    """
    X_train, y_train, X_test, y_test = train_test_extraction_func(all_features, drop_cols=drop_cols, y_col=y_col)
    return train_models_from_splitted_data(models, X_train, y_train, X_test, y_test)


def train_models_from_splitted_data(models, X_train, y_train, X_test, y_test):
    scores = []
    for model in models:
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        scores.append(mean_squared_error(y_test, y_test_pred))
    return models, scores, y_test


def my_z_score_norm(train, test):
    if len(train.shape) == 1:
        train = train[:, None]
    if len(test.shape) == 1:
        test = test[:, None]
    normalizer = StandardScaler()
    normalizer.fit(train)
    norm_train = normalizer.transform(train)
    norm_test = normalizer.transform(test)
    return norm_train, norm_test


def train_models_from_generic_dict_with_norm(num_iter, models, X_dict: np.ndarray, y: np.ndarray):
    N = y.shape[0]
    acc_lst = []
    indexes = np.arange(N)
    train_ind, test_ind = train_test_split(indexes)
    y_train, y_test = y[train_ind], y[test_ind]
    y_norm_train, y_norm_test = my_z_score_norm(y_train, y_test)
    for title, X in X_dict.items():
        X_train, X_test = X[train_ind], X[test_ind]
        X_norm_train, X_norm_test = my_z_score_norm(X_train, X_test)
        args_tuple = (models, X_norm_train, y_norm_train, X_norm_test, y_norm_test)
        acc = train_models_from_splitted_data(*args_tuple)[1][0]
        acc_lst.append(acc)
    return acc_lst


# helper function
def train_n_iter(models, X_dict: np.ndarray, y: np.ndarray, num_iter=1):
    accs_per_model = parmap.map(train_models_from_generic_dict_with_norm,
                                range(num_iter), models, X_dict, y, pm_processes=8, pm_pbar=True)
    mean_acc_per_model = np.mean(np.array(accs_per_model), axis=0)
    acc_dict = {title + f"_mean_acc_for_{num_iter}_iter": mean_acc_per_model[i]
                for i, title in enumerate(X_dict.keys())}
    return acc_dict


def plot_scores(training_cache):
    """
    print regression scores from the cache returned from the train_models function
    Args:
        training_cache: cache from train_models function

    """
    models, scores, y_test = training_cache
    print("mean price - ", np.mean(y_test))
    print(f"MSE: linear regression - {scores[0]}, catboost - {scores[1]}")
    print(f"RMSE: linear regression - {np.sqrt(scores[0])}, catboost - {np.sqrt(scores[1])}")
