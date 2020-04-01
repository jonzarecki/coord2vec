import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import parmap
from geopandas import GeoDataFrame
from shapely.geometry import Point
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from coord2vec.Noam_Adir.pipeline.utils import get_non_repeating_coords
from coord2vec.config import BUILDINGS_FEATURES_TABLE
from coord2vec.feature_extraction.feature_bundles import karka_bundle_features, create_building_features
from coord2vec.feature_extraction.features_builders import FeaturesBuilder

from coord2vec.Noam_Adir.pipeline.preprocess import get_csv_data, get_manhattan_data
from coord2vec.Noam_Adir.pipeline.preprocess import generic_clean_col, ALL_FILTER_FUNCS_LIST, ALL_MANHATTAN_FILTER_FUNCS_LIST


def get_data(dataset="manhattan", use_all_data=True):
    if dataset == "manhattan":
        if not use_all_data:
            debug_cache_filename = "small_data_manhattan.pickle"
        else:
            debug_cache_filename = "full_data_manhattan.pickle"
        if os.path.isfile(debug_cache_filename):
            features, y_col = pickle.load(open(debug_cache_filename, "rb"))
        else:
            features, y_col = extract_and_filter_manhattan_data(use_full_dataset=use_all_data)
            features = extract_geographical_features(features)
            pickle.dump((features, y_col), open(debug_cache_filename, "wb"))
    else:  # dataset == "beijing":
        all_features, y_col = extract_and_filter_csv_data(use_full_dataset=use_all_data)
        all_features = extract_geographical_features(all_features)
        features = get_non_repeating_coords(all_features)

    features = features.set_index("coord_id").sort_index()
    X = features.drop(columns=["coord", y_col]).values.astype(float)
    y = features[y_col].values.astype(float)[:, None]
    coords = features["coord"]
    return coords, X, y


def extract_and_filter_csv_data(clean_funcs=ALL_FILTER_FUNCS_LIST,
                                use_full_dataset=True) -> Tuple[pd.DataFrame, str]:
    """
    load anf filter data from csv, at the moment uses get_csv_data specific for Beijing dataset
    Args:
        clean_funcs: the filter functions to use on the csv raw data
        use_full_dataset: parameter for the csv loading function

    Returns: tuple of : filtered DataFrame and the label column name

    """
    csv_features, y_col = get_csv_data(use_full_dataset=use_full_dataset)
    cleaned_features = generic_clean_col(csv_features, clean_funcs)
    return cleaned_features, y_col


def extract_and_filter_manhattan_data(clean_funcs=ALL_MANHATTAN_FILTER_FUNCS_LIST,
                                      use_full_dataset=True) -> Tuple[pd.DataFrame, str]:
    """
    apply filter functions to the manhatta data loaded from pickle
    Args:
        clean_funcs: the functions to apply
        use_full_dataset: if True gets all the data else return according to get_manhattan_data

    Returns: tuple of : filtered DataFrame and the label column name

    """
    pickle_features, y_col = get_manhattan_data(use_full_dataset=use_full_dataset, non_repeating_coord=True)
    cleaned_features = generic_clean_col(pickle_features, clean_funcs)
    return cleaned_features, y_col


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
        models, scores, y_test
    """
    X_train, y_train, X_test, y_test = train_test_extraction_func(all_features, drop_cols=drop_cols, y_col=y_col)
    return train_models_from_splitted_data(models, X_train, y_train, X_test, y_test)


def train_models_from_splitted_data(models, X_train, y_train, X_test, y_test):
    """
    train each model on X_train, y_train and return mse accuracies on X_test, y_test as scores
    Args:
        models: models
        X_train: (n_train, D)
        y_train: (n_train, )
        X_test: (n_test, D)
        y_test: (n_test, )

    Returns:
        models, scores, y_test
    """
    scores = []
    for model in models:
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        scores.append(mean_absolute_error(y_test, y_test_pred))
    return models, scores, y_test


def my_z_score_norm(train: np.ndarray, test: np.ndarray = None, return_scalers=False):
    """
    operate a z_score normalization on train and test dataset.
    Args:
        train: array with shape (n_train, D) D is the dimension of the features.
        test: array with shape (n_test, D) or None
        return_scalers: bool - True means return the norm scalers

    Returns:
        normalized train if test is None
        else returns also normalized test according to train norm expectation and std

    """
    if len(train.shape) == 1:
        train = train[:, None]
    normalizer = StandardScaler()
    normalizer.fit(train)
    norm_train = normalizer.transform(train)
    if test is not None:
        if len(test.shape) == 1:
            test = test[:, None]
        norm_test = normalizer.transform(test)
        if not return_scalers:
            return norm_train, norm_test
        else:
            return norm_train, norm_test, normalizer
    return norm_train if not return_scalers else norm_train, normalizer


def train_models_from_generic_dict_with_norm(num_iter: int, model, X_dict: dict, y: np.ndarray) -> List[float]:
    """
    Train model on each dataset in X_dict.values() and check mse-accuracy on y
    we'll call this training experiment
    Args:
        num_iter: num of times we want to execute the experiment
        (used only from wraps function that run these experiments paralleled)
        model: a model we want to train
        X_dict: (name_of_dataset: dataset) for each dataset we want to train on
        y: target dataset on which we are compute the mse accuracy

    Returns:
        acc_lst list of accuracies per dataset in X_dict.values() on y
    """
    N = y.shape[0]
    acc_lst = []
    indexes = np.arange(N)
    np.random.seed()
    np.random.shuffle(indexes)
    train_ind, test_ind = train_test_split(indexes)
    y_train, y_test = y[train_ind], y[test_ind]
    # y_norm_train, y_norm_test, normalizer = my_z_score_norm(y_train, y_test, return_scalers=True)
    for title, X in X_dict.items():
        X_train, X_test = X[train_ind], X[test_ind]
        X_norm_train, X_norm_test = my_z_score_norm(X_train, X_test)
        args_tuple = ([model], X_norm_train, y_train, X_norm_test, y_test)
        acc = train_models_from_splitted_data(*args_tuple)[1][0]
        acc_lst.append(acc)
    return acc_lst


# helper function
def train_n_iter(model, X_dict: dict, y: np.ndarray, num_iter=1) -> dict:
    """
    Train model on each dataset in X_dict.values() and check mse-accuracy on y num_iter times in parallel
    Args:
        model: a model we want to train
        X_dict: (name_of_dataset: dataset) for each dataset we want to train on
        y: target dataset on which we are compute the mse accuracy
        num_iter: number of times we do the experiment.

    Returns:
        acc_dict (name of dataset: mean accuracy over num iter times on dataset)
    """

    accs_per_model = parmap.map(train_models_from_generic_dict_with_norm,
                                range(num_iter), model, X_dict, y, pm_processes=8, pm_pbar=True)
    mean_acc_per_model = np.mean(np.array(accs_per_model), axis=0).astype(int)  # no norm for y
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
