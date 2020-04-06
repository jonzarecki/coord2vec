import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def save_to_pickle_features(file_path, all_features):
    pickle_out_features = open(file_path, "wb")
    pickle.dump(all_features, pickle_out_features)
    pickle_out_features.close()


# save_to_pickle_features_manhattan('cleaned_manhattan_features_df', long_get_manhattan_df_from_pickle())

def load_from_pickle_features(file_path):
    pickle_in_features = open(file_path, "rb")
    features = pickle.load(pickle_in_features)
    pickle_in_features.close()
    return features


def generic_clean_col(df: pd.DataFrame, clean_funcs) -> pd.DataFrame:
    """
    apply functions of df and return new dataframe
    Args:
        df: data frame
        clean_funcs: list of funcs that clean cols that should be cleand in df

    Returns: cleaned_df w

    """
    for i, col in enumerate(clean_funcs):
        df = clean_funcs[i](df)
    cleaned_df = df.fillna(0)
    return cleaned_df

# from timeit import timeit
# check how much time loading data
# print(timeit("command", setup='import', number=3))

def norm_for_train_and_test(train: np.ndarray, test: np.ndarray = None, return_scalers=False):
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