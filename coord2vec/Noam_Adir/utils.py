import pickle

import pandas as pd


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
