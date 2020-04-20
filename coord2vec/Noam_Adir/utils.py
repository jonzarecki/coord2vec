import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from staticmap import StaticMap
from tqdm import tqdm
from coord2vec.image_extraction.tile_utils import build_tile_extent
# from coord2vec.Noam_Adir.Adir.loc2vec.tile_image import render_single_tile
from coord2vec.image_extraction.tile_image import render_single_tile

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


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def render_tiles_from_coords(coords, url_template, img_width, img_height, save_path, idx_path):
    idx_lst = []
    tiles_lst = []
    for i in tqdm(range(len(coords)), desc='rendering tiles', unit='tile'):
        center = [coords[i][1], coords[i][0]]
        m = StaticMap(img_width, img_height, url_template=url_template,
                      delay_between_retries=15, tile_request_timeout=5)
        lon, lat = center
        ext = [lon - 0.0005, lat - 0.0005, lon + 0.0005, lat + 0.0005]
        tile = np.array(render_single_tile(m, ext))
        if tile.mean() > 220:  # not new-york city
            continue
        idx_lst.append(i)
        tiles_lst.append(tile)
    tiles = np.stack(tiles_lst)
    indexes = np.stack(idx_lst)
    print(tiles.shape)
    np.save(save_path, tiles)
    np.save(idx_path, indexes)
    # for tile in tqdm(tiles_lst):
    #     plt.imshow(tile)
    #     plt.show()

    print('done saving tiles :)')
