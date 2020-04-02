import pickle

from staticmap import StaticMap
from tqdm import tqdm

from coord2vec.Noam_Adir.pipeline.base_pipeline import *
from coord2vec.Noam_Adir.pipeline.preprocess import generic_clean_col, \
    ALL_MANHATTAN_FILTER_FUNCS_LIST
from coord2vec.config import *
from coord2vec.image_extraction.tile_image import render_single_tile
from coord2vec.image_extraction.tile_utils import build_tile_extent
from coord2vec.Noam_Adir.Adir.tile2vec.tile2vec_utils import *


def long_get_manhattan_df_from_pickle():
    MANHATTAN_PKL_PATH = r'/data/home/morpheus/coord2vec_Adir/coord2vec/Noam_Adir/Adir/tile2vec/tiles_data/manhattan_house_prices.pkl'
    pickle_file_name = MANHATTAN_PKL_PATH
    manhattan_df = load_from_pickle_features_manhattan(f"{pickle_file_name}", "lon", "lat")
    manhattan_df = generic_clean_col(manhattan_df, ALL_MANHATTAN_FILTER_FUNCS_LIST)
    cleaned_features = manhattan_df[['sold', 'numBedrooms', 'numBathrooms', 'sqft', 'coord']]
    all_features = extract_geographical_features(cleaned_features)

    # print(all_features)
    return all_features


def save_to_pickle_features_manhattan(file_path, all_features):
    pickle_out_features = open(file_path, "wb")
    pickle.dump(all_features, pickle_out_features)
    pickle_out_features.close()


# save_to_pickle_features_manhattan('cleaned_manhattan_features_df', long_get_manhattan_df_from_pickle())

def load_from_pickle_features_manhattan(file_path):
    pickle_in_features = open(file_path, "rb")
    features = pickle.load(pickle_in_features)
    pickle_in_features.close()
    return features


def get_manhattan_df_from_cleaned_pickle():
    file_path = r'/data/home/morpheus/coord2vec_Adir/coord2vec/Noam_Adir/Adir/tile2vec/tiles_data/cleaned_manhattan_features_df'
    all_features = load_from_pickle_features_manhattan(file_path)
    return all_features


def get_render_tiles_from_server_and_save_it():
    # there are 16006 coords for conveinence I take 16000
    NUM_OF_TILES = 5  # jz:as constant
    features = get_manhattan_df_from_cleaned_pickle()[:NUM_OF_TILES]
    coords = features['coord'].values
    coords_unique, indexes_unique = np.unique(coords, return_index=True)

    tiles_lst = []
    for i in tqdm(range(NUM_OF_TILES)):  # jz: tqdm
        center = [coords[i][1], coords[i][0]]
        m = StaticMap(IMG_WIDTH, IMG_HEIGHT, url_template='http://40.127.166.177:8103/tile/{z}/{x}/{y}.png',
                      delay_between_retries=15, tile_request_timeout=5)
        ext = build_tile_extent(center, radius_in_meters=50)  # jz: 50 as constant
        tile = np.array(render_single_tile(m, ext))
        tiles_lst.append(tile)
        # if i % 1000 == 999:
        #     print(f'{i + 1} tiles has been successfully loaded')
        #     tiles = np.stack(tiles_lst)
        #     tiles_lst = []
        #     np.save(f'tiles_data/parts/tiles_till{i + 1}', tiles)
    for tile in tqdm(tiles_lst):
        plt.imshow(tile)
        plt.show()

    print('done saving tiles :)')


def merge_parts_of_tiles_files():
    tiles_lst = []
    for i in range(16):
        tile = np.load(f'tiles_data/parts/tiles_till{i + 1}000.npy')
        tiles_lst.append(tile)
    tiles = np.stack(tiles_lst).reshape(16000, 224, 224, 3)  # jz: constants
    np.save(f'tiles_data/16000_tiles_images', tiles)  # jz:final
    print(tiles.shape)


# from timeit import timeit
# check how much time loading data
# print(timeit("np.load('tiles_data/16000_tiles_images.npy')", setup='import numpy as np', number=1))


def show_some_tiles_images(tiles, n_tiles=16000, nrows=3, ncols=5, with_red_mark=False, with_hist_eq=False):
    random_indexes = np.random.randint(0, n_tiles, nrows * ncols)
    tiles_to_show = [tiles[i].copy() for i in random_indexes]
    if with_hist_eq:
        tiles_to_show = list(map(equalize_hist, tiles_to_show))
    if with_red_mark:
        tiles_to_show = map(draw_circle, tiles_to_show)
        tiles_to_show = list(map(lambda x: draw_circle(x, radius=40, fill=False), tiles_to_show))
    display_grid(tiles_to_show, ncols=ncols)


def get_fast_the_tiles():
    tiles = np.load('tiles_data/16000_tiles_images.npy')
    return tiles
