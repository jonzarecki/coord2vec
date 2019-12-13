import os
import pickle
import shutil
import time

import numpy as np
from tqdm.auto import tqdm

from coord2vec import config
from coord2vec.common import multiproc_util
from coord2vec.common.multiproc_util import parmap
from coord2vec.config import TRAIN_CACHE_DIR, VAL_CACHE_DIR, VAL_SAMPLE_NUM, TRAIN_SAMPLE_NUM, ENTROPY_THRESHOLD, \
    HALF_TILE_LENGTH, tile_server_ports, get_builder
from coord2vec.feature_extraction.features_builders import example_features_builder, house_price_builder, \
    only_build_area_builder
from coord2vec.feature_extraction.osm import OsmPolygonFeature
from coord2vec.feature_extraction.osm.osm_tag_filters import BUILDING
from coord2vec.image_extraction.tile_image import render_multi_channel, generate_static_maps
from coord2vec.image_extraction.tile_utils import sample_coordinate_in_range, build_tile_extent


def feature_extractor(coord) -> np.array:
    # placeholder for building more complicated features
    building_count_feat = OsmPolygonFeature(BUILDING, 'number_of', max_radius=50)
    res = building_count_feat.extract_single_coord(coord)
    return np.array([res])


def _get_image_entropy(image):
    signal = image.flatten()
    hist = np.histogram(signal, bins=256)[0]
    probas = hist[hist > 0] / signal.size
    entropy = -np.sum(probas * np.log2(probas), axis=0)
    return entropy


def sample_and_save_dataset(cache_dir, entropy_threshold=ENTROPY_THRESHOLD, coord_range=config.israel_range,
                            sample_num=TRAIN_SAMPLE_NUM, use_existing=True, feature_builder=example_features_builder):
    s = generate_static_maps(config.tile_server_dns_noport, tile_server_ports)
    if not use_existing:
        shutil.rmtree(cache_dir, ignore_errors=True)  # remove old directory
    os.makedirs(cache_dir, exist_ok=True)

    do_files_exist = lambda i: os.path.exists(f"{cache_dir}/{i}_img.pkl") and os.path.exists(f"{cache_dir}/{i}_features.pkl")

    def build_training_example(i):
        if use_existing and do_files_exist(i):
            return
        entropy = 0
        counter = 0
        while entropy < entropy_threshold:
            counter += 1
            coord = sample_coordinate_in_range(*coord_range)
            ext = build_tile_extent(coord, radius_in_meters=HALF_TILE_LENGTH)
            image = render_multi_channel(s, ext)
            entropy = _get_image_entropy(image)

        np.save(f"{cache_dir}/{i}_img.npy", image)  # much smaller in memory

        return coord

    coords = parmap(build_training_example, range(sample_num), use_tqdm=True, desc='Building image dataset')
    # print(coords)
    print("Calculating features:   ", end="", flush=True)
    st = time.time()
    all_coords_feature_vec = feature_builder.extract_coordinates([c for c in coords if c is not None])
    print(f"Calculation took {time.time()-st}")

    skipped = 0
    for i, coord in enumerate(tqdm(coords, total=len(coords), desc="Writing back features")):
        if use_existing and do_files_exist(i):
            skipped += 1
            return
        np.save(f"{cache_dir}/{i}_features.npy", all_coords_feature_vec.iloc[i-skipped].values)


def convert_dataset_to_npy(cache_dir, sample_num=TRAIN_SAMPLE_NUM, **kwargs):
    def convert_pkl_to_npy(i):
        with open(f"{cache_dir}/{i}.pkl", 'rb') as f:
            img, features = pickle.load(f)

        img = img.astype(np.uint8)
        features = features.values
        np.save(f"{cache_dir}/{i}_img.npy", img)  # much smaller in memory
        np.save(f"{cache_dir}/{i}_features.npy", features)  # much smaller in memory

        os.remove(f"{cache_dir}/{i}.pkl")

    parmap(convert_pkl_to_npy, range(sample_num), use_tqdm=True, desc='Converting to npy')



if __name__ == '__main__':
    # multiproc_util.force_serial = True
    sample_and_save_dataset(VAL_CACHE_DIR, sample_num=VAL_SAMPLE_NUM, feature_builder=get_builder(),
                            use_existing=True)
    sample_and_save_dataset(TRAIN_CACHE_DIR, sample_num=50_000, feature_builder=get_builder(),
                            use_existing=True)
