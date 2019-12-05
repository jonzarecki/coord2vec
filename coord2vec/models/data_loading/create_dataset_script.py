import os
import pickle
import shutil

import numpy as np

from coord2vec import config
from coord2vec.common.multiproc_util import parmap
from coord2vec.config import TRAIN_CACHE_DIR, VAL_CACHE_DIR, VAL_SAMPLE_NUM, TRAIN_SAMPLE_NUM, ENTROPY_THRESHOLD, \
    HALF_TILE_LENGTH, tile_server_ports
from coord2vec.feature_extraction.features_builders import example_features_builder, house_price_builder, \
    only_build_distance_builder
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

    def build_training_example(i):
        if use_existing and os.path.exists(f"{cache_dir}/{i}.pkl"):
            return
        entropy = 0
        counter = 0
        while entropy < entropy_threshold:
            counter += 1
            coord = sample_coordinate_in_range(*coord_range)
            ext = build_tile_extent(coord, radius_in_meters=HALF_TILE_LENGTH)
            image = render_multi_channel(s, ext)
            entropy = _get_image_entropy(image)

        feature_vec = feature_builder.extract_coordinates([coord])

        with open(f"{cache_dir}/{i}.pkl", 'wb') as f:
            pickle.dump((image, feature_vec), f)

    parmap(build_training_example, range(0, sample_num), use_tqdm=True, desc='building_dataset', nprocs=1)


if __name__ == '__main__':
    sample_and_save_dataset(VAL_CACHE_DIR, sample_num=VAL_SAMPLE_NUM, feature_builder=house_price_builder,
                            use_existing=True)
    sample_and_save_dataset(TRAIN_CACHE_DIR, sample_num=TRAIN_SAMPLE_NUM, feature_builder=house_price_builder,
                            use_existing=True)
