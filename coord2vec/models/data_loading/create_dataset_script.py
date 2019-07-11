import os
import pickle
import numpy as np
from tqdm import tqdm

from coord2vec import config
from coord2vec.common.multiproc_util import parmap
from coord2vec.config import CACHE_DIR, SAMPLE_NUM, ENTROPY_THRESHOLD
from coord2vec.feature_extraction.features_builders import example_features_builder, house_price_builder
from coord2vec.feature_extraction.osm import OsmPolygonFeature
from coord2vec.feature_extraction.osm.osm_tag_filters import BUILDING
from coord2vec.image_extraction.tile_utils import sample_coordinate_in_range, build_tile_extent
from coord2vec.image_extraction.tile_image import render_multi_channel, generate_static_maps


def feature_extractor(coord) -> np.array:
    # placeholder for building more complicated features
    building_count_feat = OsmPolygonFeature(BUILDING, 'number_of', max_radius_meter=50)
    res = building_count_feat.extract_single_coord(coord)
    return np.array([res])


def sample_and_save_dataset(cache_dir, entropy_threshold=ENTROPY_THRESHOLD, coord_range=config.israel_range,
                            sample_num=SAMPLE_NUM):
    s = generate_static_maps(config.tile_server_dns_noport, [8080, 8081])
    os.makedirs(cache_dir, exist_ok=True)

    def foo(i):
        entropy = 0
        while entropy < entropy_threshold:
            coord = sample_coordinate_in_range(*coord_range)
            ext = build_tile_extent(coord, radius_in_meters=50)

            image = render_multi_channel(s, ext)

            signal = image.flatten()
            hist = np.histogram(signal, bins=256)[0]
            probas = hist[hist > 0] / signal.size
            entropy = -np.sum(probas * np.log2(probas), axis=0)

        feature_vec = house_price_builder.extract_coordinates([coord])

        with open(f"{cache_dir}/{i}.pkl", 'wb') as f:
            pickle.dump((image, feature_vec), f)

    parmap(foo, range(sample_num))


if __name__ == '__main__':
    sample_and_save_dataset(CACHE_DIR)
