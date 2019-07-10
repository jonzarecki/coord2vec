import os
import pickle
import numpy as np
from tqdm import tqdm

from coord2vec import config
from coord2vec.config import CACHE_DIR, SAMPLE_NUM
from coord2vec.feature_extraction.features_builders import example_features_builder
from coord2vec.feature_extraction.osm import OsmPolygonFeature
from coord2vec.feature_extraction.osm.osm_tag_filters import BUILDING
from coord2vec.image_extraction.tile_utils import sample_coordinate_in_range, build_tile_extent
from coord2vec.image_extraction.tile_image import render_multi_channel, generate_static_maps

ENTROPY_THRESHOLD = 0.1


def feature_extractor(coord) -> np.array:
    # placeholder for building more complicated features
    building_count_feat = OsmPolygonFeature(BUILDING, 'number_of', max_radius_meter=50)
    res = building_count_feat.extract_single_coord(coord)
    return np.array([res])


israel_range = [29.593, 34.085, 32.857, 34.958]

if __name__ == '__main__':
    os.makedirs(CACHE_DIR, exist_ok=True)
    sampled_coords = {}
    s = generate_static_maps(config.tile_server_dns_noport, [8080, 8081])
    for i in tqdm(range(SAMPLE_NUM)):

        entropy, counter = 0, 0
        while entropy < ENTROPY_THRESHOLD:
            coord = sample_coordinate_in_range(*israel_range)
            ext = build_tile_extent(coord, radius_in_meters=50)

            image = render_multi_channel(s, ext)

            signal = image.flatten() / image.sum()
            entropy = -np.sum(signal * np.log2(signal), axis=0)

            counter += 1
            assert counter <= 5

        features = example_features_builder.extract_coordinate(coord)

        with open(f"{CACHE_DIR}/{i}.pkl", 'wb') as f:
            pickle.dump((image, features), f)
