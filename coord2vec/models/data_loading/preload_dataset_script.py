import os
import pickle

from coord2vec import config
from coord2vec.image_extraction.tile_utils import sample_coordinate_in_range, build_tile_extent
from coord2vec.image_extraction.tile_image import render_multi_channel, generate_static_maps


def feature_extractor(coord):
    pass


israel_range = [29.593, 34.085, 32.857, 34.958]
curdir = os.path.dirname(__file__)
cache_dir = os.path.join(curdir, "cache")
if __name__ == '__main__':
    SAMPLE_NUM = 50_000
    sampled_coords = {}
    s = generate_static_maps(config.tile_server_dns_noport, [8080, 8081])
    for i in range(SAMPLE_NUM):
        coord = sample_coordinate_in_range(*israel_range)
        ext = build_tile_extent(coord, radius_in_meters=50)

        image = render_multi_channel(s, ext)

        features = feature_extractor(coord)

        os.mkdir(cache_dir)
        with open(f"{cache_dir}/{i}.pkl") as f:
            pickle.dump((image, features), f)
