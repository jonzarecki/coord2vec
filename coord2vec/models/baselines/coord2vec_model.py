from sklearn.base import BaseEstimator

from coord2vec import config
from coord2vec.image_extraction.tile_image import generate_static_maps, render_multi_channel
from coord2vec.image_extraction.tile_utils import build_tile_extent
from coord2vec.models.data_loading.create_dataset_script import save_sampled_dataset
from coord2vec.models.model_utils import get_data_loader


class Coord2Vec(BaseEstimator):
    """
    Wrapper for the coord2vec algorithm
    """
    def __init__(self):
        pass

    def fit(self, cache_dir, sample=False, entropy_threshold=0.1, coord_range=config.israel_range, sample_num=50000):

        # get dataset
        if sample:
            save_sampled_dataset(cache_dir, entropy_threshold=entropy_threshold, coord_range=coord_range, sample_num=sample_num)

        #################### brus ######################
        data_loader = get_data_loader(cache_dir)

        ################################################

    def load_trained_model(self):
        #################### brus ######################

        ################################################
        pass

    def predict(self, coords):
        s = generate_static_maps(config.tile_server_dns_noport, [8080, 8081])

        images = []
        for coord in coords:
            ext = build_tile_extent(coord, radius_in_meters=50)
            image = render_multi_channel(s, ext)
            images.append(image)

        embeddings = []
        #################### brus ######################

        ################################################
        return embeddings
