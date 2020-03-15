import os
import shutil
from unittest import TestCase
from coord2vec.config import TEST_CACHE_DIR, TENSORBOARD_DIR, VAL_CACHE_DIR
from coord2vec.models.baselines import Coord2Vec
from coord2vec.models.data_loading.create_dataset_script import sample_and_save_dataset
# from coord2vec.feature_extraction.features_builders import example_features_builder, house_price_builder
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset
from coord2vec.models.losses import ScaledLoss


class TestCoord2Vec(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        pass
        # cls.embedding_dim = 16
        # cls.tb_dir = 'test'
        # losses = [ScaledLoss() for l in range(9)]
        # cls.coord2vec = Coord2Vec(house_price_builder, n_channels=3, embedding_dim=cls.embedding_dim,
        #                           tb_dir=cls.tb_dir, losses=losses)

    @classmethod
    def tearDownClass(cls) -> None:
        pass
        # shutil.rmtree(os.path.join(TENSORBOARD_DIR, cls.tb_dir))
        # shutil.rmtree(TEST_CACHE_DIR)
