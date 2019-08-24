import os
import shutil
from unittest import TestCase
from coord2vec.config import TEST_CACHE_DIR, TENSORBOARD_DIR, VAL_CACHE_DIR, washington_range
from coord2vec.models.baselines import Coord2Vec
from coord2vec.models.data_loading.create_dataset_script import sample_and_save_dataset
from coord2vec.feature_extraction.features_builders import example_features_builder, house_price_builder
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset


class TestCoord2Vec(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.embedding_dim = 16
        cls.tb_dir = 'test'
        cls.coord2vec = Coord2Vec(example_features_builder, n_channels=3, embedding_dim=cls.embedding_dim,
                                  tb_dir=cls.tb_dir)

    def test_fit_predict(self):
        # test fit
        sample_and_save_dataset(TEST_CACHE_DIR, sample_num=7, use_existing=False, entropy_threshold=0.2,
                                coord_range=washington_range)
        train_dataset = TileFeaturesDataset(TEST_CACHE_DIR)
        val_dataset = TileFeaturesDataset(TEST_CACHE_DIR)

        self.coord2vec.fit(train_dataset, val_dataset, epochs=2, batch_size=4, evaluate_every=2)

        # test predict
        coord_pred = [(34.8576548, 32.1869038), (34.8583825, 32.1874658)]
        embeddings = self.coord2vec.predict(coord_pred)
        self.assertTupleEqual(embeddings.shape, (len(coord_pred), self.embedding_dim))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(os.path.join(TENSORBOARD_DIR, cls.tb_dir))
        shutil.rmtree(TEST_CACHE_DIR)
