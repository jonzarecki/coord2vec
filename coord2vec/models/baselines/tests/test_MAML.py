import os
import shutil
from unittest import TestCase

from coord2vec.models.baselines.maml import MAML
from coord2vec.config import TEST_CACHE_DIR, TENSORBOARD_DIR
from coord2vec.models.data_loading.create_dataset_script import sample_and_save_dataset
from coord2vec.feature_extraction.features_builders import example_features_builder
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset

class TestMAML(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.embedding_dim = 16
        cls.tb_dir = 'test'
        cls.coord2vec = MAML(example_features_builder, n_channels=3, embedding_dim=cls.embedding_dim,
                                  tb_dir=cls.tb_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(os.path.join(TENSORBOARD_DIR, cls.tb_dir), ignore_errors=True)
        shutil.rmtree(TEST_CACHE_DIR, ignore_errors=True)

    def test_fit_predict(self):
        return
        # test fit
        sample_and_save_dataset(TEST_CACHE_DIR, sample_num=7, use_existing=False)
        dataset = TileFeaturesDataset(TEST_CACHE_DIR)
        self.coord2vec.fit(dataset, n_epochs=3, batch_size=1)

        # test predict
        # coord_pred = [(34.8576548, 32.1869038), (34.8583825, 32.1874658)]
        # embeddings = self.coord2vec.predict(coord_pred)
        # self.assertTupleEqual(embeddings.shape, (len(coord_pred), self.embedding_dim))
