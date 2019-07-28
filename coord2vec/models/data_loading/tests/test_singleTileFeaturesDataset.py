import os
import shutil
from unittest import TestCase

from coord2vec.config import TEST_CACHE_DIR, TENSORBOARD_DIR
from coord2vec.models.data_loading.create_dataset_script import sample_and_save_dataset
from coord2vec.models.data_loading.tile_features_loader import SingleTileFeaturesDataset


class TestSingleTileFeaturesDataset(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tb_dir = 'test'

    def test___getitem__(self):
        sample_and_save_dataset(TEST_CACHE_DIR, sample_num=7, use_existing=False)
        dataset = SingleTileFeaturesDataset(TEST_CACHE_DIR, feature_index=2)
        self.assertTupleEqual(dataset[0][1].shape, (1,))

    @classmethod
    def tearDownClass(cls) -> None:
        # shutil.rmtree(os.path.join(TENSORBOARD_DIR, cls.tb_dir))
        shutil.rmtree(TEST_CACHE_DIR)
