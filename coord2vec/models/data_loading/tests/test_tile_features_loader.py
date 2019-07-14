import unittest
import numpy as np

from coord2vec import config
from coord2vec.models.data_loading.create_dataset_script import sample_and_save_dataset
from coord2vec.models.data_loading.tile_features_loader import TileFeaturesDataset


class TestTileFeaturesDataset(unittest.TestCase):
    def test_no_nones_in_dataset(self):
        sample_and_save_dataset(config.TEST_CACHE_DIR, sample_num=3)
        ds = TileFeaturesDataset(config.TEST_CACHE_DIR)

        for i in range(len(ds)):
            im, feats = ds[i]
            self.assertTrue(all(feats != np.nan))


if __name__ == '__main__':
    unittest.main()
