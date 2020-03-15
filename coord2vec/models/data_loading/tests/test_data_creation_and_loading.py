import pickle
import shutil
import unittest

import torch

from coord2vec.config import TEST_CACHE_DIR, IMG_WIDTH, IMG_HEIGHT, tile_server_ports
from coord2vec.models.data_loading.create_dataset_script import sample_and_save_dataset
from coord2vec.models.data_loading.tile_features_loader import get_files_from_path, TileFeaturesDataset


class TestDataCreation(unittest.TestCase):
    def test_script_creates_correct_number_of_samples(self):
        self.skipTest("Tiles not relevant at the moment")
        self.fail("hangs for a long time")
        sample_and_save_dataset(TEST_CACHE_DIR, sample_num=7, use_existing=False)
        for p in get_files_from_path(TEST_CACHE_DIR):
            with open(p, 'rb') as f:
                image, feats = pickle.load(f)
                self._check_pkl_ok(feats, image)

    def _check_pkl_ok(self, feats, image):
        if isinstance(feats, torch.Tensor):
            self.assertFalse(any(torch.isnan(feats)), f"{feats}")
        else:
            self.assertFalse(feats.isna().values.any(), f"{feats}")
        self.assertTupleEqual((len(tile_server_ports), IMG_WIDTH, IMG_HEIGHT), image.shape)

    def test_no_nones_in_dataset(self):
        self.skipTest("Tiles not relevant at the moment")
        self.fail("hangs for a long time")
        sample_and_save_dataset(TEST_CACHE_DIR, sample_num=3, use_existing=False)
        ds = TileFeaturesDataset(TEST_CACHE_DIR)

        for i in range(len(ds)):
            im, feats = ds[i]
            self._check_pkl_ok(feats, im)

    @classmethod
    def tearDownClass(cls) -> None:
        pass
        # shutil.rmtree(TEST_CACHE_DIR)

if __name__ == '__main__':
    unittest.main()
