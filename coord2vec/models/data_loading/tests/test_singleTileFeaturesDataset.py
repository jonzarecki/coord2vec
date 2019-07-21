from unittest import TestCase

from coord2vec.config import TEST_CACHE_DIR
from coord2vec.models.data_loading.tile_features_loader import SingleTileFeaturesDataset


class TestSingleTileFeaturesDataset(TestCase):
    def test___getitem__(self):
        dataset = SingleTileFeaturesDataset(TEST_CACHE_DIR, feature_index=2)
        self.assertTupleEqual(dataset[0][1].shape, (1,))
