from unittest import TestCase

from coord2vec.feature_extraction.features_builders import house_price_builder
from coord2vec.models.baselines import Coord2Features


class TestCoord2Features(TestCase):
    def test_predict(self):
        seattle_coords = [(47.5112, -122.257), (47.721, -122.319)]
        coord2features = Coord2Features(feature_builder = house_price_builder)
        self.skipTest("not updated, coord2features does not hold the predict")
        vectors = coord2features.predict(seattle_coords)
        self.assertEqual(2, vectors.shape[0])
