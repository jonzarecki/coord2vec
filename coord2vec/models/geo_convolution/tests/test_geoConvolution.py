from unittest import TestCase
import numpy as np

from coord2vec.models.geo_convolution.geo_convolution import GeoConvolution


class TestGeoConvolution(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GeoConvolution()

    def test_tiles2image2tiles(self):
        image = np.ones((10, 10, 3))
        tile_size = 5
        tiles = self.model._image2tiles(image, tile_size)
        self.assertTupleEqual(tiles.shape, (2, 2, 5, 5, 3))
        new_image = self.model._tiles2image(tiles)
        self.assertTrue(np.array_equal(image, new_image))
