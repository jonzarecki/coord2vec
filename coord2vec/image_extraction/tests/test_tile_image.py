import unittest
import numpy as np
from staticmap import StaticMap

from coord2vec.image_extraction.tile_image import render_single_tile, generate_static_maps, render_multi_channel


class TestTileImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = StaticMap(500, 500, url_template='http://52.232.47.43:8080/tile/{z}/{x}/{y}.png')
        cls.center = [34.7855, 32.1070]
        s = generate_static_maps('http://52.232.47.43:{p}/tile/{z}/{x}/{y}.png', [8080, 8081])
        cls.ext = [34.7855, 32.1070, 34.7855 + 0.001, 32.1070 + 0.001]

    def test_rendering_single_image_works(self):
        image = np.array(render_single_tile(self.m, self.ext))
        self.assertTupleEqual((500, 500, 3), image.shape)

    def test_rendering_multi_channel_image_works(self):
        s = generate_static_maps('http://52.232.47.43:{p}/tile/{z}/{x}/{y}.png', [8080, 8081])
        image = render_multi_channel(s, self.ext)
        self.assertTupleEqual((500, 500, 2), image.shape)


if __name__ == '__main__':
    unittest.main()
