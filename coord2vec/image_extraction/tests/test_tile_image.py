import unittest
import numpy as np
from staticmap import StaticMap

from coord2vec.config import IMG_WIDTH, IMG_HEIGHT, tile_server_dns_noport, tile_server_ports
from coord2vec.image_extraction.tile_image import render_single_tile, generate_static_maps, render_multi_channel
from coord2vec.image_extraction.tile_utils import build_tile_extent


class TestTileImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = StaticMap(IMG_WIDTH, IMG_HEIGHT, url_template=tile_server_dns_noport.replace('{p}', '8080'))
        cls.center = [34.7855, 32.1070]
        cls.s = generate_static_maps(tile_server_dns_noport, tile_server_ports)
        cls.ext = build_tile_extent(cls.center, radius_in_meters=50)

    def test_rendering_single_image_works(self):
        image = np.array(render_single_tile(self.m, self.ext))
        self.assertTupleEqual((IMG_HEIGHT, IMG_WIDTH, 3), image.shape)

    def test_rendering_multi_channel_image_works(self):
        image = render_multi_channel(self.s, self.ext)
        self.assertTupleEqual((3, IMG_HEIGHT, IMG_WIDTH), image.shape)

    def test_multi_channel_layers_are_just_rgb_converted_to_greyscale(self):
        image_single = render_single_tile(self.m, self.ext)
        image_multi = render_multi_channel(self.s, self.ext)

        self.assertTrue(np.array_equal(np.array(image_single.convert('L')), image_multi[0, :, :]))

if __name__ == '__main__':
    unittest.main()
