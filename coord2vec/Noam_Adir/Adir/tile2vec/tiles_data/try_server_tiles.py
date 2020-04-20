import unittest
import numpy as np
from staticmap import StaticMap

from coord2vec.config import IMG_WIDTH, IMG_HEIGHT, LOC2VEC_URL_TEMPLATE, TILE_SERVER_PORTS
from coord2vec.image_extraction.tile_image import render_single_tile, generate_static_maps, render_multi_channel
from coord2vec.image_extraction.tile_utils import build_tile_extent

class TestTileImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 'http://a.tile.openstreetmap.us/usgs_large_scale/{z}/{x}/{y}.jpg'
        cls.m = StaticMap(IMG_WIDTH, IMG_HEIGHT, url_template='http://40.127.166.177:8103/tile/{z}/{x}/{y}.png',
                      delay_between_retries=15, tile_request_timeout=5)
        # cls.center = list(reversed([40.750096, -73.984929]))
        cls.center = [40.720096, -74.000000]
        # cls.center = [40.777665, -74.02085500000001]
        # cls.center = [40.710383, -74.06526600]



        # cls.s = generate_static_maps(LOC2VEC_URL_TEMPLATE, [80])
        cls.ext = build_tile_extent(cls.center, radius_in_meters=50)

    def test_rendering_single_image_works(self):
        image = np.array(render_single_tile(self.m, self.ext))
        # self.assertTupleEqual((IMG_HEIGHT, IMG_WIDTH, 3), image.shape)
        import matplotlib.pyplot as plt
        print(image.shape)
        plt.imshow(image)
        plt.show()