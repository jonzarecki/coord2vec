import unittest
import numpy as np
from staticmap import StaticMap

# from coord2vec.config import IMG_WIDTH, IMG_HEIGHT, LOC2VEC_URL_TEMPLATE, TILE_SERVER_PORTS
from coord2vec.image_extraction.tile_image import render_single_tile, generate_static_maps, render_multi_channel


# from coord2vec.image_extraction.tile_utils import build_tile_extent

class TestTileImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # url = 'http://a.tile.openstreetmap.us/usgs_large_scale/{z}/{x}/{y}.jpg'

        # google osm
        url = 'https://khms1.google.com/kh/v=865?x={x}&y={y}&z={z}'

        # openstreetmap like wanderfall land not working!!
        # url = 'http://tile.openstreetmap.org/{z}/{x}/{y}.png'

        # url = 'http://mt1.google.com/vt/lyrs=h@162000000&hl=en&x={x}&s=&y={y}&z={z}'
        cls.m = StaticMap(700, 700, url_template=url,
                          delay_between_retries=15, tile_request_timeout=5)
        import folium
        folium.LatLngPopup()
        # cls.center = list(reversed([40.750096, -73.984929]))
        # cls.center = [40.720096, -74.000000]
        cls.center = [40.802187, -73.957066]
        # cls.center = [32.0219, 34.7747]
        # cls.center = [32.031245, 34.762359]
        # cls.center = [40.777665, -74.02085500000001]
        # cls.center = [40.710383, -74.06526600]

        # cls.s = generate_static_maps(LOC2VEC_URL_TEMPLATE, [80])
        # cls.ext = build_tile_extent(cls.center, radius_in_meters=50)
        lon, lat = cls.center
        cls.ext = [lon, lat, lon + 0.001, lat + 0.001]

    def test_rendering_single_image_works(self):
        image = np.array(render_single_tile(self.m, self.ext))
        # self.assertTupleEqual((IMG_HEIGHT, IMG_WIDTH, 3), image.shape)
        import matplotlib.pyplot as plt
        print(image.shape)
        plt.imshow(image)
        plt.show()
