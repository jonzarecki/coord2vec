import unittest

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt

from feature_extraction.osm.osm_polygon_feature import OsmPolygonFeature
from feature_extraction.osm.osm_tag_filters import HOSPITAL


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        near_levinshtein_house = wkt.loads('POINT (34.8576548 32.1869038)')
        cls.gdf = GeoDataFrame(pd.DataFrame({'geom': [near_levinshtein_house]}), geometry='geom')

    def test_fetches_beit_lewinstein_hospital_in_raanana(self):
        hospital_feat = OsmPolygonFeature(HOSPITAL, 'nearest_neighbour')
        res = hospital_feat.extract(self.gdf)
        self.assertLess(res.iloc[0], 200)


if __name__ == '__main__':
    unittest.main()
