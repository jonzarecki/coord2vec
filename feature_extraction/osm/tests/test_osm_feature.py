import unittest

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt

from feature_extraction.osm.osm_line_feature import OsmLineFeature
from feature_extraction.osm.osm_polygon_feature import OsmPolygonFeature
from feature_extraction.osm.osm_tag_filters import HOSPITAL, RESIDENTIAL_ROAD


class TestOsmFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        near_levinshtein_house = wkt.loads('POINT (34.8576548 32.1869038)')
        hatlalim_rd_raanana = wkt.loads('POINT (34.8583825 32.1874658)')
        cls.gdf = GeoDataFrame(pd.DataFrame({'geom': [near_levinshtein_house]}), geometry='geom')
        cls.hatlalim_gdf = GeoDataFrame(pd.DataFrame({'geom': [hatlalim_rd_raanana]}), geometry='geom')

    def test_beit_lewinstein_hospital_nearest_in_raanana(self):
        nearest_hospital_feat = OsmPolygonFeature(HOSPITAL, 'nearest_neighbour')
        res = nearest_hospital_feat.extract(self.gdf)
        self.assertLess(res.iloc[0], 150)  # the coordinate is very close

    def test_bet_lewinstein_area_for_radius_2km(self):
        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'area_of', max_radius_meter=2 * 1000)
        res = hospital_area_feat.extract(self.gdf)
        # NOT VERIFIED YET
        self.assertAlmostEqual(res.iloc[0], 30257, delta=1)  # area is pretty large

    def test_bet_lewinstein_is_the_only_hospital_for_radius_2km(self):
        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'number_of', max_radius_meter=2 * 1000)
        res = hospital_area_feat.extract(self.gdf)
        self.assertEqual(1, res.iloc[0])

    def test_residential_roads_near_bet_lewinstein(self):
        hospital_area_feat = OsmLineFeature(RESIDENTIAL_ROAD, 'length_of', max_radius_meter=10)
        res = hospital_area_feat.extract(self.hatlalim_gdf)
        self.assertAlmostEqual(res.iloc[0], 434, delta=1)
if __name__ == '__main__':
    unittest.main()
