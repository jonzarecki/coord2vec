import math
import unittest

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt
from unittest.mock import patch

from coord2vec.feature_extraction.postgres_feature import PostgresFeature
from coord2vec.feature_extraction.osm import OsmLineFeature
from coord2vec.feature_extraction.osm.osm_polygon_feature import OsmPolygonFeature
from coord2vec.feature_extraction.osm.osm_tag_filters import HOSPITAL, RESIDENTIAL_ROAD, BUILDING


class TestOsmFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        near_levinshtein_house = wkt.loads('POINT (34.8576548 32.1869038)')
        hatlalim_rd_raanana = wkt.loads('POINT (34.8583825 32.1874658)')
        nowhere = wkt.loads('POINT (10.0 10.0)')
        seattle = wkt.loads('POINT (47.595915 -122.310114)')

        cls.gdf = GeoDataFrame(pd.DataFrame({'geom': [near_levinshtein_house, hatlalim_rd_raanana]}), geometry='geom')
        cls.hatlalim_gdf = GeoDataFrame(pd.DataFrame({'geom': [hatlalim_rd_raanana]}), geometry='geom')
        cls.nowhere_gdf = GeoDataFrame(pd.DataFrame({'geom': [nowhere]}), geometry='geom')
        cls.seattle_gdf = GeoDataFrame(pd.DataFrame({'geom': [seattle]}), geometry='geom')

        # check if Israel osm docker is up
        cls.israel_osm = cls.is_osm_up(cls.gdf)

        cls.beijing_gdf = GeoDataFrame(pd.DataFrame({'geom': [wkt.loads('POINT (39.9207 116.3976)')]}), geometry='geom')
        cls.china_osm = cls.is_osm_up(cls.beijing_gdf)

        cls.manhattan_gdf = GeoDataFrame(pd.DataFrame({'geom': [wkt.loads('POINT (40.7612 -73.9826)')]}), geometry='geom')
        cls.na_osm = cls.is_osm_up(cls.manhattan_gdf)

        assert any([cls.israel_osm, cls.china_osm, cls.na_osm])

    @staticmethod
    def is_osm_up(gdf):
        building_area_feat = OsmPolygonFeature(BUILDING, 'area_of', object_name='building', max_radius=2 * 1000)
        res = building_area_feat.extract(gdf)
        return res[building_area_feat.feature_names[0]].iloc[0] > 0

    def test_beit_lewinstein_hospital_nearest_in_raanana(self):
        if not self.israel_osm:
            return
        nearest_hospital_feat = OsmPolygonFeature(HOSPITAL, 'nearest_neighbour', object_name='hospital',
                                                  max_radius=2 * 1000)
        res = nearest_hospital_feat.extract(self.gdf)
        self.assertLess(res.iloc[0][nearest_hospital_feat.all_feature_names[0]], 150)  # the coordinate is very close
        self.assertLess(res.iloc[0][nearest_hospital_feat.all_feature_names[1]], 150 / nearest_hospital_feat.max_radius)

    def test_bet_lewinstein_area_for_radius_2km(self):
        if not self.israel_osm:
            return
        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'area_of', object_name='hospital', max_radius=2 * 1000)
        res = hospital_area_feat.extract(self.gdf)
        # NOT VERIFIED YET
        self.assertAlmostEqual(res[hospital_area_feat.all_feature_names[0]].iloc[0], 2811, delta=1)  # area is pretty large

        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'area_of', object_name='hospital', max_radius=2 * 1000,
                                               normed=True)
        res = hospital_area_feat.extract(self.gdf)
        self.assertAlmostEqual(res[hospital_area_feat.all_feature_names[1]].iloc[0],
                               2811 / (hospital_area_feat.max_radius ** 2 * math.pi), delta=1e-4)

    def test_max_radius_intersection_works(self):
        if not self.israel_osm:
            return
        hospital_area_feat_big = OsmPolygonFeature(HOSPITAL, 'area_of', object_name='hospital', max_radius=120)
        big_radius_area = hospital_area_feat_big.extract(self.gdf)

        hospital_area_feat_small = OsmPolygonFeature(HOSPITAL, 'area_of', object_name='hospital', max_radius=10)
        small_radius_area = hospital_area_feat_small.extract(self.gdf)
        # NOT VERIFIED YET
        self.assertGreater(big_radius_area[hospital_area_feat_big.feature_names[0]].iloc[0],
                           small_radius_area[hospital_area_feat_small.feature_names[0]].iloc[0])

    def test_bet_lewinstein_is_the_only_hospital_for_radius_2km(self):
        if not self.israel_osm:
            return
        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'number_of', object_name='hospital', max_radius=2 * 1000)
        res = hospital_area_feat.extract(self.gdf)
        self.assertEqual(1, res[hospital_area_feat.feature_names[0]].iloc[0])

    def test_residential_roads_length_near_bet_lewinstein_only_tlalim(self):
        if not self.israel_osm:
            return
        hospital_area_feat = OsmLineFeature(RESIDENTIAL_ROAD, 'length_of', object_name='road', max_radius=10)
        res = hospital_area_feat.extract(self.gdf)
        self.assertAlmostEqual(res[hospital_area_feat.feature_names[0]].iloc[1], 9, delta=1)

    def test_residential_roads_number_near_bet_lewinstein_only_tlalim(self):
        if not self.israel_osm:
            return
        hospital_area_feat = OsmLineFeature(RESIDENTIAL_ROAD, 'number_of', object_name='road', max_radius=10)
        res = hospital_area_feat.extract(self.gdf)
        self.assertEqual(res[hospital_area_feat.feature_names[0]].iloc[1], 1)

    ########## extreme test cases ##########

    def test_nowhere_returns_very_far_hospital(self):
        nearest_hospital_feat = OsmPolygonFeature(HOSPITAL, 'nearest_neighbour', object_name='hospital',
                                                  max_radius=1000)
        res = nearest_hospital_feat.extract(self.nowhere_gdf)  # the coordinate is very far from everything
        self.assertEqual(res[nearest_hospital_feat.feature_names[0]].iloc[0], 1000)

    def test_nowhere_has_zero_hospital_area(self):
        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'area_of', object_name='hospital', max_radius=2 * 1000)
        res = hospital_area_feat.extract(self.nowhere_gdf)
        self.assertEqual(res[hospital_area_feat.feature_names[0]].iloc[0], 0)

    def test_nowhere_returns_zero_hospitals_number_of(self):
        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'number_of', object_name='hospital', max_radius=2 * 1000)
        res = hospital_area_feat.extract(self.nowhere_gdf)
        self.assertEqual(res[hospital_area_feat.feature_names[0]].iloc[0], 0)

    def test_nowhere_returns_zero_road_length(self):
        hospital_area_feat = OsmLineFeature(RESIDENTIAL_ROAD, 'length_of', object_name='hospital', max_radius=10)
        res = hospital_area_feat.extract(self.nowhere_gdf)
        self.assertEqual(0, res[hospital_area_feat.feature_names[0]].iloc[0])

    ## Beijing tests
    def test_beijing_buildings_area(self):
        if not self.china_osm:
            return
        building_area_feat = OsmPolygonFeature(BUILDING, 'area_of', object_name='building', max_radius=2 * 1000)
        res = building_area_feat.extract(self.beijing_gdf)
        self.assertGreater(res[building_area_feat.feature_names[0]].iloc[0], 0)

    ## NA tests
    def test_na_buildings_area(self):
        if not self.na_osm:
            return
        building_area_feat = OsmPolygonFeature(BUILDING, 'area_of', object_name='building', max_radius=2 * 1000)
        res = building_area_feat.extract(self.beijing_gdf)
        self.assertGreater(res[building_area_feat.feature_names[0]].iloc[0], 0)


# @patch('coord2vec.feature_extraction.osm.OsmLineFeature.extract', return_value='pumpkins')
# @patch.multiple(Feature, __abstractmethods__=set())
# def test_extract_single_coord_calls_gdf(self):
#     hospital_area_feat = Feature(RESIDENTIAL_ROAD, 'number_of', max_radius=10)
#     res = hospital_area_feat.extract_single_coord((1,2))
#     self.assertEqual(res, 'pumpkins')


if __name__ == '__main__':
    unittest.main()
