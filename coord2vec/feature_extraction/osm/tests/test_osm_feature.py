import unittest

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt
from unittest.mock import patch

from coord2vec.feature_extraction.feature import Feature
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
        cls.israel_osm = cls.is_israel_up()

        # beijing class variables
        beijing_building = wkt.loads('POINT (39.9207 116.39766)')
        cls.beijing_gdf = GeoDataFrame(pd.DataFrame({'geom': [beijing_building]}), geometry='geom')

    @classmethod
    def is_israel_up(cls):
        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'number_of', max_radius_meter=2 * 1000)
        res = hospital_area_feat.extract(cls.gdf)
        return res.iloc[0] > 0

    def test_beit_lewinstein_hospital_nearest_in_raanana(self):
        if not self.israel_osm:
            return
        nearest_hospital_feat = OsmPolygonFeature(HOSPITAL, 'nearest_neighbour', max_radius_meter=2 * 1000)
        res = nearest_hospital_feat.extract(self.gdf)
        self.assertLess(res.iloc[0], 150)  # the coordinate is very close

    def test_bet_lewinstein_area_for_radius_2km(self):
        if not self.israel_osm:
            return
        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'area_of', max_radius_meter=2 * 1000)
        res = hospital_area_feat.extract(self.gdf)
        # NOT VERIFIED YET
        self.assertAlmostEqual(res.iloc[0], 2811, delta=1)  # area is pretty large

    def test_max_radius_intersection_works(self):
        if not self.israel_osm:
            return
        hospital_area_feat_big = OsmPolygonFeature(HOSPITAL, 'area_of', max_radius_meter=120)
        big_radius_area = hospital_area_feat_big.extract(self.gdf)

        hospital_area_feat_small = OsmPolygonFeature(HOSPITAL, 'area_of', max_radius_meter=100)
        small_radius_area = hospital_area_feat_small.extract(self.gdf)
        # NOT VERIFIED YET
        self.assertGreater(big_radius_area.iloc[0], small_radius_area.iloc[0])

    def test_bet_lewinstein_is_the_only_hospital_for_radius_2km(self):
        if not self.israel_osm:
            return
        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'number_of', max_radius_meter=2 * 1000)
        res = hospital_area_feat.extract(self.gdf)
        self.assertEqual(1, res.iloc[0])

    def test_residential_roads_length_near_bet_lewinstein_only_tlalim(self):
        if not self.israel_osm:
            return
        hospital_area_feat = OsmLineFeature(RESIDENTIAL_ROAD, 'length_of', max_radius_meter=10)
        res = hospital_area_feat.extract(self.gdf)
        self.assertAlmostEqual(res.iloc[1], 9, delta=1)

    def test_residential_roads_number_near_bet_lewinstein_only_tlalim(self):
        if not self.israel_osm:
            return
        hospital_area_feat = OsmLineFeature(RESIDENTIAL_ROAD, 'number_of', max_radius_meter=10)
        res = hospital_area_feat.extract(self.gdf)
        self.assertEqual(res.iloc[1], 1)

    ########## extreme test cases ##########

    def test_nowhere_returns_very_far_hospital(self):
        nearest_hospital_feat = OsmPolygonFeature(HOSPITAL, 'nearest_neighbour', max_radius_meter=1000)
        res = nearest_hospital_feat.extract(self.nowhere_gdf)
        self.assertGreater(res.iloc[0], 1_000_000)  # the coordinate is very far from everything

    def test_nowhere_has_zero_hospital_area(self):
        # TODO: check why some objects return NULL when applied with ST_Area
        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'area_of', max_radius_meter=2 * 1000)
        res = hospital_area_feat.extract(self.nowhere_gdf)
        self.assertEqual(res.iloc[0], 0)

    def test_nowhere_returns_zero_hospitals_number_of(self):
        hospital_area_feat = OsmPolygonFeature(HOSPITAL, 'number_of', max_radius_meter=2 * 1000)
        res = hospital_area_feat.extract(self.nowhere_gdf)
        self.assertEqual(res.iloc[0], 0)

    def test_nowhere_returns_zero_road_length(self):
        hospital_area_feat = OsmLineFeature(RESIDENTIAL_ROAD, 'length_of', max_radius_meter=10)
        res = hospital_area_feat.extract(self.nowhere_gdf)
        self.assertEqual(0, res.iloc[0])

    ## Beijing tests
    def test_beijing_buildings_area(self):
        if self.israel_osm:
            return
        building_area_feat = OsmPolygonFeature(BUILDING, 'area_of', max_radius_meter=2 * 1000)
        res = building_area_feat.extract(self.beijing_gdf)
        self.assertGreater(res.iloc[0], 0)


# @patch('coord2vec.feature_extraction.osm.OsmLineFeature.extract', return_value='pumpkins')
# @patch.multiple(Feature, __abstractmethods__=set())
# def test_extract_single_coord_calls_gdf(self):
#     hospital_area_feat = Feature(RESIDENTIAL_ROAD, 'number_of', max_radius_meter=10)
#     res = hospital_area_feat.extract_single_coord((1,2))
#     self.assertEqual(res, 'pumpkins')


if __name__ == '__main__':
    unittest.main()
