from unittest import TestCase

from geopandas import GeoDataFrame, GeoSeries
from shapely import wkt
from shapely.geometry import Point

from coord2vec.config import BUILDINGS_FEATURES_TABLE
from coord2vec.feature_extraction.feature_bundles import karka_bundle_features, create_building_features
import pandas as pd

from coord2vec.feature_extraction.features_builders import FeaturesBuilder


class TestFeaturesBuilder(TestCase):

    @classmethod
    def setUpClass(cls):
        feats = create_building_features(karka_bundle_features)
        cls.builder = FeaturesBuilder(feats, cache_table=BUILDINGS_FEATURES_TABLE)
        near_levinshtein_house = wkt.loads('POINT (34.8576548 32.1869038)')
        hatlalim_rd_raanana = wkt.loads('POINT (34.8583825 32.1874658)')
        cls.gdf = GeoDataFrame(pd.DataFrame({'geom': [near_levinshtein_house, hatlalim_rd_raanana]}), geometry='geom')
        # cls.hatlalim_gdf = GeoDataFrame(pd.DataFrame({'geom': [hatlalim_rd_raanana]}), geometry='geom')

    def test_extract(self):
        results = self.builder.transform(self.gdf.geometry)
        self.assertEqual(results.shape[0], self.gdf.shape[0])
        self.assertEqual(results.shape[1], len(self.builder.all_feat_names))

    def test_extract_fails_on_duplicates(self):
        points = [(116.475489, 40.01952),
                  (116.45391699999999, 39.881534),
                  (116.561978, 39.877145),
                  (116.43801, 40.076114000000004),
                  (116.42839199999999, 39.886229),
                  (116.46628, 39.991363),
                  (116.48258600000001, 39.891991),
                  (116.45391699999999, 39.881534)]
        gs = GeoSeries([Point(*c) for c in points])
        with self.assertRaises(AssertionError):
            results = self.builder.transform(gs)

    # def test_extract_coordinates(self):
    #     results = self.builder.transform([(34.8576548, 32.1869038)])
    #     self.assertEqual(results.shape[0], 1)
    #     self.assertEqual(results.shape[1], len(self.builder.features_names))
    #
    # def test_extract_multiple_coordinates(self):
    #     results = self.builder.extract_coordinates([(34.8576548, 32.1869038), (34.8576548, 32.1869038)])
    #     self.assertEqual(results.shape[0], 2)
    #     self.assertEqual(results.shape[1], len(self.builder.features_names))
