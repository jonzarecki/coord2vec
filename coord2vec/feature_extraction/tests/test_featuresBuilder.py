from unittest import TestCase

from geopandas import GeoDataFrame
from shapely import wkt

from coord2vec.feature_extraction.features_builders import example_features_builder, house_price_builder
import pandas as pd


class TestFeaturesBuilder(TestCase):

    @classmethod
    def setUpClass(cls):
        near_levinshtein_house = wkt.loads('POINT (34.8576548 32.1869038)')
        hatlalim_rd_raanana = wkt.loads('POINT (34.8583825 32.1874658)')
        cls.gdf = GeoDataFrame(pd.DataFrame({'geom': [near_levinshtein_house, hatlalim_rd_raanana]}), geometry='geom')
        # cls.hatlalim_gdf = GeoDataFrame(pd.DataFrame({'geom': [hatlalim_rd_raanana]}), geometry='geom')

    def test_extract(self):
        results = example_features_builder.extract(self.gdf, only_relevant=True)
        self.assertEqual(results.shape[0], self.gdf.shape[0])
        self.assertEqual(results.shape[1], len(example_features_builder.features_names))

    def test_extract_coordinates(self):
        results = house_price_builder.extract_coordinates([(34.8576548, 32.1869038)], only_relevant=True)
        self.assertEqual(results.shape[0], 1)
        self.assertEqual(results.shape[1], len(house_price_builder.features_names))

    def test_extract_multiple_coordinates(self):
        results = house_price_builder.extract_coordinates([(34.8576548, 32.1869038), (34.8576548, 32.1869038)],
                                                          only_relevant=True)
        self.assertEqual(results.shape[0], 2)
        self.assertEqual(results.shape[1], len(house_price_builder.features_names))
