from unittest import TestCase

from geopandas import GeoDataFrame
from shapely import wkt

from coord2vec.feature_extraction.features_builders import baseline_builder
import pandas as pd


class TestFeaturesBuilder(TestCase):

    @classmethod
    def setUpClass(cls):
        near_levinshtein_house = wkt.loads('POINT (34.8576548 32.1869038)')
        hatlalim_rd_raanana = wkt.loads('POINT (34.8583825 32.1874658)')
        cls.gdf = GeoDataFrame(pd.DataFrame({'geom': [near_levinshtein_house, hatlalim_rd_raanana]}), geometry='geom')
        # cls.hatlalim_gdf = GeoDataFrame(pd.DataFrame({'geom': [hatlalim_rd_raanana]}), geometry='geom')

    def test_extract(self):
        results = baseline_builder.extract(self.gdf)
        self.assertEqual(len(results), len(baseline_builder.features))
