import unittest

from geopandas import GeoDataFrame
import pandas as pd
from shapely import wkt

from feature_extraction.osm.osm_feature import OsmFeature
from feature_extraction.osm.osm_tag_filters import HOSPITAL


class MyTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        near_levinshtein_house = wkt.loads('POINT(32.1869038 34.8576548)')
        cls.gdf = GeoDataFrame(pd.DataFrame({'geom': [near_levinshtein_house]}), geometry='geom')


    def test_fetches_meir_hospital_in_raanana(self):
        hospital_feat = OsmFeature(HOSPITAL, 'nearest_neighbour')

        conn = hospital_feat.get_postgis_connection()
        hospital_feat.apply_func(self.gdf.geometry.iloc[0], conn=conn)
        hospital_feat.extract(self.gdf)


        # FAILS for now because the hospital is in planet_osm_polygon and not planet_osm_point

        # select tags from planet_osm_polygon
        # where amenity='hospital' and name = 'בית לוינשטיין' limit 20;

        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
