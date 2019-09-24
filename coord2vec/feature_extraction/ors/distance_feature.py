import openrouteservice
from geopandas import GeoDataFrame
import pandas as pd

from coord2vec.common.db.postgres import connect_to_db, get_sqlalchemy_engine, save_gdf_to_temp_table_postgres, get_df
from coord2vec.feature_extraction.ors.ors_feature import OrsFeature
from coord2vec.feature_extraction.postgres_feature import geo2sql


class DistanceTimeFeature(OrsFeature):
    """
    This class filters objects from the ors service.
    It gives nice methods for calling ors functions.
    """
    def __init__(self, filter_tag: str, filter_table: str, transportation_type='waling', max_distance_meters=1000, **kwargs):
        super().__init__(**kwargs)
        self.max_distance_meters = max_distance_meters
        self.filter_table = filter_table
        self.filter_tag = filter_tag
        self.transportation_type = transportation_type

    def extract(self, gdf: GeoDataFrame) -> pd.DataFrame:
        eng = get_sqlalchemy_engine()
        tbl_name = save_gdf_to_temp_table_postgres(gdf, eng)

        nearest_road_query = f"""
        with q_geoms as (SELECT * FROM {tbl_name})
        
        SELECT ST_ClosestPoint(t.geom, q_geoms.geom)
            FROM (SELECT way as geom FROM {self.filter_table} WHERE {self.filter_tag}) t JOIN {tbl_name} q_geoms
                    ON ST_DWithin(t.geom, geom.geom, {self.max_radius_meter}, true)
        """

        conn = connect_to_db()
        res = get_df(nearest_road_query, conn, dispose_conn=True)

        eng.execute(f"DROP TABLE {tbl_name}")
        eng.dispose()


        routes = self.client.directions(coords, instructions=False, geometry=False)

