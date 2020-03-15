from shapely.geometry.base import BaseGeometry
import geopandas as gpd
from coord2vec.feature_extraction.osm.base_postgres_feature import BasePostgresFeature


class AreaOf(BasePostgresFeature):
    def __init__(self, object_filter: str, table: str, object_name: str = None, **kwargs):
        table_filter_dict = {table: {object_name: object_filter}}
        feature_name = f"area_of_{object_name}"
        self.object_name = object_name
        self.table = table
        super().__init__(table_filter_dict=table_filter_dict, feature_names=[feature_name], **kwargs)

    def _build_postgres_query(self):
        intersection_table = self.intersect_tbl_name_dict[self.table]
        # TODO: doc magic number
        query = f"""
                SELECT geom_id, SUM(COALESCE(ST_Area(f.t_geom, TRUE), 0.) * coverage) / 10.764 AS total_area
                FROM {intersection_table} f 
                where {self.object_name} = 1
                GROUP BY geom_id
                """
        return query

    def set_default_value(self, radius):
        self.default_value = {self.feature_names[0]: 0}
