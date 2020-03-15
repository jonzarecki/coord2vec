from shapely.geometry.base import BaseGeometry
import geopandas as gpd
from coord2vec.feature_extraction.osm.base_postgres_feature import BasePostgresFeature


class DistanceTo(BasePostgresFeature):
    def __init__(self, object_filter: str, table: str, object_name: str = None, **kwargs):
        table_filter_dict = {table: {object_name: object_filter}}
        feature_name = f"distance_to_{object_name}"
        self.object_name = object_name
        self.table = table
        super().__init__(table_filter_dict=table_filter_dict, feature_names=[feature_name], **kwargs)

    def _build_postgres_query(self):
        intersection_table = self.intersect_tbl_name_dict[self.table]
        query = f"""
                SELECT geom_id, dist
                FROM (SELECT geom_id,
                             ST_Distance(q_geom, t_geom) dist,
                             row_number() OVER (PARTITION BY geom_id ORDER BY ST_Distance(q_geom, t_geom) ASC, t_geom) r
                      FROM {intersection_table}
                      Where {self.object_name} = 1
                      and not st_asText(q_geom) =  st_asText(t_geom)) s
                WHERE r = 1
                """

        return query

    def set_default_value(self, radius):
        self.default_value = radius
