from shapely.geometry.base import BaseGeometry
import geopandas as gpd
from coord2vec.feature_extraction.osm.base_postgres_feature import BasePostgresFeature


class Heights(BasePostgresFeature):
    def __init__(self, object_filter: str, table: str, object_name: str = None, **kwargs):
        table_filter_dict = {table: {object_name: object_filter}}
        self.agg_functions = ['avg', 'min', 'max', 'stddev']
        feature_names = [f"height_of_{object_name}_{agg}" for agg in self.agg_functions]
        feature_names += [f"absolute_height_of_{object_name}_{agg}" for agg in self.agg_functions]
        self.object_name = object_name
        self.table = table
        super().__init__(table_filter_dict=table_filter_dict, feature_names=feature_names, **kwargs)

    def _build_postgres_query(self):
        intersection_table = self.intersect_tbl_name_dict[self.table]
        height_agg_sql = ' , '.join(f'{agg}(height) as {agg}' for agg in self.agg_functions)
        abs_height_agg_sql = ' , '.join(f'{agg}(absolute_height) as {agg}' for agg in self.agg_functions)
        query = f"""
                 SELECT geom_id, {height_agg_sql}, {abs_height_agg_sql}
                 FROM {intersection_table} f 
                 where {self.object_name} = 1
                 GROUP BY geom_id
                """

        return query

    def set_default_value(self, radius):
        self.default_value = 0
