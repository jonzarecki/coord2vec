from coord2vec.feature_extraction.osm.base_postgres_feature import BasePostgresFeature


class ObjectDistances(BasePostgresFeature):
    def __init__(self, object_filter: str, table: str, object_name: str = None, **kwargs):
        table_filter_dict = {table: {object_name: object_filter}}
        self.agg_functions = ['avg', 'min', 'max', 'stddev']
        feature_names = [f"object_distances_{object_name}_{agg}" for agg in self.agg_functions]
        self.object_name = object_name
        self.table = table
        super().__init__(table_filter_dict=table_filter_dict, feature_names=feature_names , **kwargs)

    def _build_postgres_query(self):
        intersection_table = self.intersect_tbl_name_dict[self.table]
        agg_sql = ' , '.join(f'{agg}(dist) as {agg}' for agg in self.agg_functions)
        query = f"""
                with distances as (
                    select a.q_geom as q_geom, a.geom_id as geom_id,
                           a.t_geom, b.t_geom, ST_Distance(a.t_geog, b.t_geog) as dist
                    from {intersection_table} a
                    join {intersection_table} b
                    on st_astext(a.q_geom) = st_astext(b.q_geom)
                    where not a.t_geom = b.t_geom
                    )
                    
                SELECT geom_id, {agg_sql}
                from distances
                group by geom_id
                """

        return query

    def set_default_value(self, radius):
        self.default_value = 0
