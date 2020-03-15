from coord2vec.evaluation.tasks.scores_table import SCORES, GEOM, EXPERIMENT, MODEL, TRAIN_HASH
from coord2vec.feature_extraction.osm.base_postgres_feature import BasePostgresFeature
from coord2vec.feature_extraction.osm.osm_tag_filters import BUILDING, OSM_POLYGON_TABLE


class BuildingScores(BasePostgresFeature):
    object_filter = BUILDING
    object_name = 'building'

    def __init__(self, scores_table: str, experiment: str, model_name: str, train_geom_hash: str,
                 geos_table: str = OSM_POLYGON_TABLE, **kwargs):
        table_filter_dict = {geos_table: {self.object_name: self.object_filter}}
        self.agg_functions = ['avg', 'min', 'max', 'stddev']
        feature_names = [f"{self.object_name}_scores_{agg}" for agg in self.agg_functions]
        self.object_name = self.object_name
        self.table = geos_table
        self.scores_table = scores_table
        self.experiment = experiment
        self.model_name = model_name
        self.train_geom_hash = train_geom_hash
        super().__init__(table_filter_dict=table_filter_dict, feature_names=feature_names, **kwargs)

    def _build_postgres_query(self):
        intersection_table = self.intersect_tbl_name_dict[self.table]
        agg_sql = ' , '.join(f'{agg}({SCORES}) as {agg}' for agg in self.agg_functions)
        query = f"""
                SELECT f.geom_id, {agg_sql}
                FROM {intersection_table} f
                join {self.scores_table} s
                on st_astext(f.t_geom) = st_astext(s.{GEOM})
                where {self.object_name} = 1
                and s.{EXPERIMENT} = '{self.experiment}'
                and {MODEL} = '{self.model_name}'
                and {TRAIN_HASH} = '{self.train_geom_hash}'
                GROUP BY geom_id
                """
        return query

    def set_default_value(self, radius):
        self.default_value = {feature: 0 for feature in self.feature_names}
