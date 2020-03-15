from shapely.geometry.base import BaseGeometry

from coord2vec.feature_extraction.ors.base_pgr_feature import BasePGRFeature


class PGRRoute(BasePGRFeature):
    def __init__(self, object_filter: str, table: str, object_name: str = None,
                 transportation_type='driving-car', **kwargs):
        table_filter_dict = {table: {object_name: object_filter}}
        feature_name = f"nearest_{object_name}"
        self.object_name = object_name
        self.table = table
        """
        Args:
            object_filter: An object postgis filter from "coord2vec/feature_extraction/osm/osm_tag_filters.py"
            table: name of postgis table to extract object from
            object_name: name of the object, used just for the feature name
            transportation_type: whether to intersect a real circle around the objects, or just fast within function
        """
        super().__init__(transportation_type=transportation_type,
                         table_filter_dict=table_filter_dict,
                         feature_names=[f"{transportation_type}_distance_to_{feature_name}",
                                        f"{transportation_type}_time_to_{feature_name}"], **kwargs)

    def _build_postgres_query(self):
        intersection_table = self.intersect_tbl_name_dict[self.table]
        
        raise AssertionError("Update query from ors")
        query = f"""
                     SELECT geom_id,
                            ST_asText(q_geom) AS source_point,
                            ST_asText(closest) AS dest_point
                     FROM (SELECT ST_ClosestPoint(closest.t_geom ::geometry, q.q_geom ::geometry) AS closest, geom_id, q_geom
                           FROM (SELECT DISTINCT geom_id, q_geom FROM  {intersection_table} WHERE {self.object_name} = 1) as q
                                     CROSS JOIN LATERAL
                                          (SELECT t_geom
                                           FROM  {intersection_table} t
                                           WHERE {self.object_name} = 1
                                           ORDER BY q.q_geom ::geometry <#> t.t_geom ::geometry
                                           LIMIT 1) AS closest) r;
                  """

        return query

    def set_default_value(self, radius):
        if self.transportation_type == 'foot-walking':
            self.default_value = {self.feature_names[0]: radius,
                                  self.feature_names[1]: ((radius / 1000) / 4) * 60}
        elif self.transportation_type == 'driving-car':
            self.default_value = {self.feature_names[0]: radius,
                                  self.feature_names[1]: ((radius / 1000) / 50) * 60}
        # duration unit: minutes
        # /1000 - to convert from meters to km
        # /4 - to calculate the walking time assuming 4 km/h is the average walking speed
        # /50 - to calculate the driving time assuming 50 km/h is the average driving speed
        # *60 - to convert hours to minutes
