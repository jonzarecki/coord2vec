from functools import partial

from shapely.geometry.base import BaseGeometry

from coord2vec.common.db.postgres import connection, get_df
from coord2vec.feature_extraction.postgres_feature import PostgresFeature, geo2sql, LENGTH_OF_line


class LineMixin(PostgresFeature):
    def __init__(self, apply_type: str, **kwargs):
        super().__init__(apply_type, **kwargs)

        line_func = {
            LENGTH_OF_line: partial(LineMixin.apply_total_length, **kwargs),

        }
        self.apply_functions.update(line_func)

    @staticmethod
    def apply_total_length(base_query: str, geo: BaseGeometry, conn: connection, max_radius_meter: float, **kwargs) -> float:
        """
        Retrieves the total length of the line geometries within $max_radius_meter
        Args:
            base_query: The base query to retrieve the objects, returns geometries in 'geom'
            geo: The geometry object
            conn: The connection to the DB
            max_radius_meter: The maximum radius to fetch the geometries

        Returns:
            The total length as float
        """
        q = f"""
            SELECT 
                CASE WHEN COUNT(*) > 0 THEN 
                    SUM(ST_Length(t.geom, true)) 
                ELSE 0. END as total_length
            FROM ({PostgresFeature._intersect_circle_query(base_query, geo, max_radius_meter)}) t
            WHERE ST_DWithin(t.geom, {geo2sql(geo)}, {max_radius_meter}, true);
            """

        df = get_df(q, conn)

        return df['total_length'].iloc[0]
