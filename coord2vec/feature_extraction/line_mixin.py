import pandas as pd
from functools import partial

from coord2vec.common.db.postgres import connection, get_df
from coord2vec.feature_extraction.postgres_feature import PostgresFeature, LENGTH_OF_line


class LineMixin(PostgresFeature):
    def __init__(self, apply_type: str, **kwargs):
        super().__init__(apply_type, **kwargs)

        line_func = {
            LENGTH_OF_line: partial(LineMixin.apply_total_length, **kwargs),

        }
        self.apply_functions.update(line_func)

    @staticmethod
    def apply_total_length(base_query: str, q_geoms: str, conn: connection, max_radius: float, **kwargs) -> pd.DataFrame:
        """
        Retrieves the total length of the line geometries within $max_radius
        Args:
            base_query: The base query to retrieve the objects, returns geometries in 'geom'
            q_geoms: table name holding the queries geometries
            conn: The connection to the DB
            max_radius: The maximum radius to fetch the geometries

        Returns:
            The total length as float
        """
        q = f"""
        with filtered_osm_geoms as ({PostgresFeature._intersect_circle_query(base_query, q_geoms, max_radius)}),
            
        joined_filt_geoms as (
        SELECT q_geom, t_geom FROM
            filtered_osm_geoms RIGHT JOIN {q_geoms} q_geoms
        ON q_geoms.geom=filtered_osm_geoms.q_geom
        )            
            
        SELECT 
            CASE WHEN COUNT(f.t_geom) > 0 THEN 
                SUM(COALESCE (ST_Length(f.t_geom, true), 0.)) 
            ELSE 0. END as total_length
        FROM joined_filt_geoms f GROUP BY q_geom
            """

        df = get_df(q, conn)

        return df
