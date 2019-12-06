from functools import partial

import pandas as pd

from coord2vec.common.db.postgres import connection, get_df
from coord2vec.feature_extraction.postgres_feature import PostgresFeature, AREA_OF_poly


class PolygonMixin(PostgresFeature):
    def __init__(self, apply_type: str, **kwargs):
        super().__init__(apply_type, **kwargs)

        poly_func = {
            AREA_OF_poly: partial(PolygonMixin.apply_area_of, **kwargs),

        }
        self.apply_functions.update(poly_func)

    @staticmethod
    def apply_area_of(base_query: str, q_geoms: str, conn: connection, max_radius: float, **kwargs) -> pd.DataFrame:
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
            filtered_osm_geoms LEFT JOIN {q_geoms} q_geoms
        ON q_geoms.geom=filtered_osm_geoms.q_geom
        )            
            
        SELECT 
            (SELECT CASE WHEN COUNT(*) > 0 THEN 
                        SUM(COALESCE (ST_Area(t_geom, TRUE), 0.)) / 10.764
                    ELSE 0. END
            FROM joined_filt_geoms where q_geom=q_geoms.geom
            ) as total_area
        FROM {q_geoms} q_geoms;
        """

        df = get_df(q, conn)

        return df
