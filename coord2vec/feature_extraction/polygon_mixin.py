from functools import partial

from shapely.geometry.base import BaseGeometry

from coord2vec.common.db.postgres import connection, get_df
from coord2vec.feature_extraction.postgres_feature import PostgresFeature, geo2sql, AREA_OF_poly


class PolygonMixin(PostgresFeature):
    def __init__(self, apply_type: str, **kwargs):
        super().__init__(apply_type, **kwargs)

        poly_func = {
            AREA_OF_poly: partial(PolygonMixin.apply_area_of, **kwargs),

        }
        self.apply_functions.update(poly_func)

    @staticmethod
    def apply_area_of(base_query: str, geo: BaseGeometry, conn: connection, max_radius_meter: float, **kwargs) -> float:
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
                    SUM(COALESCE (ST_Area(t.geom, TRUE), 0.)) / 10.764
                ELSE 0. END as total_area
            FROM ({PostgresFeature._intersect_circle_query(base_query, geo, max_radius_meter)}) t
                """

        df = get_df(q, conn)

        return df['total_area'].iloc[0]
