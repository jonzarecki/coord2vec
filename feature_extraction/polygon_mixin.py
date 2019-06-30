from abc import ABC
from functools import partial

from shapely.geometry.base import BaseGeometry
from common.db.postgres import connection, get_df

from feature_extraction.feature import Feature, geo2sql


class PolygonMixin(Feature):
    def __init__(self, apply_type: str, **kwargs):
        super().__init__(apply_type, **kwargs)

        poly_func = {
            'area_of': partial(PolygonMixin.apply_area_of, **kwargs),

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
                SELECT SUM(ST_Area(t.geom)) as total_area
                    FROM ({base_query}) t
                    WHERE ST_DWithin(t.geom, {geo2sql(geo)}, {max_radius_meter});
                """

        df = get_df(q, conn)

        return df['total_area'].iloc[0]
