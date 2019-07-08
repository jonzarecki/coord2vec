from functools import partial

from shapely.geometry.base import BaseGeometry

from coord2vec.common.db.postgres import connection, get_df
from coord2vec.feature_extraction.feature import Feature, geo2sql


class LineMixin(Feature):
    def __init__(self, apply_type: str, **kwargs):
        super().__init__(apply_type, **kwargs)

        line_func = {
            'length_of': partial(LineMixin.apply_total_length, **kwargs),

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
                SELECT SUM(ST_Length(t.geom, true)) as total_length
                    FROM ({base_query}) t
                    WHERE ST_DWithin(t.geom, {geo2sql(geo)}, {max_radius_meter}, true);
                """

        df = get_df(q, conn)

        return df['total_length'].iloc[0]
