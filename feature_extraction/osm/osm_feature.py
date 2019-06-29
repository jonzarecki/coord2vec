from common.db.postgres import connect_to_db, connection
from feature_extraction.feature import Feature


class OsmFeature(Feature):
    """
    This class filters objects from planet_osm_point.
    It filters different tags to build the different features
    """

    def get_postgis_connection(self) -> connection:
        return connect_to_db()

    def __init__(self, filter_sql: str, apply_type: str):
        super().__init__(apply_type)
        self.filter_sql = filter_sql

    def _build_postgres_query(self) -> str:
        return f"""
        select * from planet_osm_point
            where {self.filter_sql}
        """

