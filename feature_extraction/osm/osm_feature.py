from common.db.postgres import connect_to_db, connection
from feature_extraction.feature import Feature


class OsmFeature(Feature):
    """
    This class filters objects from the osm postgres db.
    It filters different tags to build the different features
    """

    def __init__(self, filter_sql: str, apply_type: str, **kwargs):
        super().__init__(apply_type, **kwargs)
        self.filter_sql = filter_sql

    def get_postgis_connection(self) -> connection:
        return connect_to_db()
