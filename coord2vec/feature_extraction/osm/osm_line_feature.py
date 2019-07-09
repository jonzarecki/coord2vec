from coord2vec.feature_extraction.line_mixin import LineMixin
from coord2vec.feature_extraction.osm import OsmFeature


class OsmLineFeature(OsmFeature, LineMixin):
    """
    This class filters objects from planet_osm_line.
    It filters different tags to build the different features
    """

    def _build_postgres_query(self) -> str:
        return f"""
        select way as geom from planet_osm_line
            where {self.filter_sql}
        """
