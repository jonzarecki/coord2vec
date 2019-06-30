from feature_extraction.osm.osm_feature import OsmFeature


class OsmPointFeature(OsmFeature):
    """
    This class filters objects from planet_osm_point.
    It filters different tags to build the different features
    """

    def _build_postgres_query(self) -> str:
        return f"""
        select way as geom from planet_osm_point
            where {self.filter_sql}
        """
