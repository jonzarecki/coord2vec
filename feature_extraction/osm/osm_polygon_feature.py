from feature_extraction.osm.osm_feature import OsmFeature


class OsmPolygonFeature(OsmFeature):
    """
    This class filters objects from planet_osm_polygon.
    It filters different tags to build the different features
    """

    def _build_postgres_query(self) -> str:
        return f"""
        select way as geom from planet_osm_polygon
            where {self.filter_sql}
        """
