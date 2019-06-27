from typing import Tuple, List

from feature_extraction.feature import Feature


class OsmFeature(Feature):
    """
    This class filters objects from planet_osm_point.
    It filters different tags to build the different features
    """
    def __init__(self, filters: List[Tuple[str, str]]):
        self.filters = filters

    def _build_postgres_query(self) -> str:
        filter_terms = [f"{f[0]}={f[1]}" for f in self.filters]

        return f"""
        select * from planet_osm_point
            where {" and ".join(filter_terms)}
        """

