from typing import List

from coord2vec.feature_extraction.feature import Feature, NEAREST_NEIGHBOUR_all, AREA_OF_poly, NUMBER_OF_all, LENGTH_OF_line
from coord2vec.feature_extraction.osm.osm_line_feature import OsmLineFeature
from coord2vec.feature_extraction.osm.osm_polygon_feature import OsmPolygonFeature
from coord2vec.feature_extraction.osm.osm_tag_filters import HOSPITAL, RESIDENTIAL_ROAD


class FeaturesBuilder():
    def __init__(self, features: List[Feature]):
        self.features = features

    def extract(self, gdf):
        return [feature.extract(gdf) for feature in self.features]


baseline_builder = FeaturesBuilder(
    [OsmPolygonFeature(HOSPITAL, apply_type=NEAREST_NEIGHBOUR_all),
     OsmPolygonFeature(HOSPITAL, apply_type=AREA_OF_poly, max_radius_meter=2 * 1000),
     OsmPolygonFeature(HOSPITAL, apply_type=NUMBER_OF_all, max_radius_meter=2 * 1000),
     OsmLineFeature(RESIDENTIAL_ROAD, apply_type=LENGTH_OF_line, max_radius_meter=10),
     OsmLineFeature(RESIDENTIAL_ROAD, apply_type=NUMBER_OF_all, max_radius_meter=10)
     ])