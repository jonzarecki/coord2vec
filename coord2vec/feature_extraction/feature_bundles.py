from itertools import product
from typing import List

import geopandas as gpd
from shapely.geometry import Polygon

from coord2vec.common.geographic.geo_utils import sample_grid_in_poly
from coord2vec.common.itertools import flatten
from coord2vec.feature_extraction.feature import Feature
from coord2vec.feature_extraction.features.ors_features.osm_route import OSMRoute
from coord2vec.feature_extraction.features.osm_features.area_of import AreaOf
from coord2vec.feature_extraction.features.osm_features.area_of_nearest import AreaOfNearest
from coord2vec.feature_extraction.features.osm_features.building_scores import BuildingScores
from coord2vec.feature_extraction.features.osm_features.distnace_to_nearest import DistanceTo
from coord2vec.feature_extraction.features.osm_features.heights import Heights
from coord2vec.feature_extraction.features.osm_features.number_of import NumberOf
from coord2vec.feature_extraction.features.osm_features.object_distances import ObjectDistances
from coord2vec.feature_extraction.features.osm_features.total_length import TotalLength
from coord2vec.feature_extraction.features.other_features.area_of_self import AreaOfSelf
from coord2vec.feature_extraction.features_builders import FeaturesBuilder
from coord2vec.feature_extraction.osm.base_postgres_feature import BasePostgresFeature
from coord2vec.feature_extraction.osm.osm_tag_filters import *

from coord2vec.feature_extraction.sparse import SparseFilter


def relevant_feat_types(f_type: str):
    f_type_upper = f_type.upper()
    l = [DistanceTo, NumberOf]
    # TODO: order everything
    if "LINE" in f_type_upper:
        l = l[:-1] + [TotalLength, AreaOfNearest]
    if "POLYGON" in f_type_upper:
        l += [AreaOf, AreaOfNearest]
    if "RARE" in f_type_upper:
        pass  # ors is not up in server yet
        # l.append(OSMRoute)

    return l


def all_radii_up_to(min_radius: float = 0, max_radius: float = 500000) -> List[float]:
    all_radii = [0, 50, 100, 250, 500, 1000, 3000]
    return [r for r in all_radii if min_radius <= r <= max_radius]


karka_bundle_features = [
    # ROADS
    (MAJOR_ROAD, OSM_LINE_TABLE, "major_road", all_radii_up_to(100, 1000), relevant_feat_types("line_length"), 1),
    (MINOR_ROAD, OSM_LINE_TABLE, "minor_road", all_radii_up_to(50, 250), relevant_feat_types("line_length"), 1),

    # BUILDING
    (BUILDING, OSM_POLYGON_TABLE, "building", all_radii_up_to(50, 250), relevant_feat_types("polygon_area"), 1),
    # (BUILDING, OSM_POLYGON_TABLE, "building", all_radii_up_to(0, 250), [Heights], 2),  # doesn't work for normal osm
    (BUILDING, OSM_POLYGON_TABLE, "building", all_radii_up_to(max_radius=0), [AreaOf], 3),
    # (JUNCTIONS, JUNCTIONS_TABLE, "junction", all_radii_up_to(max_radius=250), [OSMRoute]),
]

all_bundle_features = flatten([
    karka_bundle_features
])


# TODO: bad name
def create_building_features(elements=None, level_importance=10):
    if elements is None:
        elements = all_bundle_features

    all_features = []
    for filt, table_name, obj_name, radii, features_types, importance in elements:
        if importance <= level_importance:
            for radius, feat_type in product(radii, features_types):
                all_features.append(feat_type(filt, table_name, obj_name, radius=radius))
    return all_features