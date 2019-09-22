import itertools
from typing import List, Tuple, Union

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt

from coord2vec.feature_extraction.postgres_feature import PostgresFeature, NEAREST_NEIGHBOUR_all, AREA_OF_poly, NUMBER_OF_all, \
    LENGTH_OF_line
from coord2vec.feature_extraction.osm.osm_line_feature import OsmLineFeature
from coord2vec.feature_extraction.osm.osm_polygon_feature import OsmPolygonFeature
from coord2vec.feature_extraction.osm.osm_tag_filters import *


class FeaturesBuilder:
    """
    A data class for choosing the desired features
    """

    def __init__(self, features: List[Union[PostgresFeature, List[PostgresFeature]]]):
        """
        features to be used in this builder
        Args:
            features: a list of features (of class Feature)
        """
        flatten = lambda l: list(itertools.chain.from_iterable([(i if isinstance(i, list) else [i]) for i in l]))
        self.features = flatten(features)

    def extract(self, gdf: GeoDataFrame):
        """
        extract the desired features on desired points
        Args:
            gdf: a GeoDataFrame with at least one column of the desired POINTS

        Returns:
            a pandas dataframe, with columns as features, and rows as the points in gdf
        """
        features_gs_list = [feature.extract(gdf) for feature in self.features]
        features_df = pd.concat(features_gs_list, axis=1)
        features_df.columns = [feature.name for feature in self.features]
        return features_df

    def extract_coordinates(self, coords: List[Tuple[float, float]]) -> pd.DataFrame:
        """
        extract the desired features on desired points
        Args:
            coords: list of coordinates

        Returns:
            a pandas dataframe, with columns as features, and rows as the points in gdf
        """
        wkt_points = [wkt.loads(f'POINT ({coord[1]} {coord[0]})') for coord in coords]
        gdf = GeoDataFrame(pd.DataFrame({'geom': wkt_points}), geometry='geom')
        return self.extract(gdf)


def poly_multi_feature(filter, name, radii: List[int] = [50]) -> List[PostgresFeature]:
    features = []
    for radius in radii:
        features += [OsmPolygonFeature(filter, name=f'nearest_{name}_{radius}_m', apply_type=NEAREST_NEIGHBOUR_all,
                                       max_radius_meter=radius),
                     OsmPolygonFeature(filter, name=f'area_of_{name}_{radius}m', apply_type=AREA_OF_poly,
                                       max_radius_meter=radius),
                     OsmPolygonFeature(filter, name=f'number_of_{name}_{radius}m', apply_type=NUMBER_OF_all,
                                       max_radius_meter=radius)]
    return features


def line_multi_feature(filter, name, radii: List[int] = [50]) -> List[PostgresFeature]:
    features = []
    for radius in radii:
        features += [OsmLineFeature(filter, name=f'dist_2_nearest_{name}_{radius}m', apply_type=NEAREST_NEIGHBOUR_all,
                                    max_radius_meter=radius),
                     OsmLineFeature(filter, name=f'length_of_{name}_{radius}m', apply_type=LENGTH_OF_line,
                                    max_radius_meter=radius),
                     OsmLineFeature(filter, name=f'number_of_{name}_{radius}m', apply_type=NUMBER_OF_all,
                                    max_radius_meter=radius)]
    return features


house_price_builder = FeaturesBuilder(
    [poly_multi_feature(BUILDING, 'building'),
     poly_multi_feature(PARK, 'park'),
     line_multi_feature(ROAD, 'road')]
)

example_features_builder = FeaturesBuilder(
    [OsmPolygonFeature(HOSPITAL, name='nearest_hospital', apply_type=NEAREST_NEIGHBOUR_all),
     OsmPolygonFeature(HOSPITAL, name='area_of_hospital_1km', apply_type=AREA_OF_poly, max_radius_meter=1000),
     OsmPolygonFeature(HOSPITAL, name='number_of_hospital_1km', apply_type=NUMBER_OF_all, max_radius_meter=1000),
     OsmLineFeature(RESIDENTIAL_ROAD, name='length_of_residential_roads_10m', apply_type=LENGTH_OF_line,
                    max_radius_meter=10),
     OsmLineFeature(RESIDENTIAL_ROAD, name='number_of_residential_roads_10m', apply_type=NUMBER_OF_all,
                    max_radius_meter=10)
     ])
