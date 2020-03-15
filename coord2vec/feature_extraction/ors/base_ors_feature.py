from abc import ABC

import geopandas as gpd
import openrouteservice
import pandas as pd
from multiprocess.pool import ThreadPool

from coord2vec.common.db.postgres import get_df, connect_to_db, get_sqlalchemy_engine
from coord2vec.common.geographic.geo_utils import wkt_to_centers
from coord2vec.feature_extraction.feature import Feature


class BaseORSFeature(Feature, ABC):
    def __init__(self, transportation_type, **kwargs):
        """This class filters objects from the ors service.
        It gives nice methods for calling ors functions.
        """
        self.client = openrouteservice.Client(base_url=None, retry_timeout=30 * 60 * 60)
        self.transportation_type = transportation_type
        super().__init__(**kwargs)

    def _calculate_feature(self, input_gs: gpd.GeoSeries):
        if self.intersect_tbl_name_dict is None or self.input_geom_table is None:
            raise Exception("Must use an OSM feature factory before extracting the feature")

        eng = get_sqlalchemy_engine()
        # input_table = save_geo_series_to_tmp_table(input_gs, eng)

        # calculate the feature
        conn = connect_to_db()
        query = self._build_postgres_query()
        res = get_df(query, conn=conn)
        routes = res[['source_point', 'dest_point']].values

        def f(route_start_dest):
            route_coords = wkt_to_centers(route_start_dest)
            # distance in meters, time in seconds
            shortest_route = self.client.directions(route_coords, profile=self.transportation_type,
                                                    preference='recommended',
                                                    instructions=False,
                                                    geometry=False)

            summary = shortest_route['routes'][0]['summary']
            if not summary:
                # for source and target points that are very close to each other
                distance = 0
                duration = 0
            else:
                distance = summary['distance']
                duration = summary['duration'] / 60.0 if 'duration' in summary else 0  # we want duration in minutes
            return [distance, duration]

        with ThreadPool(4) as p:
            results_list = p.map(f, routes)

        results_df = pd.DataFrame(results_list)
        results_df = pd.concat([results_df, res['geom_id']], names=[self.feature_names, 'geom_id'], axis=1)

        # edit the df
        full_df = pd.DataFrame(index=range(len(input_gs)), columns=self.feature_names)
        if len(res['geom_id']) != 0:
            full_df.iloc[results_df['geom_id']] = results_df.drop('geom_id', axis=1).values
        full_df.fillna(self.default_value, inplace=True)
        full_df['geom'] = input_gs

        # close up the business
        conn.close()
        eng.dispose()

        return full_df
