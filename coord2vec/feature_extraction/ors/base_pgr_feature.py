from abc import ABC

import geopandas as gpd
import openrouteservice
import pandas as pd
from multiprocess.pool import ThreadPool

from coord2vec.common.db.postgres import get_df, connect_to_db, get_sqlalchemy_engine
from coord2vec.feature_extraction.feature import Feature


class BasePGRFeature(Feature, ABC):
    def __init__(self, transportation_type, **kwargs):
        """This class filters objects from the ors service.
        It gives nice methods for calling ors functions.
        """
        self.client = openrouteservice.Client(base_url=None)
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

        def f(route):
            conn = connect_to_db()
            rout_query = f"""
            with source as (select st_setsrid(st_astext({"'" + route[0] + "'"}), 4326)),
            
                 target as (select st_setsrid(st_astext({"'" + route[1] + "'"}), 4326)),
            
                 node1 as(select way, id::integer from ways_vertices_pgr order by way <#> (select * from source) limit 1),

                 node2 as(select way, id::integer from ways_vertices_pgr order by way <#> (select * from target) limit 1),

                 route as (select * from pgr_dijkstra('select gid::integer as id, 
                                                   source::integer, target::integer, 
                                                   length_m::float as cost from ways',
                                                   (select id from node1),
                                                   (select id from node2), false))
            select sum(route.cost)::float as distance, 
                   sum(((route.cost/1000)/ways.maxspeed_forward)*60*60)::float as duration 
            from route join ways on route.edge=ways.gid;
            """
            shortest_route = get_df(rout_query, conn=conn)
            distance = shortest_route.distance[0]
            duration = shortest_route.duration[0] / 60.0  # shortest_route.duration[0] is in seconds !
            conn.close()
            return [distance, duration]

        # results_list = [f(r) for r in routes]
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
