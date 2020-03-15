import copy
import datetime
import time
from random import random
from typing import List, Dict, Union, Tuple, Set

import geopandas as gpd
import logging
import sqlalchemy as sa
from geoalchemy2 import Geography, Geometry, WKTElement
from shapely import wkt
from sqlalchemy import FLOAT, INTEGER

from coord2vec.common.db.connectors import get_connection
from coord2vec.common.db.postgres import get_sqlalchemy_engine, save_geo_series_to_tmp_table, add_postgis_index
from coord2vec.common.db.sqlalchemy_utils import get_df, insert_into_table, get_temp_table_name, column_exists
from coord2vec.common.parallel.multiproc_util import parmap
from coord2vec.feature_extraction.ors.base_ors_feature import BaseORSFeature
from coord2vec.feature_extraction.osm.base_postgres_feature import BasePostgresFeature


class PostgresFeatureFactory:
    def __init__(self, features: List[Union[BasePostgresFeature, BaseORSFeature]], input_gs: gpd.GeoSeries):
        """
        Args:
            features: feature to be considered in the feature factory
            input_gs: GeoSeries with the geometries to extract the features on
        """
        self.input_gs = input_gs
        self.features = [copy.deepcopy(feature) for feature in features]
        self.table_filter_dict, self.table_radii = self._create_table_filter_dict()

    def __enter__(self):
        start_time = time.time()
        self.eng = get_connection('POSTGRES')
        self.input_geom_table = save_geo_series_to_tmp_table(self.input_gs, self.eng)
        self.intersection_table_names = self._create_intersection_table(self.input_geom_table, self.eng)
        for feature in self.features:
            if feature.radius in self.intersection_table_names:
                feature.set_intersection_table_names(self.intersection_table_names[feature.radius])
            feature.set_input_geom_table(self.input_geom_table)
            feature.set_input_gs(self.input_gs)
        logging.debug(f"Intersection calculation took {int(time.time() - start_time)} seconds")

    def __exit__(self, exc_type, exc_val, exc_tb):
        with self.eng.begin() as con:
            con.execute(f"DROP TABLE {self.input_geom_table}")
            for radius in self.intersection_table_names:
                for table in self.intersection_table_names[radius].values():
                    con.execute(f"DROP TABLE {table}")
        self.eng.dispose()

    def _create_intersection_table(self, geom_table_name: str, eng: sa.engine.Engine) -> Dict[float, Dict[str, str]]:
        """
        Create a temporary intersection table, to be later used for all the sub-features.
        The table is created using self.table_filter_dict, and the features radii

        Args:
            geom_table_name: name of the geometries table to intersect on.
            eng: sql Alchemy engine.

        Returns:
            The names of the temporary intersection table for each radius, then original table
        """
        radius_table_to_tmp_table_names = {}
        all_tpls = []
        for table, filters_dict in self.table_filter_dict.items():
            all_radii = self.table_radii[table]
            for radius in all_radii:
                all_tpls.append((table, filters_dict[radius], radius))

        def calc_intersect(tpl_idx, table, filters_dict, radius):
            eng = get_connection('POSTGRES')
            filters_columns_sql = ',\n'.join(
                [f"CASE WHEN {filter_sql} THEN 1 ELSE 0 end as {filter_name}" for filter_name, filter_sql in
                 filters_dict.items()])

            filters_sql = ' or '.join(filters_dict.values())
            tbl_name = f"{get_temp_table_name()}{tpl_idx}"

            # add height if exists
            height_exists = column_exists('height', table, eng)
            inner_height_sql = "height, absolute_ground_surface_height," if height_exists else ""
            outer_height_sql = """t.height as height,
                        t.absolute_ground_surface_height as ground_height,
                        t.absolute_ground_surface_height + t.height as absolute_height,""" if height_exists else ""

            query = f"""
                    create UNLOGGED TABLE {tbl_name}
                    as
                        select
                        1.0 as coverage,
                        {outer_height_sql}
                        q.geom_id as geom_id,
                        q.geom as q_geom,
                        t.geom as t_geom,
                        Geography(t.geom) as t_geog,
                        {', '.join(filters_dict.keys())}

                        from {geom_table_name} q
                        JOIN (select way as geom,
                                    {inner_height_sql}
                                    {filters_columns_sql}
                                    from {table}
                                    WHERE {filters_sql}) t
                        ON ST_DWithin(t.geom, q.geom, {radius}, true)

                    """

            eng.execute(query)
            add_postgis_index(eng, tbl_name, 'q_geom')
            add_postgis_index(eng, tbl_name, 't_geom')
            add_postgis_index(eng, tbl_name, 't_geog')
            eng.dispose()
            return radius, table, tbl_name

        res = parmap(lambda p: calc_intersect(p[0], *p[1]), list(enumerate(all_tpls)),
                     use_tqdm=True, desc="Calculating intersection", unit="table", leave=False)
        for radius, table, tbl_name in res:
            radius_table_to_tmp_table_names.setdefault(radius, {}).update({table: tbl_name})

        return radius_table_to_tmp_table_names

    def _create_table_filter_dict(self) -> Tuple[Dict[str, Dict[float, Dict[str, str]]], Dict[str, Set[float]]]:
        """
        combine the table-filter dictionaries of all the different features into one.

        Returns:
            a combined table filter dictionary
        """
        table_radius_filter_dict = {}
        table_radii = {}
        for feature in self.features:
            for table, filter_dict in feature.table_filter_dict.items():
                r = feature.radius
                if table not in table_radius_filter_dict:
                    table_radius_filter_dict[table] = {}
                    table_radii[table] = set()

                if r not in table_radius_filter_dict[table]:
                    table_radius_filter_dict[table][r] = {}

                table_radius_filter_dict[table][r].update(filter_dict)  # features for table and radius
                table_radii[table].add(feature.radius)  # all radius

        return table_radius_filter_dict, table_radii
