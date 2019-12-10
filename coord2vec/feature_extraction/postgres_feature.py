import datetime
from abc import abstractmethod
from functools import partial
from typing import Tuple, Callable, List

import pandas as pd
from geopandas import GeoDataFrame
from shapely import wkt
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from coord2vec.common.db.postgres import get_df, connect_to_db, connection, get_sqlalchemy_engine, \
    save_gdf_to_temp_table_postgres

# general feature types
from coord2vec.common.itertools import flatten
from coord2vec.feature_extraction.feature import Feature

NEAREST_NEIGHBOUR_all = 'nearest_neighbour'
NUMBER_OF_all = 'number_of'

# polygon feature types
AREA_OF_poly = 'area_of'

# line feature types
LENGTH_OF_line = 'length_of'


def geo2sql(geo: BaseGeometry, to_geography: bool = False) -> str:
    """
    Transforms $geo to the correct srid geometry sql statement
    Args:
        geo: The geometry we want to transform
        to_geography: whether to transform geo into a geometry or geography object

    Returns:
        The query as a str
    """
    sql = f"ST_GeomFromText('{wkt.dumps(geo)}', 4326)"
    sql = f"geography({sql})" if to_geography else sql
    return sql


class PostgresFeature(Feature):
    def __init__(self, apply_type: str, object_name: str, **kwargs):
        super().__init__(**kwargs)
        #  Classes that add apply functions should add them to the dictionary
        self.object_name = object_name
        self.apply_functions = {
            NEAREST_NEIGHBOUR_all: partial(self.apply_nearest_neighbour, **kwargs),
            NUMBER_OF_all: partial(self.apply_number_of, **kwargs)
        }
        self.apply_type = apply_type

        # override parent
        self.all_feature_names = [f"{self.apply_type}_{self.object_name}_{self.max_radius}m"]
        self.all_feature_names = flatten([[feat_name, f"{feat_name}_norm"] for feat_name in self.all_feature_names])
        if self.normed:
            self.feature_names = [feat_name for feat_name in self.all_feature_names if ('norm' in feat_name)]
        else:
            self.feature_names = [feat_name for feat_name in self.all_feature_names if ('norm' not in feat_name)]

    @staticmethod
    def apply_nearest_neighbour(base_query: str, q_geoms: str, conn: connection, max_radius,
                                **kwargs) -> pd.DataFrame:
        q = f"""
        with filtered_osm_geoms as ({PostgresFeature._intersect_circle_query(base_query, q_geoms, max_radius)}),
        
        joined_filt_geoms as (
        SELECT * FROM
            filtered_osm_geoms LEFT JOIN {q_geoms} q_geoms
        ON q_geoms.geom=filtered_osm_geoms.q_geom
        ),
        
        result as (
        SELECT COALESCE ((SELECT MIN(ST_Distance(q_geom, t_geom)) FROM joined_filt_geoms where q_geom=q_geoms.geom), {max_radius}) as dist
            FROM {q_geoms} q_geoms
        )

        SELECT dist, dist / {max_radius} as dist_norm FROM result;
        """

        df = get_df(q, conn)

        return df

    @staticmethod
    def apply_number_of(base_query: str, q_geoms: str, conn: connection, max_radius: float,
                        **kwargs) -> pd.DataFrame:
        q = f"""
        with filtered_osm_geoms as ({PostgresFeature._intersect_circle_query(base_query, q_geoms, max_radius)}),

        joined_filt_geoms as (
        SELECT * FROM
            filtered_osm_geoms LEFT JOIN {q_geoms} q_geoms
        ON q_geoms.geom=filtered_osm_geoms.q_geom
        ),
        
        result as (
        SELECT (SELECT count(*) FROM joined_filt_geoms where q_geom=q_geoms.geom) as cnt   
            FROM {q_geoms} q_geoms
        )

        SELECT cnt, cnt / (PI() * pow({max_radius}, 2)) as cnt_norm FROM result;
        """

        df = get_df(q, conn)

        return df

    @staticmethod
    def _intersect_circle_query(base_query: str, q_geoms: str, max_radius: float) -> str:
        """
        Transform a normal base_query into a query after only elements within the radius from geo remain
        Args:
            base_query: the postgres base query to get geo elements
            q_geoms: table name holding the queries geometries
            max_radius: the radius of the circle to intersect with

        Returns:
            a Postgres query the return only the data on max radius from geo
        """
        query = f"""
        select 
            {q_geoms}.geom as q_geom,
            ST_Intersection(t.geom::geography, ST_Buffer({q_geoms}.geom, {max_radius})) as t_geom
            from {q_geoms} 
                JOIN ({base_query}) t 
                ON ST_DWithin(t.geom, {q_geoms}.geom, {max_radius}, true)
        """
        return query

    @abstractmethod
    def get_postgis_connection(self) -> connection:
        """
        Retrieves the correct connection object in the correct db scheme
        Returns:
            The connection object
        """
        pass

    @abstractmethod
    def _build_postgres_query(self) -> str:
        """
        Returns the postgres query filtering the geometries object relevant in the column 'geom'

        For example: filter all buildings and retrieve their geometry in 'geom'
        :return: The filtering query as a string
        """
        pass

    def extract(self, gdf: GeoDataFrame, only_relevant=False) -> pd.DataFrame:
        """
        Applies the feature on the gdf, returns the series after the apply
        Args:
            gdf: The gdf we want to apply the feature on
            only_relevant: extract only relevant features, (normed or not normed)

        Returns:
            The return values as a Series
        """
        assert self.apply_type in self.apply_functions, f"apply_type {self.apply_type} does not match a function"
        eng = get_sqlalchemy_engine()
        tbl_name = save_gdf_to_temp_table_postgres(gdf, eng)

        res = self.extract_with_tblname(tbl_name)
        if only_relevant:
            res = res[self.feature_names]

        with eng.begin() as con:
            con.execute(f"DROP TABLE {tbl_name}")
        eng.dispose()
        return res

    def extract_with_tblname(self, tbl_name: str) -> pd.DataFrame:
        """
        Applies the feature on the geoms in the table, returns the series after the apply
        Args:
            tbl_name: The table name holding queried geometries

        Returns:
            The return values as a Series
        """
        assert self.apply_type in self.apply_functions, "apply_type does not match a function"
        func = self.apply_functions[self.apply_type]
        conn = connect_to_db()
        res = func(base_query=self._build_postgres_query(), q_geoms=tbl_name, conn=conn)
        assert len(res.columns) == len(self.all_feature_names), \
            f"number of features {res.columns} to feature names {self.all_feature_names} does not match "
        res.columns = self.all_feature_names
        conn.close()

        return res
