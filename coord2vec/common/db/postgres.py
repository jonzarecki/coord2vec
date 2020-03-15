import datetime
from typing import List

import geopandas as gpd
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import sqlalchemy as sa
from geoalchemy2 import WKTElement, Geography
from geoalchemy2.types import _GISType
from psycopg2._psycopg import connection
from shapely.geometry import Point
from sqlalchemy import create_engine

from coord2vec import config
from coord2vec.common.db.sqlalchemy_utils import insert_into_table, get_temp_table_name


def connect_to_db() -> connection:
    """
    Build connection object for the postgres server
    """
    conn = psycopg2.connect(host=config.postgis_server_ip, port=config.postgis_port, database='gis', user='renderer')
    return conn


def get_sqlalchemy_engine() -> sa.engine.Engine:
    return create_engine(
        f"postgresql://renderer:@{config.postgis_server_ip}:{config.postgis_port}/gis"
    )


# TODO: why 2 get_df. delete one
def get_df(query: str, conn: connection, dispose_conn=False) -> pd.DataFrame:
    """
    Executes the query and fetches the results
    Args:
        query: The sql query
        conn: The connection object to the postgres
        dispose_conn: Whether to close the connection after the query

    Returns:
        The results of the query as a DataFrame
    """
    res = sqlio.read_sql_query(query, conn)

    if dispose_conn:
        conn.close()

    return res


def save_geo_series_to_tmp_table(geo_series: gpd.GeoSeries, eng: sa.engine.Engine) -> str:
    """
    Save a geo series as a table in the db, for better performance
    Args:
        geo_series: The GeoSeries to be inserted into a db table
        eng: SQL Alchemy engine

    Returns:
        The name of the new table
    """
    geo_series = geo_series.rename('geom')
    gdf = gpd.GeoDataFrame(geo_series, columns=['geom'], geometry='geom')
    gdf['geom'] = gdf.geometry.apply(lambda x: WKTElement(x.wkt, srid=4326))
    gdf['geom_id'] = range(len(gdf))
    tbl_name = get_temp_table_name()
    insert_into_table(eng, gdf, tbl_name, dtypes={'geom': Geography(srid=4326), 'geom_id': sa.INT})
    add_postgis_index(eng, tbl_name, 'geom')
    return tbl_name


def get_index_str_for_unique(index_columns: List[str], dtypes: dict):
    return ",".join([f"ST_GeoHash({col})" if isinstance(dtypes[col], _GISType) else col
                     for col in index_columns])


def add_postgis_index(eng: sa.engine.Engine, table_name: str, geom_col: str):
    with eng.begin() as con:
        con.execute(f"create index {table_name}_{geom_col}_idx on {table_name} using gist ({geom_col});")