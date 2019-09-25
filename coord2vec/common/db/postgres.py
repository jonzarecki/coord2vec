import datetime

import psycopg2

import pandas as pd
import pandas.io.sql as sqlio
import sqlalchemy as sa
from geopandas import GeoDataFrame
from psycopg2._psycopg import connection
from sqlalchemy import create_engine
from geoalchemy2 import Geometry, WKTElement, Geography

from coord2vec import config


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


def save_gdf_to_temp_table_postgres(gdf: GeoDataFrame, eng: sa.engine.Engine) -> str:
    gdf = gdf.copy(deep=True)
    gdf['geom'] = gdf.geometry.apply(lambda x: WKTElement(x.wkt, srid=4326))
    # drop the geometry column as it is now duplicative
    # gdf.drop('geometry', 1, inplace=True)

    # Use 'dtype' to specify column's type
    # For the geom column, we will use GeoAlchemy's type 'Geometry'
    tbl_name = f"t{datetime.datetime.now().strftime('%H%M%S%f')}"
    gdf.to_sql(tbl_name, eng, if_exists='replace', index=False,
                        dtype={'geom': Geography('POINT', srid=4326)})
    eng.execute(f"create index {tbl_name}_geom_idx on {tbl_name} using gist (geom);")
    return tbl_name