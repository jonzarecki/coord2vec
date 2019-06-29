import psycopg2

import pandas as pd
import pandas.io.sql as sqlio

from psycopg2._psycopg import connection


def connect_to_db() -> connection:
    """
    Build connection object for the postgres server
    """
    conn = psycopg2.connect(host='127.0.0.1', port=15432, database='gis', user='renderer')
    return conn


def read_df(query: str, conn: connection, dispose_conn=False) -> pd.DataFrame:
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
