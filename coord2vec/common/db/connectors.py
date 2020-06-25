# import psycopg2
from time import sleep

import sqlalchemy as sa

from coord2vec import config


def get_connection(db_name: str = "POSTGRES") -> sa.engine.Engine:
    """
    Gets the connection that will enable to connect to the db
    Args:
        db_name: connection name

    Returns: sql-alchemy connector

    """
    if db_name == 'POSTGRES':
        connection = sa.create_engine(f"postgresql://postgres:1q2w3e4r@localhost:5432")
    else:
        raise AttributeError("Invalid DB tablespace name")
    return connection


import pandas as pd


def read_sql(sql_query, eng=get_connection()):
    df = pd.read_sql(sql_query, eng)
    eng.dispose()
    return df

def insert_df_to_db(df, eng=get_connection()):
    with eng.begin() as con:
        df.to_sql('table1', con, if_exists="append", index=False)
    eng.dispose()

def delete_from_db(delete_sql, eng=get_connection()):
    with eng.begin() as con:
        con.execute(delete_sql)
    eng.dispose()


import numpy as np

if __name__ == '__main__':
    # df = pd.DataFrame({'id': [16891293, 123456789], 'name': ['tsila', 'moshe'], 'age': [52, np.nan]})
    # insert_df_to_db(df)
    # delete_from_db('DELETE FROM table1 WHERE age=52')
    df = read_sql("SELECT * from table1")
    print(df)
