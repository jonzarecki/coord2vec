import logging
import random
import time
from datetime import datetime
from typing import List, Optional

import cx_Oracle
import pandas as pd
import sqlalchemy as sa
from geoalchemy2 import WKTElement
from geoalchemy2.types import _GISType, Geometry
from tqdm import tqdm

from coord2vec.common.db.connectors import get_connection
from coord2vec.feature_extraction.feature_table import FEATURE_NAME

MAX_TABLE_NAME_ORACLE = 30
VARCHAR2_MAX_LEN = 4000


def get_df(sql: str, eng: sa.engine.Engine, retry_num=3, dispose_eng=False) -> pd.DataFrame:
    """
    Queries the DB in order to retrieve the DF. logs and retries when failed.
    Column names in the returned DF are capitalized
    :param sql: the sql string for getting the DB
    :param eng: The connection to the DB
    :param retry_num: The maximum number of retries that will be attempted in case a DB error will be thrown
    :param dispose_eng: Whether to dispose of the connection after the read
    :return: The dataframe as read from the DB
    """
    st_time = time.time()
    read_id = random.randint(1, 500)
    logging.debug(f"id={read_id} - Starting SQL query ")
    for _ in range(retry_num):
        try:
            ret_df = pd.read_sql(sql, eng)
            break
        except cx_Oracle.DatabaseError:  # TODO: add similar for postgres
            logging.error("DB error: continuing . . .")
            time.sleep(2)
        except Exception as e:
            print("error ! ")
            logging.error(f"Exception at query - {sql}", stack_info=True)
            logging.error(e)
            raise
    else:
        logging.error(f"Could not read query - {sql}", stack_info=True)
        raise cx_Oracle.DatabaseError

    logging.debug(f"id={read_id} - Read {len(ret_df)} lines successfully in {int(time.time() - st_time)} secs")
    if dispose_eng:
        eng.dispose()

    return ret_df


def drop_table(tbl_name: str, eng: sa.engine.Engine, dispose_eng=False):
    try:
        if eng.has_table(tbl_name):
            with eng.begin() as con:
                con.execute(f"DROP TABLE {tbl_name}")
    finally:
        if dispose_eng:
            eng.dispose()


def datetime2sql(dt: datetime) -> str:
    """
    Converts datetime format into SQL string format. Also applies the "to_date" function
    Args:
        dt: An datetime object

    Returns:
    String containing the datetime
    """
    return f"to_date('{dt.strftime('%Y-%m-%d %H:%M:%S')}', 'YYYY-MM-DD HH24:MI:SS')"


def str2datetime(s: str) -> Optional[datetime]:
    """
    Converts the string into a datetime format, if unparsable, returns None
    Args:
        s: The date string

    Returns:
        The datetime object, if unparsable, returns None
    """
    try:
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None


def add_postgis_index(eng: sa.engine.Engine, table_name: str, geom_col: str):
    with eng.begin() as con:
        con.execute(f"create index {table_name}_{geom_col}_idx on {table_name} using gist ({geom_col});")


def get_temp_table_name() -> str:
    """
    Returns a random table name (should be used by all functions to avoid using the same name)
    """
    return f"t{datetime.now().strftime('%H%M%S%f')}{int(1000000 * random.random())}"


def insert_into_table(eng: sa.engine.Engine, df: pd.DataFrame, table_name: str, dtypes: dict = None,
                      unique_columns=None, index_columns=None, hash_index_columns=None, dispose_eng=False):
    """
    Adds df to a new table called $table_name
    Args:
        eng: An engine object connecting the db
        df: The dataframe we want to insert to the DB
        table_name: The new table's name, assuming it is not in the DB
        dtypes: The data-types for each column in the DB
        unique_columns: Optional param for adding a unique key index for several columns,
                            needed for using merge_to_db in postgresql. If set, $dtypes also needs to be set
        dispose_eng: Whether to dispose of the engine after the read

    Returns:
        None
    """
    table_name = table_name.lower()
    if unique_columns is not None:
        assert dtypes is not None, "if unique_columns is set, dtypes cannot be none, to handle gis columns correctly"

    if dtypes is None:
        dtypes = {}
    with eng.begin() as con:
        df.to_sql(table_name, con, if_exists="append", index=False, dtype=dtypes)

        # for some reason oracle does problems with this, it is only needed in postgres so whatever
        if unique_columns is not None and eng.dialect.name == "postgresql":
            from coord2vec.common.db.postgres import get_index_str_for_unique
            con.execute(f"CREATE UNIQUE INDEX {table_name}_uind "
                        f"ON {table_name} ({get_index_str_for_unique(unique_columns, dtypes)});")

        if index_columns is not None:
            for col in index_columns:
                con.execute(f"CREATE INDEX {table_name}_{col}_ind ON {table_name} (col);")
        if hash_index_columns is not None:
            for col in hash_index_columns:
                con.execute(f"CREATE INDEX {table_name}_{col}_ind ON {table_name} using hash(col);")

    if dispose_eng:
        eng.dispose()


def merge_to_table(eng: sa.engine.Engine, df: pd.DataFrame, table_name: str, compare_columns: List[str],
                   update_columns: List[str], dtypes: dict, temp_table_name: str = None, dispose_eng=False):
    """
    Merges the dataframe into an existing table by creating a temp table with for the df and the merging it into the existing one.
    For rows with matching $compare columns we UPDATE the other values in $UPDATE_COLUMNS
    Args:
        eng: An engine object connecting the db
        df: The dataframe we want to insert to the DB
        table_name: The existing table's name
        compare_columns: The columns we want to compare existing rows with
        update_columns: The columns we want to update in case a matching row is found
        temp_table_name: optional, a name for the temp table for the DB
        dtypes: The data-types for each column in the DB
        dispose_eng: Whether to dispose of the engine after the read

    Returns:
        None
    """
    table_name = table_name.lower()  # fixes stuff for postgres

    if df.empty:
        return

    if dtypes is None:
        dtypes = {}
    if temp_table_name is None:
        temp_table_name = get_temp_table_name()
    if eng.dialect.name.lower() == "oracle" and (len(temp_table_name) > MAX_TABLE_NAME_ORACLE or \
                                                 len(table_name) > MAX_TABLE_NAME_ORACLE):
        raise Exception('table name is too long')

    if len(df) > 200_000:
        chunk_size = 100_000
        for i in tqdm(range(0, len(df), chunk_size), desc=f"Merging into {table_name}", unit="100_000 chunk"):
            df_chunk = df.iloc[i:min(len(df), i + chunk_size)]
            merge_to_table(eng, df_chunk, table_name, compare_columns, update_columns, dtypes=dtypes)

    else:
        try:
            # logger = logging.getLogger()
            # logger.info(f"Writing {len(df)} rows to {table_name} table")
            if not eng.has_table(table_name):
                insert_into_table(eng, df, table_name, dtypes, compare_columns)
            else:
                if eng.dialect.name.lower() not in ("oracle", "postgresql"):
                    raise RuntimeError(f"merge into does not work for {eng.dialect.name}")

                insert_into_table(eng, df, temp_table_name, dtypes, compare_columns)

                if eng.dialect.name.lower() == "oracle":
                    on_statment = "\nAND ".join([f"curr.{col} = tmp.{col}" for col in compare_columns])
                    set_statment = "\n,".join([f"curr.{col} = tmp.{col}" for col in update_columns])
                    all_columns = compare_columns + update_columns
                    all_columns_names = ",".join(all_columns)
                    all_columns_values = ",".join([f"tmp.{col}" for col in all_columns])
                    sql = f"""
                    merge into {table_name} curr
                    using (select {all_columns_names} from {temp_table_name}) tmp
                    on ({on_statment})
                    when matched then
                    update set {set_statment}
                    when not matched then
                    insert ({all_columns_names})
                    values ({all_columns_values})
                    """
                else:  # postgresql
                    set_statment = ",".join([f"{col} = EXCLUDED.{col}" for col in update_columns])  # postgres syntax
                    all_columns = compare_columns + update_columns
                    all_columns_names = ",".join(all_columns)
                    from coord2vec.common.db.postgres import get_index_str_for_unique
                    on_statment = get_index_str_for_unique(compare_columns, dtypes)

                    sql = f"""
                    INSERT INTO {table_name} ({all_columns_names})
                    SELECT {all_columns_names}
                    FROM {temp_table_name} tmp
                    
                    ON CONFLICT ({on_statment})
                    DO UPDATE SET {set_statment};
                    """  # can fail if no key is saved on the on_statement columns

                with eng.begin() as con:
                    con.execute(sql)
                    con.execute(f"drop table {temp_table_name}")
        finally:
            if eng.has_table(temp_table_name):
                with eng.begin() as con:
                    con.execute(f"drop table {temp_table_name}")

    if dispose_eng:
        eng.dispose()


# TODO: move (oracle specific) + remove is_wkt
def add_sdo_geo_to_table(table_name: str, wkt_geo_column: str, geo_sdo_column: str, eng: sa.engine.Engine,
                         is_wkt=True, no_asserts=False, dispose_eng=False):
    """
    Adds a separate SDO_GEOMETRY column from an existing wkt/wkb column
    Args:
        table_name: The table we're working on
        wkt_geo_column: The name of the column containing the wkt/wkb
        geo_sdo_column: The name of the column we want to store the sdo_geometry object
        eng: An engine object connecting the db
        is_wkt: True if wk_geo_column contains data in wkt format (otherwise in wkb)
        no_asserts: True if no asserts on columns are made (will override existing data)
        dispose_eng: Whether to dispose of the engine after the function

    Returns:
         None
    """
    df = get_df(f"SELECT * FROM {table_name} WHERE ROWNUM < 1", eng)  # fetch only one row
    if not no_asserts:
        assert wkt_geo_column in df.columns, f"{wkt_geo_column} not in table {table_name}"
        assert geo_sdo_column not in df.columns, f"{geo_sdo_column} already in table {table_name}"

    if geo_sdo_column not in df.columns and wkt_geo_column in df.columns:
        eng.execute(f"""
        ALTER TABLE {table_name}
        ADD {geo_sdo_column} SDO_GEOMETRY
                     """)
        eng.execute("COMMIT")

    # run for each feature seperetly
    feature_names = pd.read_sql(f"""select distinct {FEATURE_NAME} from {table_name}""", eng).iloc[:, 0]
    conn = eng.raw_connection()
    cur = conn.cursor()

    def add_sdo(feature_name):
        SELECT_SDO_GEO = f"""select SDO_GEOMETRY({wkt_geo_column}, 4326) as {geo_sdo_column}, ROWID as rid 
                            from {table_name}
                            where {geo_sdo_column} IS NULL
                            and {FEATURE_NAME} = '{feature_name}'
                            """

        # TIP: when using weird SDO_UTIL functions its better to use the raw connection.
        # In this case no values were returned by the merge into. only with the
        cur.execute(f"""
        merge into {table_name} curr
                    using ({SELECT_SDO_GEO}) tmp
                    on (curr.ROWID = tmp.rid)
                    when matched then
                    update set curr.{geo_sdo_column} = tmp.{geo_sdo_column}    
        """)
        conn.commit()

    [add_sdo(feature_name) for feature_name in tqdm(feature_names, desc='adding SDO to features', unit='feature')]
    cur.close()

    # fix coordinate system
    eng.execute(f"update {table_name} T set T.{geo_sdo_column}.SDO_SRID = 4326 WHERE T.{geo_sdo_column} is not null")

    # add spatial index and add to user_sdo_geom_metadata table
    usersdo_df = get_df("SELECT * FROM user_sdo_geom_metadata", eng)
    if (table_name, geo_sdo_column) not in [tuple(row) for row in usersdo_df[['TABLE_NAME', 'COLUMN_NAME']].values]:
        eng.execute(f"""
        INSERT INTO user_sdo_geom_metadata
        VALUES ('{table_name}', '{geo_sdo_column}', sdo_dim_array(sdo_dim_element('X', -100, 100, 0.000005),
                                                       sdo_dim_element('Y', -100, 100, 0.000005)), 4326)
        """)

    is_there_index = len(eng.execute(f"""
                    select index_name
                    from SYS.ALL_INDEXES
                    where table_name = '{table_name}'
                    """).fetchall()) > 0

    if not is_there_index:
        acronym_short_geo_sdo = ''.join([s[0] for s in geo_sdo_column.split('_')])  # first letter of each word
        eng.execute(f"""
                    CREATE INDEX {table_name}_{acronym_short_geo_sdo}_idx
                    ON {table_name} ({geo_sdo_column}) INDEXTYPE IS MDSYS.SPATIAL_INDEX
                    """)

    if dispose_eng:
        eng.dispose()


def column_exists(column: str, table: str, eng, dispose_eng=False):
    query = f"select * from {table} limit 1"
    columns = get_df(query, eng).columns
    if dispose_eng:
        eng.dispose()
    return column in columns


def oracle_to_postgres(oracle_con: sa.engine.Engine, postgres_con: sa.engine.Engine,
                       source_table_name: str, destination_table_name: str = None,
                       geom_col: str = None, drop_columns: List = None, limit: int = None):
    """
    Replicate table from oracle DB to postgres DB.
    Args:
        oracle_con: connector to the oracle DB
        postgres_con: connector to the postgres DB
        source_table_name: table name in the oracle DB
        destination_table_name: desired table name in the postgres DB. If None, then gives 'source_table_name'.
        geom_col: name of the geo column. if None then no column will be treated as a geo column
        drop_columns: list of all columns in the source table that you want to drop
        limit: number of rows to copy. If None then copies all rows.

    Returns:
        pushed table to the postgres table
    """
    limit_condition = f"where rownum <= {limit}" if limit is not None else ""
    destination_table_name = destination_table_name if destination_table_name is not None else source_table_name

    df = pd.read_sql(f"""select * from {source_table_name} {limit_condition}""", oracle_con)
    df = df.drop(drop_columns, axis=1)

    dtype = None
    if geom_col is not None:
        df = df.rename({geom_col: 'way'}, axis=1)
        dtype = {'way': Geometry('GEOMETRY', srid=4326)}
        df['way'] = df['way'].map(lambda geo: WKTElement(geo, srid=4326))

    drop_table(destination_table_name, postgres_con)

    # df.to_sql(table_name, postgres_con, if_exists='replace', index=False, dtype=dtype)
    for i in tqdm(range(len(df)), desc="Pushing rows to DB", unit="row"):
        try:
            df.iloc[[i]].to_sql(destination_table_name, postgres_con, if_exists='append', index=False,
                                dtype=dtype)
        except:
            logging.info(f'error in sample {i}')

    if geom_col is not None:
        add_postgis_index(postgres_con, destination_table_name, 'way')
