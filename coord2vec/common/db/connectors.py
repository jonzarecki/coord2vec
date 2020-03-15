import psycopg2
import sqlalchemy as sa

from coord2vec import config


def get_connection(db_name: str = "NORTH") -> sa.engine.Engine:
    """
    Gets the connection that will enable to connect to the db
    Args:
        db_name: connection name

    Returns: sql-alchemy connector

    """
    if db_name == 'POSTGRES':
        connection = sa.create_engine(f"postgresql://renderer:@{config.postgis_server_ip}:{config.postgis_port}/gis")
    else:
        raise AttributeError("Invalid DB tablespace name")
    return connection
