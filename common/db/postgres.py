import psycopg2

def connect_to_db():
    # TODO: implement
    conn = psycopg2.connect("dbname=gis user=renderer")
    return conn


def execute_query(query: str, conn, dispose_conn=False) -> list:
    """
    Executes the query and fetches the results
    Args:
        query: The sql query
        conn: The connection object to the postgres
        dispose_conn: Whether to close the connection after the query

    Returns:
        The results of the query as a list
    """
    with conn.cursor() as cur:
        res = cur.execute(query).fetchall()

    if dispose_conn:
        dispose_conn.close()

    return res
