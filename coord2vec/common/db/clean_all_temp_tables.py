import re
import subprocess

from tqdm import tqdm

from coord2vec.common.db.connectors import get_connection

if __name__ == '__main__':
    # subprocess.run("docker exec -it osm-features service postgresql restart", shell=True)
    eng = get_connection('POSTGRES')

    for tbl in tqdm(eng.table_names(), desc="Processing all tables"):
        if len(re.findall('^t[0-9]*$', tbl)) > 0:  # matched a temp table
            with eng.begin() as con:
                con.execute(f"DROP TABLE {tbl}")

