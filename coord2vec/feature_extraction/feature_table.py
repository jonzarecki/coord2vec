# Constants for the cache features table in the Oracle DB
from geoalchemy2 import Geography
from sqlalchemy import DATE, FLOAT, TEXT, VARCHAR

GEOM_WKT_HASH = 'GEOM_WKT_HASH'.lower()
GEOM_WKT = 'GEOM_WKT'.lower()
GEOM = 'GEOM'.lower()
FEATURE_NAME = 'FEATURE_NAME'.lower()
FEATURE_VALUE = 'FEATURE_VALUE'.lower()
MODIFICATION_DATE = 'MODIFICATION_DATE'.lower()

COLUMNS = [GEOM, FEATURE_NAME, FEATURE_VALUE, MODIFICATION_DATE]

DTYPES = {
    GEOM: Geography('GEOMETRY', srid=4326),
    FEATURE_NAME: VARCHAR(100),
    FEATURE_VALUE: FLOAT,
    MODIFICATION_DATE: DATE,
    GEOM_WKT_HASH: VARCHAR(300),
    GEOM_WKT: TEXT
}
