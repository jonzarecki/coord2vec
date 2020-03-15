from geoalchemy2 import Geography
from sqlalchemy import DATE, FLOAT, TEXT, VARCHAR

GEOM = 'geom'
SCORES = 'scores'
MODEL = 'model'
MODIFICATION_DATE = 'modification_date'
GEOM_WKT_HASH = 'geom_wkt_hash'
GEOM_WKT = 'geom_wkt'
EXPERIMENT = 'experiment'
TRAIN_HASH = 'train_geom_hash'

DTYPES = {
    GEOM: Geography(srid=4326),
    EXPERIMENT: VARCHAR(100),
    MODEL: VARCHAR(100),
    SCORES: FLOAT,
    MODIFICATION_DATE: DATE,
    GEOM_WKT_HASH: VARCHAR(300),
    GEOM_WKT: TEXT,
    TRAIN_HASH: TEXT
}
