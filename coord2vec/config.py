import os

from shapely.geometry import Polygon


postgis_server_ip = "localhost"
postgis_port = 15432
ors_server_ip = "localhost"
ors_server_port = 8100

COORD2VEC_DIR_PATH = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(COORD2VEC_DIR_PATH)
USER_ROOT_DIR = os.path.dirname(os.path.dirname(PROJECT_ROOT))


CACHE_DIR = os.path.join("/mnt", "cache_data", "house_price_builder")

TRAIN_CACHE_DIR = os.path.join(CACHE_DIR, 'train')
VAL_CACHE_DIR = os.path.join(CACHE_DIR, 'validation')

DISTANCE_CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache_dir')
TEST_CACHE_DIR = os.path.join(CACHE_DIR, "test_dir")
TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, "tensorboard_runs")

VAL_SAMPLE_NUM = 5_000
TRAIN_SAMPLE_NUM = 50_000

NEG_RATIO = 1.5
TRUE_POSITIVE_RADIUS = 100

IMG_WIDTH, IMG_HEIGHT = (224, 224)
israel_range = [34.482724, 31.492354, 34.583301, 31.585196]
SMALL_TEST_POLYGON = Polygon([])
# beijing_range = [39.747142, 116.205692, 40.041051, 116.632688]
# washington_range = [47.685618, -122.550769, 47.552132, -122.209654]
ENTROPY_THRESHOLD = 1.5
HALF_TILE_LENGTH = 50


BUILDINGS_FEATURES_TABLE = "building_features"

STEP_SIZE = 20
# Oracle config
SCORES_TABLE = 'building_scores'

# tile_server_dns_noport = 'https://api.maptiler.com/tiles/satellite-mediumres/{z}/{x}/{y}.jpg?key=iS9gs1LsLOJfuF3dQvLd'
tile_server_dns_noport = 'http://a.tile.openstreetmap.us/usgs_large_scale/{z}/{x}/{y}.jpg'
tile_server_ports = [80]
RADIUS_IN_METERS = 50
