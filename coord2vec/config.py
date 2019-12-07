import os

COORD2VEC_DIR_PATH = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(COORD2VEC_DIR_PATH)
USER_ROOT_DIR = os.path.dirname(os.path.dirname(PROJECT_ROOT))

tile_server_ip = "localhost"
postgis_server_ip = "127.0.0.1"

tile_server_dns_noport = 'http://' + tile_server_ip + ':{p}/tile/{z}/{x}/{y}.png'
tile_server_ports = [8101, 8102, 8103]
h20_port = 8198
postgis_port = 15432

CACHE_DIR = os.path.join("/media/yonatanz/yz", "cache_data", "build_road_park_multi")
# CACHE_DIR = os.path.join("/media/yonatanz/yz", "cache_data", "test_fit_large_multi_build")
# CACHE_DIR = '../coord2vec_data/house_price_builder'
TRAIN_CACHE_DIR = os.path.join(CACHE_DIR, 'train')
VAL_CACHE_DIR = os.path.join(CACHE_DIR, 'validation')

def get_builder():
    from coord2vec.feature_extraction.features_builders import multi_build_builder, house_price_builder
    return multi_build_builder

TEST_CACHE_DIR = os.path.join(CACHE_DIR, "test_dir")
TENSORBOARD_DIR = os.path.join("/media/yonatanz/yz", "tensorboard_runs")
# TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, "tensorboard_runs")

VAL_SAMPLE_NUM = 5_000
TRAIN_SAMPLE_NUM = 50_000

IMG_WIDTH, IMG_HEIGHT = (224, 224)
israel_range = [34.482724, 31.492354, 34.583301, 31.585196]
beijing_range = [116.205692, 39.747142, 116.632688, 40.041051]
washington_range = [-122.550769, 47.685618, -122.209654, 47.552132]
ENTROPY_THRESHOLD = 1.5
HALF_TILE_LENGTH = 100
