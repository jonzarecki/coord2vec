import os
from datetime import datetime

COORD2VEC_DIR_PATH = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(COORD2VEC_DIR_PATH)
USER_ROOT_DIR = os.path.dirname(os.path.dirname(PROJECT_ROOT))

tile_server_ip = "localhost"
postgis_server_ip = "127.0.0.1"

tile_server_dns_noport = 'http://' + tile_server_ip + ':{p}/tile/{z}/{x}/{y}.png'
tile_server_ports = [8101, 8102, 8103]
h20_port = 8198
postgis_port = 15432


LONG_TERM_DIR = "/media/yonatanz/yz/"
PROJ_LONG_TERM_DIR = os.path.join(LONG_TERM_DIR, "coord2vec")
DATA_LONG_TERM_DIR = os.path.join(LONG_TERM_DIR, "data")
CACHE_DIR = os.path.join(DATA_LONG_TERM_DIR, "coord2vec_data")

# DATA_NAME = "build_road_park_multi_with_norm"
DATA_NAME = 'build_park_road_partial_norm'

EXPR_NAME = f'{DATA_NAME}_norm_resnet34'
TEST_CACHE_DIR = os.path.join(CACHE_DIR, "test_dir")

TRAIN_CACHE_DIR = os.path.join(CACHE_DIR, DATA_NAME, 'train')
VAL_CACHE_DIR = os.path.join(CACHE_DIR, DATA_NAME, 'validation')

_curr_time = str(datetime.now())

TENSORBOARD_DIR = os.path.join(PROJ_LONG_TERM_DIR, "tensorboard_runs", EXPR_NAME, _curr_time)
CURRENT_EXPR_DIR = os.path.join(PROJ_LONG_TERM_DIR, EXPR_NAME, _curr_time)
SAVED_MODEL_DIR = os.path.join(CURRENT_EXPR_DIR, 'models')
TMP_EXPR_FILES_DIR = os.path.join(CURRENT_EXPR_DIR, "project_files")


builder_name = "house_price_builder_partial"
def get_builder():
    from coord2vec.feature_extraction.features_builders import multi_build_builder, house_price_builder, \
        house_price_builder_partial
    print(f"working with house_price_builder_partial")
    return house_price_builder_partial

def update_params(opt):
    global TENSORBOARD_DIR, CURRENT_EXPR_DIR, SAVED_MODEL_DIR, TMP_EXPR_FILES_DIR
    EXPR_NAME = f'{DATA_NAME}_{opt.arch}_{builder_name}'

    TENSORBOARD_DIR = os.path.join(PROJ_LONG_TERM_DIR, "tensorboard_runs", EXPR_NAME, _curr_time)
    CURRENT_EXPR_DIR = os.path.join(PROJ_LONG_TERM_DIR, EXPR_NAME, _curr_time)
    SAVED_MODEL_DIR = os.path.join(CURRENT_EXPR_DIR, 'models')
    TMP_EXPR_FILES_DIR = os.path.join(CURRENT_EXPR_DIR, "project_files")


VAL_SAMPLE_NUM = 10_000
TRAIN_SAMPLE_NUM = 100_000

IMG_WIDTH, IMG_HEIGHT = (224, 224)
israel_range = [34.482724, 31.492354, 34.583301, 31.585196]
beijing_range = [116.205692, 39.747142, 116.632688, 40.041051]
washington_range = [-122.550769, 47.685618, -122.209654, 47.552132]
ENTROPY_THRESHOLD = 1.5
HALF_TILE_LENGTH = 100
