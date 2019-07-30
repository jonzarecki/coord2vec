import os

COORD2VEC_DIR_PATH = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(COORD2VEC_DIR_PATH)

tile_server_ip = "localhost"
postgis_server_ip = "127.0.0.1"

tile_server_dns_noport = 'http://' + tile_server_ip + ':{p}/tile/{z}/{x}/{y}.png'
tile_server_ports = [8101, 8102, 8103]
postgis_port = 15432

# CACHE_DIR = os.path.join(COORD2VEC_DIR_PATH, "train_cache", "house_price_builder")
CACHE_DIR = '/tmp/train_cache/house_price_builder'
TRAIN_CACHE_DIR = os.path.join(CACHE_DIR, 'train')
VAL_CACHE_DIR = os.path.join(CACHE_DIR, 'validation')

TEST_CACHE_DIR = os.path.join(COORD2VEC_DIR_PATH, "train_cache", "test_dir")
TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, "tensorboard_runs")

VAL_SAMPLE_NUM = 5_000
TRAIN_SAMPLE_NUM = 50_000

IMG_WIDTH, IMG_HEIGHT = (224, 224)
israel_range = [31.492354, 34.482724, 31.585196, 34.583301]
ENTROPY_THRESHOLD = 1.5
HALF_TILE_LENGTH = 50
