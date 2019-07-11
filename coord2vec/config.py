import os

tile_server_ip = "localhost"
postgis_server_ip = "127.0.0.1"

tile_server_dns_noport = 'http://' + tile_server_ip + ':{p}/tile/{z}/{x}/{y}.png'
postgis_port = 15432
CACHE_DIR = os.path.join(os.path.dirname(__file__), "train_cache", "house_price")
TEST_CACHE_DIR = os.path.join(os.path.dirname(__file__), "train_cache", "test_directory")
SAMPLE_NUM = 10_000
IMG_WIDTH, IMG_HEIGHT = (224, 224)
israel_range = [34.482724,31.492354,34.583301,31.585196]
ENTROPY_THRESHOLD = 0.7
