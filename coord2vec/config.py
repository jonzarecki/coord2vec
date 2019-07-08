import os

tile_server_ip = "52.232.47.43"
postgis_server_ip = "127.0.0.1"

tile_server_dns_noport = 'http://' + tile_server_ip + ':{p}/tile/{z}/{x}/{y}.png'
postgis_port = 15432
CACHE_DIR = os.path.join(os.path.dirname(__file__), "train_cache", "building_count_only")
SAMPLE_NUM = 50_000
