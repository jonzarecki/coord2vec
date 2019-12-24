import json
import os

global BASELINES

BASELINES = ['deepwalk', 'nmf', 'embedding']
REGION_ENCODER_DIR_PATH = os.path.dirname(__file__)



def get_config():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    embed_path = '/media/yonatanz/yz/region-encoder/embedding/'
    data_dir = '/media/yonatanz/yz/data/region-encoder-data/nyc/'
    config_path = '/media/yonatanz/yz/data/region-encoder-data/nyc-config.json'

    if not os.path.exists(embed_path):
        os.makedirs(embed_path)

    with open(config_path, 'r') as f:
        config = json.load(f)
    config['data_dir_main'] = data_dir
    for key, val in config.items():
        if 'file' in key:
            #for model_name in BASELINES:
            #    if model_name in key:
            #        config[key] = embed_path + val
            #        break
            #    else:
            #        config[key] = config['data_dir_main'] + val
            config[key] = config['data_dir_main'] + val

        if 'lat' in key or 'lon' in key:
            config[key] = float(val)

        if key == 'grid_size':
            config[key] = int(val)

    return config


if __name__ == '__main__':

    c = get_config()
    print(c)