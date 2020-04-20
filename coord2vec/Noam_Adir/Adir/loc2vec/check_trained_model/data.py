import os
import numpy as np
from coord2vec.Noam_Adir.utils import render_tiles_from_coords, load_from_pickle_features





def load_tiles(coords):
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    RGB = 3

    TILES_NPY_PATH = 'loc_tiles.npy'
    LOC_TILES_DIR = 'loc_tiles'
    if os.path.exists(TILES_NPY_PATH):
        tiles = np.load(TILES_NPY_PATH)
        return tiles
    elif os.path.exists(LOC_TILES_DIR):
        tiles_lst = []
        filenames = os.listdir(LOC_TILES_DIR)
        for filename in filenames:
            tile = np.load(filename)
            tiles_lst.append(tile)
        tiles = np.stack(tiles_lst).reshape(len(filenames), IMG_WIDTH, IMG_HEIGHT, RGB)
        np.save(TILES_NPY_PATH, tiles)
    else:
        # os.mkdir(LOC_TILES_DIR)
        render_tiles_from_coords(coords, url_template=, img_width=128, img_height=128, save_path=LOC_TILES_DIR)
