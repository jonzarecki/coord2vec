from itertools import product

import numpy as np
import pandas as pd
from shapely.geometry import Point


class GeoConvolution:
    def __init__(self):
        pass

    def _image2tiles(self, image: np.ndarray, tile_size) -> np.ndarray:
        """
        Crop a big image into many small tiles

        Args:
            image: the big image to crop
            tile_size: The height and width of the small tiles

        Returns:
            a 5d numpy array with all the tiles [n_row, n_col, width, height, n_channels]
        """
        # better way
        cropped_image = image[:image.shape[0] - image.shape[0] % tile_size,
                        :image.shape[1] - image.shape[1] % tile_size]
        n_rows, n_cols = int(cropped_image.shape[0] / tile_size), int(cropped_image.shape[1] / tile_size)
        n_channels = image.shape[-1]
        tiles = cropped_image.reshape(n_rows, n_cols, tile_size, tile_size, n_channels)
        return tiles

    def _tiles2image(self, tiles: np.ndarray) -> np.ndarray:
        """
        Join many tiles back into one image
        Args:
            tiles:

        Returns:

        """
        n_row, n_col = tiles.shape[0], tiles.shape[1]
        tile_size = tiles.shape[2]
        n_channels = tiles.shape[-1]
        image = tiles.reshape((n_row * tile_size, n_col * tile_size, n_channels))
        return image

    def image2points(self, image: np.ndarray, bottom_left_point: Point, top_right_point: Point) -> pd.Series:
        min_long, min_lat = bottom_left_point.coords[0]
        max_long, max_lat = top_right_point.coords[0]
        longs = np.linspace(min_long, max_long, image.shape[1])
        lats = np.linspace(min_lat, max_lat, image.shape[0])
        points = [Point(long, lat) for long, lat in product(longs, lats)]
        values = image.flatten()
        series = pd.Series(index=points, data=values)
        return series
