"""
Experiment to see if we can create a loc2vec_zip as detailed in the blogpost.
bloglink: https://www.sentiance.com/2018/05/03/venue-mapping/
"""
import random
from pathlib import Path
# from collections import OrderedDict
# import time
import pandas as pd
import numpy as np
import matplotlib
from PIL import Image
import torch
# from torch import nn, optim
from staticmap import StaticMap
from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
from torchvision import transforms

from coord2vec.Noam_Adir.Adir.loc2vec.lat2tile import deg2num
from coord2vec.image_extraction.tile_image import render_single_tile
from coord2vec.Noam_Adir.Adir.tile2vec.tile2vec_utils import display_grid, equalize_hist


def get_files_from_path(pathstring):
    """retrives file names from the folder and returns a pandas dataframe with
    four columns: path, filesize, lat, long

    Arguments:
        pathstring {string} -- relative location of file

    Returns:
        [pandas dataframe] -- sorted by the filesize
    """

    filenames = []
    for file in Path(pathstring).glob("**/*.png"):
        filenames.append((file, file.stat().st_size,
                          file.parts[-2], file.stem))
    files_df = pd.DataFrame(list(filenames),
                            columns=["path", "filesize", "x", "y"])
    sorted_files = files_df.sort_values("filesize")
    result_df = sorted_files.reset_index(drop=True)
    return result_df


def cleanse_files(df_files):
    """
    lets check filesizes and remove known useless tiles.

    103, 306, 355, 2165, 2146, 2128, 2202 are heavily
    represented and are typically grasslands/ empty / sea.

    Let's remove that from the samples!

    Arguments:
        df_files {pandas dataframe} -- should contain a column named "filesize"

    Returns:
        dataframe -- filtered dataframe with useless file sizes removed
    """

    filtered_files = df_files[(df_files["filesize"] != 103) &
                              (df_files["filesize"] != 306) &
                              (df_files["filesize"] != 355) &
                              (df_files["filesize"] != 2146) &
                              (df_files["filesize"] != 2128) &
                              (df_files["filesize"] != 2165) &
                              (df_files["filesize"] != 2202)]
    result = filtered_files.reset_index(drop=True)
    count = result.filesize.value_counts()
    freq = 1. / count
    freq_dict = freq.to_dict()
    result['frequency'] = result['filesize'].map(freq_dict)
    print(len(result))
    return result


class GeoTileDataset(Dataset):
    """
    A custom dataset to provide a batch of geotiles.
    """

    transform = None
    center_transform = None
    ten_crop = None
    pd = None

    def __init__(self, transform, center_transform):
        # for us west
        # zoom, self.startx, self.starty = deg2num(14, 48.995395, -124.866258)
        # zoom, self.endx, self.endy = deg2num(14, 31.74734, -104.052404)
        self.SAMPLE_NUM = 50_000

        self.m = StaticMap(500, 500, url_template='http://40.127.166.177:8103/tile/{z}/{x}/{y}.png')

        self.ten_crop = transforms.Compose([transforms.TenCrop(128)])
        self.transform = transform
        self.center_transform = center_transform

    def __getitem__(self, index):
        # url = 'http://a.tile.openstreetmap.us/usgs_large_scale/{z}/{x}/{y}.jpg'
        url = 'https://khms1.google.com/kh/v=865?x={x}&y={y}&z={z}'
        m = StaticMap(900, 900, url_template=url,
                      delay_between_retries=15, tile_request_timeout=5)

        # lon, lat = [40.802187, -73.957066]
        lon, lat = [40.807430, -74.003499]
        ext = [lon, lat, lon + 0.001, lat + 0.001]

        data = render_single_tile(m, ext)
        array = np.array(data)
        array = (255 * equalize_hist(array)).astype(np.uint8)
        # print(array.shape)
        data = Image.fromarray(array)
        matplotlib.image.imsave(r'C:\Users\adirdayan\OneDrive\Desktop\צבא\geoembedding\big_image2.png', array)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(data)
        plt.show()

        cropped_data = self.ten_crop(data)
        center_data_tensor = torch.stack([self.center_transform(data)
                                          for i in range(0, 10)], 0)
        ten_data = torch.stack([self.transform(x) for x in cropped_data], 0)
        twenty_data = torch.cat([center_data_tensor, ten_data], 0)
        display_grid([np.moveaxis(np.array(twenty_data), 1, -1)[i] for i in range(20)], 5, save_path=r'C:\Users\adirdayan\OneDrive\Desktop\צבא\geoembedding\cut_image2.png')
        return twenty_data

    def __len__(self):
        return self.SAMPLE_NUM


if __name__ == '__main__':
    size = 128 * 9 // 5
    anchor_transform = transforms.Compose([
        transforms.RandomAffine(degrees=90, translate=(0.25, 0.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.CenterCrop(size),
        transforms.Resize(size),
        transforms.ToTensor(),
        ])

    train_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        ])

    #  Let's use 12 while developing as it reduces the start time.
    g = GeoTileDataset(transform=train_transforms, center_transform=anchor_transform)
    g[0]
