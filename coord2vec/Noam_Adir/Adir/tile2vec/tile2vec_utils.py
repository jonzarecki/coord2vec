import matplotlib.pyplot as plt
import numpy as np
from cv2 import equalizeHist
from mpl_toolkits.axes_grid1 import ImageGrid


def draw_circle(im, cx=112, cy=112, radius=2, fill=True):
    im = im.copy()
    y, x = np.ogrid[-radius: radius, -radius: radius]
    if fill:
        index = (x ** 2 + y ** 2 <= radius ** 2)
    else:
        index = (x ** 2 + y ** 2 <= radius ** 2) & (x ** 2 + y ** 2 >= radius ** 2 - 100)
    im[cy - radius:cy + radius, cx - radius:cx + radius][index] = np.array([255, 0, 0])
    # im[0:2] = np.array([255, 0, 0])
    # im[-2:300] = np.array([255, 0, 0])
    # im[:, 0:2] = np.array([255, 0, 0])
    # im[:, -2:300] = np.array([255, 0, 0])
    return im


def display_grid(im_lst, ncols, save_path=None):
    nrows = int(np.ceil(len(im_lst) / ncols))
    fig = plt.figure(figsize=(4 * ncols, 4 * nrows))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
                     axes_pad=0.08)
    for ax, im in zip(grid, im_lst):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis('off')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


C = np.array([[0.299, 0.587, 0.114],
              [0.596, -0.275, -0.321],
              [0.212, -0.523, 0.311]])


def equalize_hist(im):
    # tiles_to_show = list(map(equalizeHist, tiles_to_show))
    im = im / 255
    imb = im.dot(C.T)
    # plt.imshow(im, cmap='gray')
    # plt.show()
    imb[:, :, 0] = equalizeHist((imb[:, :, 0] * 255).astype(np.uint8)) / 255
    imc = np.clip((imb.dot(np.linalg.inv(C.T))), 0, 1)
    # plt.imshow(imc)
    # plt.show()
    return imc
