
import numpy as np
from scipy.sparse import csc_matrix

from collections import defaultdict
import os
import shutil
import operator
import warnings
import scipy.ndimage as nd




def fill_contour(contour_arr):
    # get initial pixel positions
    pixel_positions = np.array([(i, j) for i, j in zip(np.where(contour_arr)[0], np.where(contour_arr)[1])])

    # LEFT TO RIGHT SCAN
    row_pixels = defaultdict(list)
    for i, j in pixel_positions:
        row_pixels[i].append((i, j))

    for i in row_pixels:
        pixels = row_pixels[i]
        j_pos = [j for i, j in pixels]
        for j in range(min(j_pos), max(j_pos)):
            row_pixels[i].append((i, j))
    pixels = []
    for k in row_pixels:
        pix = row_pixels[k]
        pixels.append(pix)
    pixels = list(set([val for sublist in pixels for val in sublist]))

    rows, cols = zip(*pixels)
    contour_arr[rows, cols] = 1

    # TOP TO BOTTOM SCAN
    pixel_positions = pixels  # new positions added
    row_pixels = defaultdict(list)
    for i, j in pixel_positions:
        row_pixels[j].append((i, j))

    for j in row_pixels:
        pixels = row_pixels[j]
        i_pos = [i for i, j in pixels]
        for i in range(min(i_pos), max(i_pos)):
            row_pixels[j].append((i, j))
    pixels = []
    for k in row_pixels:
        pix = row_pixels[k]
        pixels.append(pix)
    pixels = list(set([val for sublist in pixels for val in sublist]))
    rows, cols = zip(*pixels)
    contour_arr[rows, cols] = 1
    return contour_arr



