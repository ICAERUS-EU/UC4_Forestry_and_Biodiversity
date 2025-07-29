import numpy as np
import rasterio
from spectral.cube import Cube
import os


def gather_cubes(folder, rad_only=True):
    cube_paths = []
    filter_string = ".dat"
    if rad_only:
        filter_string = "radiance.dat"
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(filter_string):
                cube_paths.append(os.path.join(root, file))
    return cube_paths


def bbox_cutter(cube: Cube, points: list):
    """
    Cut bbox from cube according to points list.
    carpet locations (3 classes: 1 - small dark carpet, 2 - large dark, 3 - gray) [x1, y1, x2, y2] bottom left, top right in QGIS.
    cube shape example [6510 (y), 1024 (x), 224 (l)]
    :param cube:
    :param points:
    :return:
    """
    cut_data = cube.data[points[3]:points[1], points[0]:points[2], :]
    return cut_data


def min_max_scaler(Y, miny=None, maxy=None):
    if miny is None:
        miny = Y.min()
        print("Min: ", miny)
    if maxy is None:
        maxy = Y.max()
        print("Max: ", maxy)
    return (Y - miny) / (maxy - miny)
