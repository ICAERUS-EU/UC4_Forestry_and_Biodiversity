import numpy as np
from spectral.specim import SpecimCubeReader, SpecimFullLoader
import rasterio
from typing import Optional, List
import glob
import os
import argparse
import requests
import json
from shapely import geometry
from shapely.ops import transform
import pyproj
import rasterio.mask
import warnings
from matplotlib import cm
from PIL import Image

CALIB_VALUES = [None, 0.1, 0.15, 0.4, 0.05, 0.5, 0.94]

def gray_to_rgb(data):
    assert len(data.shape) == 2  # grayscale 2D image
    colors = cm.RdYlGn(np.linspace(0, 1, 256))[:, :3]
    colors *= 255
    colors = np.astype(colors, "uint8")
    data *= 255 
    data = np.astype(data, "uint8")
    res = np.zeros((data.shape[0], data.shape[1], 3), "uint8")
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            res[x, y, :] = colors[data[x, y], :]
    return res


def min_max(v):
    newv = (v - v.min(axis=0)) / (v.max(axis=0) - v.min(axis=0))
    return newv


def min_max_rev(v, vals):
    newv = v * (vals.max(axis=0) - vals.min(axis=0)) + vals.min(axis=0)
    return newv


def get_calibrations(calib_cube, cals, dark_tiff=None):
    with rasterio.open(calib_cube, "r") as fl:
        data = fl.read()
        data = np.transpose(data, (1,2,0))  # out -> X, Y, bands
        metadata = fl.profile
    
    calib = np.zeros((0, 224))
    calib_vals = []
    # gather calibration pixels found by the model and append to main array.
    for i in range(len(CALIB_VALUES)):
        if i == 0:
            continue
        dt = data[cals == i, :]  # pixels with class i and 224 bands
        calib = np.concatenate((calib, np.average(dt, axis=0)[np.newaxis, :]), axis=0)
        calib_vals.append(CALIB_VALUES[i])
    calib_vals = np.array(calib_vals)
    # add dark pixels if provided
    if dark_tiff is not None:
        calib = np.concatenate((calib, out_image[np.newaxis, :]), axis=0)
        calib_vals = np.concatenate((calib_vals, np.array([0])), axis=0)
    calib_lines = []
    for wl in range(calib.shape[1]):
        z = np.polyfit(calib[:, wl], calib_vals, 1)
        calib_lines.append(z.copy())
    calib_lines = np.array(calib_lines)
    # print(calib_lines.shape)
    del data
    return calib_lines


def get_dark(root_folder):
    # add dark pixels
    dark_tiff = os.path.join(root_folder, "Dark/raw-dark_fl1_20230830_140006/out/raw-dark_fl1_20230830_140006_radiance.dat")
    try:
        with rasterio.open(dark_tiff) as src:
            out_image = src.read()
        out_image = np.average(out_image, axis=(1,2)).T
        return out_image
    except:
        return None

        
def main(opts):
    """
    Hyper calibration pipeline runner
    """

    # Get parameters
    cals = np.load(opts["calib"])
    dark_tiff = get_dark(opts["folder"])
    calib_cube = opts["cube"]

    # get calibrations
    lines = get_calibrations(calib_cube, cals, dark_tiff)

    # gather tiff files
    # Gather files
    hyper_files = []
    for fl in glob.glob(root_folder + '/**/*.dat', recursive=True):
        hyper_files.append(fl)

    for fl in hyper_files:
        print(f"Processing file: {fl}")
        out_file = fl.split(".")[0] + "_calibrated.dat"
    
        # run model
        with rasterio.open(in_cube, "r") as fl:
            data = fl.read()
            data = np.transpose(data, (1,2,0)) 
            metadata = fl.profile
        data = data * lines[:, 0] + lines[:, 1]
        data = np.transpose(data, (2,0,1)) 
        with rasterio.open(out_file), "w", **metadata) as f:
            f.write(dat)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true', help='Enable existing file overwriting')
    parser.add_argument('--folder', type=str, required=True, help='Path to folder containing hyperspectral data.')
    parser.add_argument('--calib', type=str, required=True, help='Path to calibration model result .npy file.')
    parser.add_argument('--cube', type=str, required=True, help='Path to cube with the calibration plates.')
    return parser.parse_args()


if __name__ == "__main__":
    requests.packages.urllib3.disable_warnings()
    warnings.filterwarnings('ignore')
    opt = parse_opt()
    main(vars(opt))
