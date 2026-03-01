# Raw Hyperspectral Data Processing Algorithms and Pipelines

Calibrate the hyperspectral cubes using the calibration model predictions and optional dark tiff image.

## REQUIREMENTS
Ensure the following Python packages are installed: argparse, shapely, pyproj, numpy, Pillow, Matplotlib, rasterio, baselib and hyperspectral-lib from libs folder

## USAGE
python main.py --folder [path/to/folder/wtih/hyperspectral/data] --overwrite (overwrite files or not) --calib [path/to/calibration/prediction/data.npy] --cube [path/to/cube/with/calibration/plates]
