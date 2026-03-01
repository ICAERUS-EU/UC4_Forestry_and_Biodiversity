# Sentinel-2 Data ROI Processing Algorithms

Algorithm uses downloaded and processed Sentinel-2 data and CLC land classification data to create Sentinel datasets of selected region and land type

## REQUIREMENTS
Ensure the following Python packages are installed: argparse, pandas, numpy, rasterio, baselib and sentinel_tools_lib from libs folder.

## USAGE
python main.py --download [path/to/downloads/folder] --clcdir [path/where/to/place/new CLC/data] --clcpath [path/to/CLC/tiff/file]