# Drone Hyperspectral Data Processing Model and Pipelines

Algorithms to create RGB and NDVI images and mapfiles for mapserver from hyperspectral data cubes, for further analysis and displaying as raster map layers.

## REQUIREMENTS
Ensure the following Python packages are installed: argparse, shapely, pyproj, numpy, rasterio, baselib and hyperspectral-lib from libs folder

## USAGE
python main.py --folder [path/to/folder/wtih/hyperspectral/data] --rewrite (overwrite files or not)
python mapfiles.py --path [path/to/processed/hyperspectral/tiff/file] --outpath [path/to/place/mapfile]