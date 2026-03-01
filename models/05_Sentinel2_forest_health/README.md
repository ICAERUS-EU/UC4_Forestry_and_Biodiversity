# Sentinel-2 Forest Health Monitoring Model

COmpute FOrest health indicating indicies from generated Sentinel-2 datasets. By default NDVI, MSI and NBR indicies are computed and compared over the selected years (2020-2025).

## REQUIREMENTS
Ensure the following Python packages are installed: argparse, pandas, numpy, rasterio, baselib and sentinel_tools_lib from libs folder.

## USAGE
python main.py --download [path/to/downloads/folder] --clcdir [path/where/to/place/new CLC/data] --clcpath [path/to/CLC/tiff/file]