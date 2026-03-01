# Sentinel-2 Data Processing Algorithms

Algorithm uses EODATA S3 access to download Sentinel-2 data and process the data to selected indicies.

## REQUIREMENTS
Ensure the following Python packages are installed: argparse, pandas, numpy, rasterio, baselib and sentinel_tools_lib from libs folder.

## USAGE
python main.py --download [path/to/downloads/folder] --awsprofile [EODATA AWS S3 profile name] --maxcc [int max allowed cloud coverage in percent (DEFAULT 60)]