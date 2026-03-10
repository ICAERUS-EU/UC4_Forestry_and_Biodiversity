# Hyperspectral Calibraion Plate Detection Model

## How to run

Install python packages from requirements.txt into virtualenv or other environment. Ensure the following Python packages are installed: argparse, sklearn, numpy, pytorch, baselib and hyperspectral-lib from libs folder

Run using:

`python3 calibration.py  --cube "path/to/hyperspectral/cube"`

Hyperspectral cube has to be envi compatible raster file. 


## Output/Results

Numpy integer array of the same size as image with each pixel having the predicted calibration plate class number.

Results image input:

![Results image input](result.png)