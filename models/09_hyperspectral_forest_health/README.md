REQUIREMENTS
Ensure the following Python packages are installed: argparse, pandas, matplotlib, numpy, tensorflow, rasterio

USAGE
python script.py <data_path> <a_coef_path> <b_coef_path> [--output OUTPUT]

Positional Arguments
 - data_path: Path to the raster data cube (GeoTIFF format).
 - a_coef_path: Path to the .npy file containing the a coefficient array (used for calibration).
 - b_coef_path: Path to the .npy file containing the b coefficient array (used for calibration).

Optional Arguments
--output: Path to save the output prediction plot image (default: predictions.png).

OUTPUT
 - predictions.npy: NumPy array file containing the full prediction results. Shape: (height, width, num_classes).
 - predictions.png (or custom filename): PNG image visualizing the predicted class for each pixel.

ADDITIONAL NOTES
 - Predictions are visualized as the most probable class for each pixel.
 - Make sure the dimensions of the raster data and coefficient arrays are compatible.
 - This script assumes a 3D input raster (bands, height, width), which it transposes and scales before prediction.
 - The model should be saved in the file forest_fuel.keras in the same directory as the script.
 - The paths to files should be given in "quotation marks".
