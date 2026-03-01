from spectral import specim
import os
from spectral import index
import sys
import numpy as np
import multiprocessing as mp
from .mapfiles import main as Map

os.environ["GDAL_TIFF_INTERNAL_MASK"] = "True"


OUT_RGB_SUFFIX = "_rgb"
OUT_NDVI_SUFFIX = "_ndvi"
OUT_NDVI_RGB_SUFFIX = "_ndvi_rgb"


def pipeline(data_cube, rewrite=False):
    """
    Cube processing pipeline

    Parameters
        data_cube: str
            Path to input data cube (rect or radiance).
    """
    base_filename = os.path.splitext(data_cube)[0]
    ext = ".tiff" if "rect.dat" in data_cube else ".png"
    rgb_out = base_filename + OUT_RGB_SUFFIX + ext
    ndvi_out = base_filename + OUT_NDVI_SUFFIX + ext
    ndvi_rgb_out = base_filename + OUT_NDVI_RGB_SUFFIX + ext
    if os.path.exists(rgb_out) and os.path.exists(ndvi_out) and os.path.exists(ndvi_rgb_out):  # skip processing if all files exist
        if not rewrite:  # and not set to rewrite
            return False, rgb_out, ndvi_out, ndvi_rgb_out
    
    # main cube reader
    scr = specim.SpecimCubeReader(data_cube)
    # data loaders
    full_loader = specim.SpecimFullLoader(scr.metadata)

    # data writer
    full_image = specim.SpecimFullImageWriter(scr.metadata)

    # writer
    tiff_writer = specim.SpecimFullTiffWriter(scr.metadata)

    # processor
    rgb = specim.RGBProcessor(scr.wavelength)
    extractor = specim.SpecimBandExtractor(scr.metadata, wavelength_numbers=[670, 800])
    ndvi = specim.NDVIProcessor()
    scaler = specim.MinMaxScaler()
    scaler_0 = specim.MinMaxScaler(forced_min=0)
    converter = specim.FloatUint8Converter()
    masker = specim.MaskProcessor()
    painter = index.Painter(index.color_pallets["ndvi"][0], index.color_pallets["ndvi"][1], index.color_pallets["ndvi"][2])

    data = full_loader.data
    rgb_data = rgb.process(data)
    ndvi_bands = extractor.process(data)
    mask = masker.process(data)
    del data
    if ext == ".tiff":  # if georectified file, create tiffs
        rgb_data = scaler_0.process(rgb_data)
        rgb_data = converter.process(rgb_data)
        tiff_writer.write(rgb_data, {"full_path": rgb_out}, mask=mask)

        ndvi_bands = scaler.process(ndvi_bands)
        ndvi_data = ndvi.process(ndvi_bands[..., 0], ndvi_bands[..., 1], 0)
        tiff_writer.write(ndvi_data, {"full_path": ndvi_out}, mask=mask)

        ndvi_data = scaler.process(ndvi_data)
        ndvi_data = painter.process(ndvi_data)
        ndvi_data = converter.process(ndvi_data)
        tiff_writer.write(ndvi_data, {"full_path": ndvi_rgb_out}, mask=mask)
    else:  # else create png files
        rgb_data = scaler_0.process(rgb_data)
        rgb_data = converter.process(rgb_data)
        full_image.write(rgb_data, {"full_path": rgb_out})

        ndvi_bands = scaler.process(ndvi_bands)
        ndvi_data = ndvi.process(ndvi_bands[..., 0], ndvi_bands[..., 1], 0)
        ndvi_data = np.nan_to_num(ndvi_data)
        ndvi_data = scaler.process(ndvi_data)
        ndvi_data_2 = converter.process(ndvi_data)
        full_image.write(np.repeat(ndvi_data_2, 3, axis=-1), {"full_path": ndvi_out})

        ndvi_data = painter.process(ndvi_data)
        ndvi_data = converter.process(ndvi_data)
        full_image.write(ndvi_data, {"full_path": ndvi_rgb_out})
    return True, rgb_out, ndvi_out, ndvi_rgb_out


def main(params):
    """
    Hyper processing pipeline main function

    Parameters
        root_folder: str
            Path to the folder created as the output from specim processing scripts, containing _radiance and _rect data cubes.
    """
    root_folder = params["folder"]
    rewrite = params["rewrite"]
    filenames = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if os.path.isfile(os.path.join(root_folder, f))]
    process_files = []
    for fl in filenames:
        if "rect.dat" in fl or "radiance.dat" in fl:
            process_files.append(fl)
    return_vals = []
    for pr in process_files:
        res = pipeline(pr, rewrite)
        return_vals.extend(res)
    print(return_vals)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder created as the output from specim processing scripts, containing _radiance and _rect data cubes.')
    parser.add_argument('--rewrite', action='store_true', help='Enable existing file overwriting')

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(vars(opt))
