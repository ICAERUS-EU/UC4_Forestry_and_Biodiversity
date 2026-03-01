import os
from matplotlib import pyplot as plt
import numpy as np
import datetime
import pandas as pd
import shapely
import glob
import rasterio
from rasterio.windows import Window
import subprocess
import shlex


def window_read(fl, profile, read_window=6000):
    wd = read_window
    ws = np.linspace(0, profile["width"], np.round(profile["width"]/wd).astype(int)).astype(int)
    hs = np.linspace(0, profile["height"], np.round(profile["height"]/wd).astype(int)).astype(int)
    msk = None
    for x in range(1, len(ws)):
        part = None
        for y in range(1, len(hs)):
            with rasterio.open(fl) as f:
                data = f.read(window=Window(ws[x-1], hs[y-1], ws[x] - ws[x-1], hs[y] - hs[y-1]))
            yield data.squeeze(), x, y

    
def compute_averages(files, read_window=6000):
    with rasterio.open(files[0]) as f:
        profile = f.profile
    avg = np.zeros((profile["height"], profile["width"]), dtype="float32")
    data = []
    for fl in files:
        data.append(window_read(fl, profile, read_window))

    ws = np.linspace(0, profile["width"], np.round(profile["width"]/read_window).astype(int)).astype(int)
    hs = np.linspace(0, profile["height"], np.round(profile["height"]/read_window).astype(int)).astype(int)
    # print(self.mgd._processed_files)
    while True:
        try:
            vals = None
            for dt in data:
                val, x, y = next(dt)
                if vals is None:
                    vals = val[:, :, np.newaxis]
                else:
                    if val.shape[0] != vals.shape[0] or val.shape[1] != vals.shape[1]:
                        val = np.resize(val, [vals.shape[0], vals.shape[1]])
                    vals = np.append(vals, val[:, :, np.newaxis], axis=2)
            masked_data = np.ma.masked_array(vals, np.isnan(vals))
            average = np.ma.average(masked_data, axis=2)
            # nbr[hs[y-1]:hs[y], ws[x-1]:ws[x]] = (b12_data - b8a_data - b3_data - b2_data) / (b12_data + b8a_data + b3_data + b2_data)
            avg[hs[y-1]:hs[y], ws[x-1]:ws[x]] = average
        except StopIteration:
            print("stop iter")
            break
    return avg


def clip_tiff(in_file, out_dir, out_filename, path_to_shp, epsgs=[32634, 32635]):
    files = []
    for ep in epsgs:
        full_filename = f"{out_filename}_{str(ep)}.tif"
        out_file = os.path.join(out_dir, full_filename)
        files.append(out_file)
        cmd = f'gdalwarp -t_srs EPSG:{str(ep)} -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -crop_to_cutline -cutline {path_to_shp} {in_file} {out_file} '
        cmd += f'-co compress=lzw -co predictor=2 '
        proc = subprocess.Popen(shlex.split(cmd))
        print(f"Cropping tiff using shapefile {path_to_shp} to file {out_file}")
        proc.wait()
    return files


def upscale(inpath, outpath, sizex, sizey, block_size=128):
    cmd = f'gdal_translate -of GTiff -outsize {sizex} {sizey} -r near -co BLOCKXSIZE={block_size} -co BLOCKYSIZE={block_size} -co TILED=YES -co NUM_THREADS=ALL_CPUS '
    # if compress:
    #     cmd += f'-co compress=lzw -co predictor=2 '
    cmd += f'{inpath} {outpath}'
    proc = subprocess.Popen(shlex.split(cmd))
    print(f"Upscaling {inpath} tiff to file: {outpath}")
    proc.wait()

    
def roi_filter(clc, data, classes=[24]):  # 24 - Coniferous forests
    classes = np.array(classes)
    if clc.shape[0] != data.shape[0] or clc.shape[1] != data.shape[1]:
        clc = np.resize(clc, data.shape)
    data[clc != classes] = 0
    return data


def Main(params):
    # process data form selected years (need to be the same year-index pair as download (2020 is index 0))
    years = [2020, 2021, 2022, 2023, 2024, 2025]

    # indecies to compute
    index = ["NDVI", "NBRP", "MSI"]

    # root folder where data was downloaded
    root_folder = params["download"]
    
    # https://land.copernicus.eu/en/products/corine-land-cover/clc2018#download
    # Download CLC tiff file and point to it
    clc_tiff_path = params["clcpath"]
    clc_out_dir = params["clcdir"]
    clc_tiff_path = "/mnt/8TB/01-Strukturuotas/Satellite/55321/Results/u2018_clc2018_v2020_20u1_raster100m/DATA/U2018_CLC2018_V2020_20u1.tif"
    clc_out_dir = "/mnt/8TB/01-Strukturuotas/Satellite/55321/Results/u2018_clc2018_v2020_20u1_raster100m/DATA/"  
    clc_out_filename = "LT"  # no extension
    path_to_shp = "LT.geojson"

    # CLC process
    # 
    clc_files = clip_tiff(clc_tiff_path, clc_out_dir, clc_out_filename, path_to_shp)
    clc_files

    for i, y in enumerate(years):
        print(y)
        path_filter = f"downloads_{i}"
        for index_file in index:
            agg_file = os.path.join(root_folder, f"{y}_{index_file}.tiff")
        
            # Get profile data and scale CLC
            with rasterio.open(agg_file) as f:
                profile = f.profile
                profile["count"] = 1
                epsg = profile["crs"].to_epsg()
                sizex = profile["width"]
                sizey = profile["height"]
                for clc in clc_files:
                    if str(epsg) in clc:
                        clc_in = clc
                        break
                clc_out = clc_in.split(".")[0] + "_scaled.tif"
                try:  # recreate clc if the size of current file differ from previuos
                    os.remove(clc_out)
                except:
                    pass
                upscale(clc_in, clc_out, sizex, sizey)
            # read CLC
            with rasterio.open(clc_out) as f:
                clc_data = f.read().squeeze()
            print(clc_data.shape)
    
            # read aggregate
            with rasterio.open(agg_file) as f:
                agg_data = f.read().squeeze()
    
            # Gather files
            ndvi = []
            for fl in glob.glob(root_folder + '/**/*.tiff', recursive=True):
                if index_file in fl and path_filter in fl:
                    ndvi.append(fl)
            out_folder = os.path.join(root_folder, f"results_{y}")
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            for x, in_file in enumerate(ndvi):
                with rasterio.open(in_file) as f:
                    index_data = f.read().squeeze()
    
                res_file = os.path.join(out_folder, f"{index_file}_{x}")
                
                res = roi_filter(clc_data, index_data)
                if res.shape[0] != agg_data.shape[1] or res.shape[1] != agg_data.shape[2]:
                    res = np.resize(res, [agg_data.shape[1], agg_data.shape[2]])
                res = (res - agg_data[0, :, :])/agg_data[1, :, :]
                with rasterio.open(res_file, "w", **profile) as f:
                    f.write(res[np.newaxis, :, :])
        

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', type=str, default="/tmp/ICAERUS/", help='root download folder (absolute path)')
    parser.add_argument('--clcdir', type=str, default="/tmp/ICAERUS/", help='root folder where CLC data is stored') # any path to store clipped tiff (new tiff path cannot be the same as input)
    parser.add_argument('--clcpath', type=str, default="/tmp/ICAERUS/", help='path to main CLC tiff file')

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    opt = vars(opt)
    Main(opt)