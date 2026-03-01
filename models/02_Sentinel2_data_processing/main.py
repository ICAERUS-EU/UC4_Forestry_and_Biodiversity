import os
from sentinel_tools.sentinel import DataCollection, MultitileDataset, MergedProcessing, MergedDataset
from sentinel_tools.base import Geometry
import numpy as np
import datetime
import pandas as pd
import shapely
import argparse


def bbox_to_geom():
    geom = shapely.box(*BBox)
    geom = Geometry(geom)
    return geom


def Compute(start_date, end_date, geom, maxcc, bands, indices, border, download_path, SAT, SYS, main_epsg, tiles_filter, profile):
    # Sentinel 2 search
    # Data download
    col = DataCollection(SAT, SYS, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), geom, bands, maxcc=maxcc, verbose=verbose, s3_client_profile=profile)
    col.search(tile_filter=tiles_filter)
    col.download(download_path)
    print("Download completed")

    # Combine tiffs
    md = MultitileDataset(col, main_epsg, separate_bands=True, verbose=verbose)
    md.combine_tiffs()

    md.export_tiffs()

    md.merge_all(border)
    print("Merge copleted")
    
    # compute index
    mgd = MergedDataset(md, verbose=verbose)

    if "SCL" in indices:  # save SCL as raw data, not mask
        mgd.combine_data(save_scl=True)
    else:
        mgd.combine_data(save_scl=False)

    # process to index
    mp = MergedProcessing(mgd, read_window=6000)

    if "NDVI" in indices:
        ndvi_path = mp.NDVI()

    if "SCL" in indices:
        scl_path = mp.SCL()

    if "NBRP" in indices:
        nbrp_path = mp.NBRP()

    if "MSI" in indices:
        msi_path = mp.MSI()


def Main(params):
    # description of what Sentinel-2 bands are required for indecies.
    index_bands = {'SCL': ['R20m/SCL'], 'MSAVI': ['R10m/B04', 'R20m/B8A'], 'LAI': ['R10m/B04', 'R10m/B08'], 'LAI3': ['R10m/B03', 'R10m/B02'],
                'NDVI': ['R10m/B04', 'R10m/B08'], 'EVI': ['R10m/B02', 'R10m/B04', 'R10m/B08'], 'MCARI':['R10m/B03', 'R10m/B04', 'R20m/B05'],
                'SOC': ['R20m/B05', 'R20m/B06', 'R20m/B11', 'R20m/B12', 'R10m/B02', 'R10m/B03', 'R10m/B04', 'R10m/B08'],
                'LAI2': ['R10m/B02', 'R10m/B03', 'R10m/B04', 'R20m/B05', 'R20m/B06', 'R20m/B11', 'R20m/B12','R20m/B07', 'R20m/B8A'],
                'MSI': ['R10m/B08', 'R20m/B11'], 'NBRP': ['R10m/B02', 'R10m/B03', 'R20m/B8A', 'R20m/B12']} 

    # LT bounding box for EPSG:4326 (required for Sentinel search)
    BBox = [20.8745, 53.8968, 26.8355, 56.4504]
    
    # Set as main srs to use
    main_epsg = 32634
    main_srs = f'EPSG:{str(main_epsg)}'

    # base 
    SAT = "Sentinel-2"
    SYS = "AWS"
    maxcc = params["maxcc"]
    profile = params["awsprofile"]
    
    # required indicies
    indicies = ['NBRP', 'NDVI', 'MSI', 'SCL']
    
    # when to get sentinel data
    Season_start_month = 4
    
    Season_end_month = 9 # end of month 8
    
    years = [2020, 2021, 2022, 2023, 2024, 2025]

    root_folder = params["download"]
    # root_folder = "/mnt/8TB/01-Strukturuotas/ICAERUS/Sentinel"
    download_path = os.path.join(root_folder, "downloads")
    
    # tiles that are needed for the country, filtered any others
    LT_tiles_filter = ['T34UEG', 'T35ULV', 'T34VFH', 'T34VEH', 'T35UMB', 'T34UFF', 'T34UFG', 'T34UFE', 'T35ULB', 'T35VLC', 'T35ULA', 'T35UMA', 'T35VMC']

    # provide country border geojson
    BORDER = "LT.geojson"
    
    verbose = False

    ## Init parameters

    # gather bands
    bands = []
    for index in indicies:
        bands.extend(index_bands[index])
    bands = list(set(bands))
    # bbox to Geometry
    geom = bbox_to_geom()

    ## Run Data download and processing. Creates selected indicies from Sentinel-2 data, combined over the dates selected

    for i, y in enumerate(years):
        dates = pd.date_range(f"{y}-{Season_start_month}-01", f"{y}-{Season_end_month}-01", periods=10)  # more periods = fewer dates in one batch, longer processing, more data, less memory intensive.
    
        for l in range(len(dates)-1):
            download_path = os.path.join(root_folder, f"downloads_{i}_{l}")
            print(dates[l].date(), dates[l+1].date(), download_path)
            Compute(dates[l].date(), dates[l+1].date(), geom, maxcc, bands, indicies, BORDER, download_path, SAT, SYS, main_epsg, LT_tiles_filter, profile)

    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxcc', type=int, default=60, help='percentage of allowed clouds in Sentinel-2 images')
    parser.add_argument('--download', type=str, default="/tmp/ICAERUS/", help='root download folder (absolute path)')
    parser.add_argument('--awsprofile', type=str, default="eodata", help='Profile to use for AWS data search and downloading. Placed in users .aws/credentials file')  # https://documentation.dataspace.copernicus.eu/APIs/S3.html
    """
    [eodata]
    aws_access_key_id = *********************
    aws_secret_access_key = *******************************
    endpoint_url = https://eodata.dataspace.copernicus.eu
    region_name = default
    """

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    opt = vars(opt)
    Main(opt)