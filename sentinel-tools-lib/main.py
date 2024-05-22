from datetime import datetime, timedelta
import fiona
import geojson
import sentinelhub
import shapely
from sentinelhub.geometry import BBox
from sentinelhub.constants import CRS
from sentinel_tools.downloader import download
from sentinel_tools.gis_utils import point_to_buffer, flip_coords 
from pathlib import Path
import pandas as pd
import math
import warnings
import os
import rasterio
import numpy as np
from sentinel_tools.processors import CombineProcessor, CropProcessor, RGBCropProcessor, RGBProcessor, TransposeProcessor, U16U8Processor
from sentinel_tools.readers import RGB_BANDS, TileDownloader, TileReader
from sentinel_tools.writers import FullTiffWriter, RGBWriter
from rasterio.crs import CRS as rastercrs

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).parent


def download_test():
    bbox = BBox(bbox=(46.16, -16.15, 46.51, -15.58), crs=CRS.WGS84)
    ts_from, ts_to = (datetime(2022, 11, 11), datetime(2023, 1, 1))
    output_path = BASE_DIR / "downloads"
    bands = ["R10m/B02"]

    download(bbox, ts_from, ts_to, 1, bands, output_path)


def write_points_to_geojson():
    df = pd.read_csv("points.csv")
    average_area = df["damaged_area_m2"].mean()
    df["damaged_area_m2"] = df["damaged_area_m2"].fillna(average_area)

    feature_list = []
    for idx, row in df.iterrows():
        action, area_m2, lat, lon = row["action"], row["damaged_area_m2"], row["latitude"], row["longitude"]
        radius_meters = math.sqrt(area_m2 / math.pi)
#
        point = shapely.Point(lat, lon)
        buffer = point_to_buffer(point, sentinelhub.CRS.WGS84, radius_meters)

        point, buffer = flip_coords(point), flip_coords(buffer)
        properties = {
            "idx": idx,
            "area_m2": area_m2,
            "action": action
        }

        feature_list.append(geojson.Feature(geometry=point, properties=properties))
        feature_list.append(geojson.Feature(geometry=buffer, properties=properties))

    feature_collection = geojson.FeatureCollection(feature_list)
    with open("buffer.geojson", "w") as f:
        geojson.dump(feature_collection, f, indent=4)


def download_point():
    df = pd.read_csv("points.csv")
    df = df[~(df["date_entered"].isna() | df["date_edited"].isna() | df["damaged_area_m2"].isna())]

    df = df[df["action"] == "Sutvarkytas/senas židinys"]
    area_dict = df.loc[df["damaged_area_m2"].idxmax()].to_dict()

    half_year = 30 * 6
    ts_edited = datetime.utcfromtimestamp(area_dict["date_edited"])
    ts_from = ts_edited - timedelta(days=half_year)
    ts_to = ts_edited + timedelta(days=half_year)

    radius_meters = math.sqrt(area_dict["damaged_area_m2"] / math.pi)
    crs = sentinelhub.CRS.WGS84

    point = shapely.Point(area_dict["latitude"], area_dict["longitude"])
    buffer = point_to_buffer(point, sentinelhub.CRS.WGS84, radius_meters)
    buffer = flip_coords(buffer)

    rgb_bands = ["R10m/B04", "R10m/B03", "R10m/B02"]
    buffer_bbox = BBox(buffer.bounds, crs)
    download(buffer_bbox, ts_from, ts_to, 0.5, rgb_bands, BASE_DIR / "downloads")


def nomeda():
    l1_bbox = BBox(bbox=(25.0526359565603016,54.8496984881837975, 25.0548697606231983,54.8515856556220029), crs=CRS.WGS84)  # laukas 05-17
    l2_bbox = BBox(bbox=(23.9193907334730014,56.3049333764644970, 23.9236158737514018,56.3068836438937979), crs=CRS.WGS84)  # laukas 05-23
    l3_bbox = BBox(bbox=(23.7768040075347002,55.3536944039378014, 23.7794595605945993,55.3552520680298983), crs=CRS.WGS84)  # laukas 06-06
    l4_bbox = BBox(bbox=(23.9058018310370990,56.3199664199120988, 23.9076940255555002,56.3217951238555017), crs=CRS.WGS84)  # laukas 09-18
    l5_bbox = BBox(bbox=(24.1963827018501014,56.0162234579467011, 24.1975854174350999,56.0177776003915966), crs=CRS.WGS84)  # laukas 10-11
    rgb_bands = ["R10m/B04", "R10m/B03", "R10m/B02", "R20m/SCL"]
    ts_from = datetime(2023, 4, 1) 
    ts_to = datetime(2023, 9, 1)
    output_path_1 = BASE_DIR / "downloads_l1"
    output_path_2 = BASE_DIR / "downloads_l2"
    output_path_3 = BASE_DIR / "downloads_l3"
    output_path_4 = BASE_DIR / "downloads_l4"
    output_path_5 = BASE_DIR / "downloads_l5"

    #download(l1_bbox,  output_path_1, ts_from, ts_to, 0.5, rgb_bands)
    #download(l2_bbox,  output_path_2, ts_from, ts_to, 0.5, rgb_bands)
    #download(l3_bbox,  output_path_3, ts_from, ts_to, 0.5, rgb_bands)
    download(l4_bbox,  output_path_4, ts_from, ts_to, 0.5, rgb_bands)
    #download(l5_bbox,  output_path_5, ts_from, ts_to, 0.5, rgb_bands)

    for fld in os.listdir(output_path_5):
        data = []
        data_path = output_path_5 / fld / "R10m"
        r = data_path / "B04.jp2"
        g = data_path / "B03.jp2"
        b = data_path / "B02.jp2"
        with rasterio.open(r) as src:
            data.append(src.read(1))
            prof = src.profile
        with rasterio.open(g) as src:
            data.append(src.read(1))
        with rasterio.open(b) as src:
            data.append(src.read(1))
        rgb_image = np.transpose(np.dstack(data), (2, 0, 1))
        rgb_image = rgb_image.astype(np.float32)
        for i in range(3):
            #p5 = np.percentile(rgb_image[i, :, :], 2)
            #p95 = np.percentile(rgb_image[i, :, :], 98)
            p5 = 200
            p95 = 5000
            rgb_image[i, :, :] = (rgb_image[i, :, :] - p5) / (p95 - p5)
            rgb_image[i, rgb_image[i, :, :] < 0] = 0 
            rgb_image[i, rgb_image[i, :, :] > 1] = 1 
            rgb_image[i, :, :] = rgb_image[i, :, :] * 255
        rgb_image = rgb_image.astype(np.uint8)
        prof["count"] = 3
        prof["dtype"] = "uint8"
        with rasterio.open(output_path_5 / fld / "RGB.tiff", "w", **prof) as dst:
            dst.write(rgb_image)


def main():
    #download_point()
    #exit()
    #nomeda()
    
    # lib pipeline testing
    with fiona.open("/home/vytas/ab.shp", "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        print(shapefile.crs)
    
    l1_bbox = BBox(bbox=(25.0526359565603016,54.8496984881837975, 25.0548697606231983,54.8515856556220029), crs=CRS.WGS84)  # laukas 05-17
    ts_from = datetime(2023, 4, 1) 
    ts_to = datetime(2023, 9, 1)
    maxcc = 0.5
    tr = TileReader(l1_bbox, ts_from, ts_to, maxcc)
    td = TileDownloader(out_dir="/tmp/test", bands=RGB_BANDS)
    tl = tr.metadata[0]
    cp = CombineProcessor()
    tp = TransposeProcessor()
    wr = FullTiffWriter("/tmp/test/test-crop.tiff")
    rr = RGBWriter("/tmp/test/test.jpg")
    crop = CropProcessor()
    rgbcrop = RGBCropProcessor()
    rgb = RGBProcessor()
    conv = U16U8Processor()
    td.load([tl])
    print(tl.data_dir)
    tl.download()
    print(tl.data[0].shape)
    data, profile = cp.process(tl, tl.profiles[0])
    data, profile = crop.process(data, profile, shapes)
    wr.write(data, profile)
    rgb_mat = rgb.process(tl)
    rgb_mat, profile = rgbcrop.process(rgb_mat, tl.profiles[0], shapes)
    rgb_mat = conv.process(rgb_mat)
    rr.write(rgb_mat)
    
    crs = rastercrs.from_string("EPSG:3005")
    print(crs.units_factor)

    return True

    files = [
        '35VMC,2023-03-17,0/R10m/B04.jp2',  # Red
        '35VMC,2023-03-17,0/R10m/B03.jp2',  # Green
        '35VMC,2023-03-17,0/R10m/B02.jp2',  # Blue
    ]

    # Initialize an empty list to store band data
    band_data = []

    # Read the band data
    for band_path in band_paths:
        with rasterio.open(band_path) as src:
            band_data.append(src.read(1))  # Read the first band (band index 1)

    # Create an RGB image
    rgb_image = np.dstack(band_data)

    # Scale the data to the 0-255 range if needed
    rgb_image = (rgb_image / rgb_image.max() * 255).astype(np.uint8)

    # Now you can use the 'rgb_image' as your RGB image


if __name__ == "__main__":
    main()
