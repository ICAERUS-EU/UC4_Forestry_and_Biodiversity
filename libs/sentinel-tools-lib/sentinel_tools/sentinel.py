import os
from .base import Geometry
import ee
import requests
from typing import Literal
from datetime import datetime, date
import copy
from .constants import *
from requests.models import Response
from ee.imagecollection import ImageCollection
import geemap
import boto3
import copy
import rasterio
import time
from .utils import mem_raster, _upscaler
from .crs import CRS
import numpy as np
import shutil
import json
import gc
from rasterio.mask import mask
import hashlib
import subprocess
import shlex
from rasterio.windows import Window
from shapely.prepared import prep
from shapely.geometry import Point, mapping
import fiona
import xml.etree.ElementTree as ET
from .sentinel_lai import main as LAI_calc
import multiprocessing as mp
import pystac_client


def download_s3_folder(s3, bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)


def validate(date_text):
    try:
        date.fromisoformat(date_text)
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")


def process_data_ext(data, mask, use_float=False):
    assert len(data.shape) == 3, "Data needs to be 3D (bands, x, y)"
    assert len(mask.shape) == 2, "Mask needs to be 2D (x, y)"
    assert data.shape[1] == mask.shape[0], "Mask and data need to be the same size"
    assert data.shape[2] == mask.shape[1], "Mask and data need to be the same size"
    if use_float:
        out = np.zeros_like(mask, dtype="float32")
    else:
        out = np.zeros_like(mask, dtype="uint16")
    for x in np.arange(data.shape[1]):
        for y in np.arange(data.shape[2]):
            out[x, y] = data[mask[x, y], x, y]
    return out


def burn_shp_to_raster(in_tiff, out_tiff, shapefile, field, out_format="Float32"):  # gdal types: Byte, Int8, UInt16, Int16, UInt32, Int32, UInt64, Int64, Float32, Float64, CInt16, CInt32, CFloat32 or CFloat64.
    with rasterio.open(in_tiff) as f:
        prof = f.profile
    if prof["dtype"].lower() != out_format.lower():
        cmd = f'gdal_translate -ot {out_format} -co NUM_THREADS=ALL_CPUS -co compress=lzw -co predictor=2 {in_tiff} {out_tiff}'
        proc = subprocess.Popen(shlex.split(cmd))
        proc.wait()
    else:
        shutil.copyfile(in_tiff, out_tiff)
    cmd = f'gdal_rasterize -b 1 -a {field} {shapefile} {out_tiff}'
    proc = subprocess.Popen(shlex.split(cmd))
    print(f"Burning raster file to tiff {out_tiff}")
    proc.wait()
    return out_tiff


def geo_to_points(geom: Geometry, meters: float = 300):
    # https://www.sentinel-hub.com/faq/how-to-get-data-in-native-resolution-in-wgs84/
    assert geom.crs.epsg == 4326, "Geometry has to be in ESPG:4326"

    latmin, lonmin, latmax, lonmax = geom.geometry.bounds
    resolution_lat = meters / 111226.26 * 2
    resolution_lon = meters / 111226.26 / np.cos(int(latmax - latmin) * np.pi / 180)
    
    # create prepared polygon
    prep_polygon = prep(geom.geometry)
    
    # construct a rectangular mesh
    points = []
    valid_points = []
    for lat in np.arange(latmin, latmax, resolution_lat):
        for lon in np.arange(lonmin, lonmax, resolution_lon):
            points.append(Point((lat, lon)))
    
    print("filtering")
    # valid_points.extend(filter(prep_polygon.contains, points))
    # valid_points.extend([i for i in points if prep_polygon.contains(i)])
    points_array = []
    for vpt in points:
        points_array.append([vpt.x, vpt.y])
    points_array = np.array(points_array)
    return points_array, points


def parse_metadata(meta):
    # xml parseris kad istraukt saules kampus is metaduomenu failo.
    # meta (str): kelias i metaduomenu faila
    root = ET.parse(meta).getroot()
    branch = root[1]
    assert 'Geometric' in branch.tag
    angles = branch.find('Tile_Angles')
    grid = angles.find('Sun_Angles_Grid')
    zenith = grid.find('Zenith').find('Values_List')
    azimuth = grid.find('Azimuth').find('Values_List')
    grid_zenith = []
    for lst in list(zenith):
        grid_zenith.append(lst.text.split(' '))
    grid_zenith = np.array(grid_zenith).astype('float')
    if grid_zenith.shape[0] == grid_zenith.shape[1] and grid_zenith.shape[0] == 23:
        grid_zenith = grid_zenith[:-1, :-1]

    grid_azimuth = []
    for lst in list(azimuth):
        grid_azimuth.append(lst.text.split(' '))
    grid_azimuth = np.array(grid_azimuth).astype('float')
    if grid_azimuth.shape[0] == grid_azimuth.shape[1] and grid_azimuth.shape[0] == 23:
        grid_azimuth = grid_azimuth[:-1, :-1]

    return grid_zenith, grid_azimuth


def matrix_to_tiff(data, profile, pth, compress=True):
    if data.dtype == np.dtype("float"):  # reduce the float accuracy for image saving
        X = data.astype("float32")
    else:
        X = data
    if len(X.shape) == 2:
        X = X[np.newaxis, :, :]
    meta = profile
    if meta["width"] != X.shape[1]:
        scale = meta["width"]//X.shape[1]
        X = X.repeat(scale,axis=1).repeat(scale,axis=2)
        meta["width"] = X.shape[1]
        meta["height"] = X.shape[2]
    meta['count'] = X.shape[0]
    meta['dtype'] = X.dtype
    meta['driver'] = "GTiff"
    meta["interleave"] = "band"
    if compress:
        meta["compress"] = "lzw"
        meta["predictor"] = 2
    with rasterio.open(pth, "w", **meta) as f:
        f.write(X)


class Filter:
    def __init__(self, system: Literal["AWS", "EE"], start_date: str | None, end_date: str | None, geom: Geometry | None, maxcc: float = 100, sensorMode: str = "IW", orbit: str = "DESCENDING", productType: str = "GRD"):
        """
        Filter main class. Used to create a sentinel data search filter

        Creates Copernicus OpenSearch filter or Google EE filter.

        Parameters
        ----------
        system: str Literal
            Specify which system to use AWS or EE
        """
        self.system = system
        self.start_date = start_date
        self.end_date = end_date
        self.geom = geom
        self.maxcc = maxcc
        self.sensor = sensorMode
        self.orbit = orbit
        self.filters_s1 = None
        self.filters_s2 = None
        self.productType = productType
        self._check()

    def _check(self):
        assert self.start_date is not None or self.end_date is not None or self.geom is not None, "Filter creation error, at least one of start_date, end_date or geometry has to be provided"
        if self.start_date is not None:
            validate(self.start_date)
        if self.end_date is not None:
            validate(self.end_date)
        if self.sensor is not None:
            assert self.sensor in S1_INSTRUMENTS, "Filter creation error, wrong sentinel 1 instrument selected"
        if self.orbit is not None:
            assert self.orbit in ["ASCENDING", "DESCENDING"], f"Filter creation error, wrong orbit, needs to be ASCENDING or DESCENDING is {self.orbit}"

    def _filter_aws_stac(self, force=False):
        if self.filters_s1 is None or force is True:
            filters = {}
            if self.start_date is not None:
                filters["startDate"] = self.start_date
            if self.end_date is not None:
                filters["endDate"] = self.end_date
            if self.geom is not None:
                gm = copy.deepcopy(self.geom)
                filters["geom"] = gm.export_geojson(None)["features"][0]["geometry"]
            filters["sortParam"] = "datetime"
            filters["params"] = []
            self.filters_s1 = copy.deepcopy(filters)
            self.filters_s2 = copy.deepcopy(filters)

            # Sentinel 1 specific
            if self.sensor is not None:
                self.filters_s1["params"].append({"op": "=", "args": [{"property": "sar:instrument_mode"}, self.sensor]})
            if self.orbit is not None:
                self.filters_s1["params"].append({"op": "=", "args": [{"property": "sat:orbit_state"}, self.orbit.lower()]})
            if self.productType is not None:  #unused 
                pass
                # self.filters_s1 += "productType="+self.productType+"&"

            # Sentinel 2 specific
            if self.maxcc is not None:
                self.filters_s2["params"].append({"op": "<", "args": [{"property": "eo:cloud_cover"}, self.maxcc]})
        return self.filters_s1, self.filters_s2

    def _filter_aws(self, force=False):
        if self.filters_s1 is None or force is True:
            filters = ""
            if self.start_date is not None:
                df = date.fromisoformat(self.start_date).strftime(OPENSEARCH_TIME_FORMAT)
                filters += "startDate="+df + "&"
            if self.end_date is not None:
                df = date.fromisoformat(self.end_date).strftime(OPENSEARCH_TIME_FORMAT)
                filters += "completionDate="+df + "&"
            if self.geom is not None:
                bbox = [str(x) for x in self.geom.bbox]
                filters += "box="+",".join(bbox)+"&"
            filters += "sortParam=startDate&"
            self.filters_s1 = copy.deepcopy(filters)
            self.filters_s2 = copy.deepcopy(filters)

            # Sentinel 1 specific
            if self.sensor is not None:
                self.filters_s1 += "sensorMode="+self.sensor+"&"
            if self.orbit is not None:
                self.filters_s1 += "orbitDirection="+self.orbit+"&"
            if self.productType is not None:
                self.filters_s1 += "productType="+self.productType+"&"

            # Sentinel 2 specific
            if self.maxcc is not None:
                self.filters_s2 += f"cloudCover=[0,{self.maxcc}]&"

            self.filters_s2 += f"productType={S2_PROCESSING_LEVEL}&"
        if self.filters_s1[-1] == "&":
            self.filters_s1 = self.filters_s1[:-1]
        if self.filters_s2[-1] == "&":
            self.filters_s2 = self.filters_s2[:-1]
        return self.filters_s1, self.filters_s2

    def _filter_ee(self, force=False):
        if self.filters_s1 is None or force is True:
            filters = {}
            if self.start_date is not None:
                df = ee.Date(self.start_date)
                if self.end_date is not None:
                    df = ee.Filter.date(self.start_date, self.end_date)
                filters["date"] = df
            if self.geom is not None:
                filters["geometry"] = ee.Filter.bounds(ee.Geometry.Rectangle(self.geom.bbox))
            self.filters_s1 = copy.deepcopy(filters)
            self.filters_s2 = copy.deepcopy(filters)

            # Sentinel 1 specific
            if self.sensor is not None:
                self.filters_s1["sensor"] = ee.Filter.eq('instrumentMode', self.sensor)
            if self.orbit is not None:
                self.filters_s1["orbit"] = ee.Filter.eq('orbitProperties_pass', self.orbit)

            # Sentinel 2 specific
            if self.maxcc is not None:
                self.filters_s2["cloud"] = ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.maxcc)
        return self.filters_s1, self.filters_s2

    def filter(self, force=False, stac=False):
        """
        Creates and returns filter for AWS or EE.

        filters created: date, geometry, sensor (s1), cloud (s2 less than maxcc), orbit (s1)

        Parameters
        ----------
        force: bool
            Force recreation of filters, use if parameters changed
        """
        if self.system == "EE":
            return self._filter_ee(force)
        if self.system == "AWS":
            if stac:
                return self._filter_aws_stac(force)
            else:
                return self._filter_aws(force)
        raise Exception("Filter error unsuported system provided")


class Collection:
    """
    Sentinel data collection class with base metadata.
    """

    def __init__(self, satellite: Literal["Sentinel-1", "Sentinel-2"], bands: list | None, metadata_bands: list = []):
        self.satellite = satellite
        self.bands = bands
        self.metadata_bands = metadata_bands
        self.filters_aws = None
        self.filters_ee = None
        if self.bands is not None:
            if not isinstance(self.bands, list):
                self.bands = list(self.bands)

    def _filter_s1(self, force = False):
        self.filters_aws = {}
        self.filters_ee = {}
        if self.bands is not None:
            for i, bnd in enumerate(self.bands):
                self.filters_ee[f"bands_{i}"] = ee.Filter.listContains('transmitterReceiverPolarisation', bnd)
            # no Sentinel band filter for aws
        return self.filters_ee, self.filters_aws

    def _filter_s2(self, force = False):
        self.filters_aws = {}
        self.filters_ee = {}
            # no Sentinel band filter for aws
        return self.filters_ee, self.filters_aws

    def filter(self, force = False):
        if self.satellite == "Sentinel-1":
            return self._filter_s1(force)
        if self.satellite == "Sentinel-2":
            return self._filter_s2(force)
        raise Exception("Filter error unsuported satellite provided")


class Search:
    """
    Main OpenSearch and Earth Engine Sentinel data search.

    Parameters
    ----------
    filter: Filter
        filter class with set parameters.

    collection: Collection
        collection class with set parameters.
    """

    def __init__(self, filter: Filter, collection: Collection, max_results: int=20):
        self.filter = filter
        self.collection = collection
        self.max_results = max_results
        self.results = None
        self.max_pages = 20
        self.params = {}

    def save_results(self, path, force=False):
        if os.path.exists(path) and not force:
            return True
        if self.results is not None:
            if self.filter.system == "EE":
                print("writing search to file")
                with open(path, 'w') as fout:
                    json.dump(self.results.serialize(), fout)
            else:
                print("writing search to file")
                with open(path, 'w') as fout:
                    json.dump(self.results, fout)

    def read_results(self, path):
        if os.path.exists(path):
            if self.filter.system == "EE":
                with open(path) as f:
                    d = json.load(f)
                    self.results = ee.deserializer.fromJSON(d)
            else:
                with open(path) as f:
                    d = json.load(f)
                    self.results = d

    def _search_s1_EE(self):
        s1_base = ee.ImageCollection(S1_EE_SEARCH_COLLECTION)
        filters_s1, filters_s2 = self.filter.filter()
        for name, fls in filters_s1.items():
            s1_base = s1_base.filter(fls)
        ee_filters, aws_filters = self.collection.filter()
        for name, fls in ee_filters.items():
            s1_base = s1_base.filter(fls)
        size = s1_base.size().getInfo()
        self.results = s1_base.toList(size)

    def _search_s2_EE(self):
        s2_base = ee.ImageCollection(S2_EE_SEARCH_COLLECTION)
        filters_s1, filters_s2 = self.filter.filter()
        for name, fls in filters_s2.items():
            s2_base = s2_base.filter(fls)
        ee_filters, aws_filters = self.collection.filter()
        for name, fls in ee_filters.items():
            s2_base = s2_base.filter(fls)
        size = s2_base.size().getInfo()
        self.results = s2_base.toList(size)

    def _multipage_download(self):
        pagenum = 2
        while True:
            time.sleep(1)
            print(f"Searching page {pagenum}")
            new_url = copy.copy(self.search_url) + f"&page={pagenum}"
            while True:
                try:
                    res = requests.get(new_url, timeout=None)
                    res = res.json()["features"]
                    break
                except KeyError:
                    print("request key error")
                    print(res.text)
                    time.sleep(30)
                except requests.exceptions.JSONDecodeError:
                    print("Json error")
                    print(res.text)
                    time.sleep(30)
            if len(res) == 0:
                break
            if len(res) < 20:
                self.results.extend(res)
                break
            if len(res) == 20:
                self.results.extend(res)
                pagenum += 1
            if pagenum > self.max_pages:
                break

    def _stac_search_s1_AWS(self):
        search_url = EODATA_SEARCH_URL
        self.client = pystac_client.Client.open(search_url)
        self.client.add_conforms_to("ITEM_SEARCH")
        filters_s1, filters_s2 = self.filter.filter(stac=True)
        self.params = {
            "collections": S1_STAC_SEARCH_COLLECTION,
            "datetime": f"{filters_s1['startDate']}/{filters_s1['endDate']}",
            "intersects": filters_s1["geom"],
            "sortby": filters_s1["sortParam"],
            "fields": {"exclude": ["geometry"]},
            "filter": {"op": "AND", "args": filters_s1["params"]}
        }
        self.results = list(self.client.search(**self.params).items_as_dicts())

    def _search_s1_AWS(self):
        search_url = f"{OPENSEARCH_BASE_URL}{S1_OPENSEARCH_URL}"
        filters_s1, filters_s2 = self.filter.filter()
        self.search_url = search_url + filters_s1
        while True:
            try:
                self.results = requests.get(self.search_url, timeout=None)
                self.results = self.results.json()["features"]
                break
            except KeyError:
                print("request key error")
                print(self.results.text)
                time.sleep(30)
            except requests.exceptions.JSONDecodeError:
                print("Json error")
                print(self.results.text)
                time.sleep(30)
        if len(self.results) == 20:
            self._multipage_download()

    def _stac_search_s2_AWS(self):
        search_url = EODATA_SEARCH_URL
        self.client = pystac_client.Client.open(search_url)
        self.client.add_conforms_to("ITEM_SEARCH")
        filters_s1, filters_s2 = self.filter.filter(stac=True)
        self.params = {
            "collections": S2_STAC_SEARCH_COLLECTION,
            "datetime": f"{filters_s2['startDate']}/{filters_s2['endDate']}",
            "intersects": filters_s2["geom"],
            "sortby": filters_s2["sortParam"],
            "fields": {"exclude": ["geometry"]},
            "filter": {"op": "AND", "args": filters_s2["params"]}
        }
        self.results = list(self.client.search(**self.params).items_as_dicts())

    def _search_s2_AWS(self):
        search_url = f"{OPENSEARCH_BASE_URL}{S2_OPENSEARCH_URL}"
        filters_s1, filters_s2 = self.filter.filter()
        self.search_url = search_url + filters_s2
        print(self.search_url)
        while True:
            try:
                self.results = requests.get(self.search_url, timeout=None)
                self.results = self.results.json()["features"]
                break
            except KeyError:
                print("request key error")
                print(self.results.text)
                time.sleep(30)
            except requests.exceptions.JSONDecodeError:
                print("Json error")
                print(self.results.text)
                time.sleep(30)
        if len(self.results) == 20:
            self._multipage_download()

    def _filter_tiles(self, utm_filter=[]):
        print("Single tile filter, total results ", len(self.results))
        tiles = []
        for res in self.results:
            title = res['properties']['title'].split(".")[0]  # S2B_MSIL2A_20250508T094029_N0511_R036_T35ULA_20250508T115702.SAFE
            for pt in title.split("_"):
                if pt[0] == "T" and len(pt) == 6:   # T35ULA
                    tiles.append(pt)
        if len(utm_filter) > 0:
            newtl = []
            for tl in tiles:
                if int(tl[1:3]) in utm_filter:
                    newtl.append(tl)
            tiles = newtl
        arr = np.array(tiles)
        tile = None
        val, ct = np.unique(arr, return_counts=True)
        if self.tile_filter is not None:
            for l in np.argsort(ct):
                tile = str(val[l])
                if tile in self.tile_filter:
                    break
                else:
                    tile = None
        else:
            if len(val) > 0:
                tile = str(val[np.argmax(ct)])
        if tile is None:
            print("All tiles filtered no results")
            self.results = []
        else:
            print(f"Tile used {tile}")
            new_res = []
            for res in self.results:
                if tile in res["properties"]["title"]:
                    new_res.append(res)
            self.results = new_res
            print("Single tile filter, results left ", len(self.results))


    def _stac_filter_tiles(self, utm_filter=[]):
        print("Single tile filter, total results ", len(self.results))
        tiles = []
        for res in self.results:
            title = res['id']  # S2B_MSIL2A_20250508T094029_N0511_R036_T35ULA_20250508T115702
            for pt in title.split("_"):
                if pt[0] == "T" and len(pt) == 6:   # T35ULA
                    tiles.append(pt)
        if len(utm_filter) > 0:
            newtl = []
            for tl in tiles:
                if int(tl[1:3]) in utm_filter:
                    newtl.append(tl)
            tiles = newtl
        arr = np.array(tiles)
        tile = None
        val, ct = np.unique(arr, return_counts=True)
        if self.tile_filter is not None:
            for l in np.argsort(ct):
                tile = str(val[l])
                if tile in self.tile_filter:
                    break
                else:
                    tile = None
        else:
            if len(val) > 0:
                tile = str(val[np.argmax(ct)])
        if tile is None:
            print("All tiles filtered no results")
            self.results = []
        else:
            print(f"Tile used {tile}")
            new_res = []
            for res in self.results:
                if tile in res['id']:
                    new_res.append(res)
            self.results = new_res
            print("Single tile filter, results left ", len(self.results))

    def _filter_tiles_list(self, utm_filter=[]):
        if self.tile_filter is None:
            return True
        print("Tile filter, total results ", len(self.results))
        print("Filtering tiles: ", ",".join(self.tile_filter))
        new_res = []
        for res in self.results:
            title = res['id']  # S2B_MSIL2A_20250508T094029_N0511_R036_T35ULA_20250508T115702
            for pt in title.split("_"):
                if pt[0] == "T" and len(pt) == 6:   # T35ULA
                    if len(utm_filter) > 0:
                        if pt in self.tile_filter and int(pt[1:3]) in utm_filter:
                            new_res.append(res)
                    else:
                        if pt in self.tile_filter:
                            new_res.append(res)
        self.results = new_res
        print("Tiles filtered, results left ", len(self.results))

    def _filter_unique(self):
        new_res = []
        ids = []
        for res in self.results:
            title = res['id']
            if title not in ids:
                new_res.append(res)
                ids.append(title)
        self.results = new_res
        print("Unique filtered, results left ", len(self.results))

    def get_tiles(self):
        if self.results is not None:
            tiles = set()
            for res in self.results:
                title = res['id']  # S2B_MSIL2A_20250508T094029_N0511_R036_T35ULA_20250508T115702
                for pt in title.split("_"):
                    if pt[0] == "T" and len(pt) == 6:   # T35ULA
                        tiles.add(pt)
            return tiles
        else:
            print("No search results exist, try running search() first")
            return False

    def search(self, path=None, force=False, single_tile=False, tile_filter=None, ee_project=None, utm_filter=[]):
        # utm_filter - list of allowed utm projections (integers from 1 to 60)
        if self.filter.system == "EE":
            if ee_project is not None:
                ee.Initialize(project=ee_project)
        self.tile_filter = tile_filter
        if path is not None and self.results is None:
            self.read_results(path)
        if self.results is not None:
            if force is False:
                return False
        if self.filter.system == "AWS":
            if self.collection.satellite == "Sentinel-1":
                self._stac_search_s1_AWS()
            if self.collection.satellite == "Sentinel-2":
                self._stac_search_s2_AWS()
            res_len = len(self.results)
        if self.filter.system == "EE":
            if self.collection.satellite == "Sentinel-1":
                self._search_s1_EE()
            if self.collection.satellite == "Sentinel-2":
                self._search_s2_EE()
            res_len = self.results.size().getInfo()
        print("Total results before filter", res_len)
        if single_tile:
            self._stac_filter_tiles(utm_filter)
        else:
            self._filter_tiles_list(utm_filter)
        self._filter_unique()
        if path is not None:
            self.save_results(path, force)


class DataCollection:
    """
    Main class for data search and management. Used to collect and download sentinel data to a collection of tiff files and numpy datasets.
    """

    def __init__(self, satellite: Literal["Sentinel-1", "Sentinel-2"], system: Literal["AWS", "EE"], start_date: str | None, end_date: str | None,
                 geom: Geometry | None,  bands: list | None, maxcc: float = 100, sensorMode: str = "IW", orbit: str = "DESCENDING", productType: str = "GRD", metadata_bands: list = [], s3_client_profile: str = "sentinel_tools", verbose: bool = True):
        self.satellite = satellite
        self.system = system
        self.start_date = start_date
        self.end_date = end_date
        self.geom = geom
        self.maxcc = maxcc
        self.sensor = sensorMode
        self.orbit = orbit
        self.productType = productType
        self.bands = bands
        self.metadata_bands = metadata_bands
        self.s3_client_profile = s3_client_profile
        self.verbose = verbose
        self._filter = Filter(self.system, self.start_date, self.end_date, self.geom, self.maxcc, self.sensor, self.orbit, self.productType)
        self._collection = Collection(self.satellite, self.bands, self.metadata_bands)
        self._search = Search(self._filter, self._collection)
        self._data = None
        self._dataset = []
        self._files = None
        self._profiles = None
        self._matrix = None

    def search(self, result_path=None, force=False, single_tile=False, tile_filter=None, ee_project=None):
        self.tile_filter = tile_filter  # allow only specific sentinel 2 tiles
        self._search.search(result_path, force, single_tile, tile_filter, ee_project)
        if isinstance(self._search.results, ee.List):
            print(f"Search completed data items collected: {self._search.results.size().getInfo()}")
        else:
            print(f"Search completed data items collected: {len(self._search.results)}")

    @property
    def search_results(self):
        assert self._search.results is not None, f"Processing error, no search results available"
        return self._search.results

    def get_tiles(self):
        assert self._search.results is not None, f"Processing error, no search results available"
        return self._search.get_tiles()

    def _process_AWS(self):
        assert self._search.results is not None, f"Processing error, no search results available"
        data = []
        for ft in self._search.results:
            data.append(SentinelData(metadata=ft, image=None, geom=None, s3_client_profile=self.s3_client_profile, verbose=self.verbose))
        self._data = data

    def _process_EE(self):
        assert isinstance(self._search.results, ee.List), f"Processing error, the results type has to be ee.List is {type(self._search.results)}"
        data = []
        image_size = self._search.results.size().getInfo()
        for i in range(image_size):
            data.append(SentinelData(metadata=None, image=ee.Image(self._search.results.get(i)).select(self.bands), geom=self.geom, verbose=self.verbose))
        self._data = data

    def download(self, out_folder, force=False, single_download=False):
        assert out_folder is not None
        self.out_folder = out_folder
        os.makedirs(self.out_folder, exist_ok=True)
        if self._search.results is None:
            self.search(force)
        if self._data is None:
            if self.system == "AWS":
                self._process_AWS()
            if self.system == "EE":
                self._process_EE()
        if single_download:
            res = self._data[0].download(self.out_folder, self.bands, self.satellite, self.metadata_bands)
            self._data = self._data[:1]
            if res is not False:
                if "new_bands" in res:
                    for bnd in res["new_bands"]:
                        if bnd not in self.bands:
                            self.bands.append(bnd)
        else:
            for dt in self._data:
                res = dt.download(self.out_folder, self.bands, self.satellite, self.metadata_bands)
                if res is not False:
                    if "new_bands" in res:
                        for bnd in res["new_bands"]:
                            if bnd not in self.bands:
                                self.bands.append(bnd)
        self._collection.bands = self.bands
        print(self.bands)

    def read(self, clip_to_shape=True, upscale=True):
        print("Reading data")
        if clip_to_shape:
            for dt in self._data:
                dt.read(self.geom)
                if upscale:
                   dt._upscale()
        else:
            for dt in self._data:
                dt.read()
                if upscale:
                   dt._upscale()
        for dt in self._data:
            minx = 10**5
            maxx = 0
            miny = 10**5
            maxy = 0
            for dat in dt.data:  # bands, x, y
                if dat.shape[1] > maxx:
                    maxx = dat.shape[1]
                if dat.shape[1] < minx:
                    minx = dat.shape[1]
                if dat.shape[2] > maxy:
                    maxy = dat.shape[2]
                if dat.shape[2] < miny:
                    miny = dat.shape[2]
            tmp = None
            for data_part in dt.data:
                difx = data_part.shape[1] - minx
                dify = data_part.shape[2] - miny
                if tmp is None:
                    tmp = data_part[:, difx:, dify:]
                else:
                    tmp = np.append(tmp, data_part[:, difx:, dify:], axis=0)
            print("Data part shape: ", tmp.shape)
            self._dataset.append(tmp)

    def __len__(self):
        if self._data is not None:
            return len(self._data)
        else:
            return 0

    @property
    def files(self):
        if self._files is None:
            self._files = []
            for dt in self._data:
                self._files.extend(dt.files)
        return self._files

    @property
    def matrix(self):
        if len(self._dataset) == 0:
            self.read()
        # create a matrix of all the data
        if self._matrix is None:
            minx = 10**5
            maxx = 0
            miny = 10**5
            maxy = 0
            for dt in self._dataset:  # bands, x, y
                if dt.shape[1] > maxx:
                    maxx = dt.shape[1]
                if dt.shape[1] < minx:
                    minx = dt.shape[1]
                if dt.shape[2] > maxy:
                    maxy = dt.shape[2]
                if dt.shape[2] < miny:
                    miny = dt.shape[2]
            print(minx, maxx, miny, maxy)
            for dt in self._dataset:
                difx = dt.shape[1] - minx
                dify = dt.shape[2] - miny
                if self._matrix is None:
                    self._matrix = dt[np.newaxis, :, difx:, dify:]
                else:
                    self._matrix = np.append(self._matrix, dt[np.newaxis, :, difx:, dify:], axis=0)
        return self._matrix

    @property
    def dataset(self):
        """
        Return geotiff data as list of numpy arrays.

        Parameters
        ----------
        index: int or list of ints
            index of which data collections to return
        """
        if len(self._dataset) == 0:
            self.read()
        return self._dataset

    @property
    def profiles(self):
        if self._profiles is None:
            self._profiles = []
            for dt in self._data:
                self._profiles.extend(dt.profiles)
        return self._profiles

    @property
    def data(self):
        return self._data

    def get_data(self, index: int | None):
        return self._data[index]

    def sentinel_1_mass_download(self, snappy=None):  # SnappyProcessor
        for dt in self._data:
            gc.enable()
            gc.collect()
            dt.sentinel_1_mass_download()
            if snappy is not None:
                snappy(dt, self.geom)

    def delete(self, orig_only=False):
        for dt in self._data:
            dt.delete(orig_only)

    @classmethod
    def load(file_path, geom):
        col = DataCollection()

    def save(self, file_path):  # save collection metadata for loading
        save_data = {"sat": self.satellite,
                     "sys": self.system,
                     "sd": self.start_date,
                     "ed": self.end_date,
                     "maxcc": self.maxcc,
                     "sensor": self.sensor,
                     "orbit": self.orbit,
                     "prod" : self.productType,
                     "bands": self.bands}


# aws s3 data path S1   [product type]/[year]/[month]/[day]/[mode]/[polarization]/[product identifier]
# https://roda.sentinel-hub.com/sentinel-s1-l1c/GRD/readme.html

# aws s3 data path S2   tiles/[UTM code]/latitude band/square/[year]/[month]/[day]/[sequence]/DATA
# https://roda.sentinel-hub.com/sentinel-s2-l2a/readme.html

# general info https://developers.google.com/earth-engine/glossary
class SentinelData:
    """
    Data class to collect and process and manage sentinel data parts. EE Images and AWS S3 data parts.
    """

    def __init__(self, metadata: str | None, image: ee.Image | None, geom: Geometry | None, s3_client_profile: str = "sentinel_tools", verbose: bool = True):
        self.metadata = metadata
        self.image = image
        assert self.metadata is not None or self.image is not None, "Image or metadata has to be provided"
        self.geom = geom
        if self.geom is not None:
            self.ee_geom = ee.Geometry.Rectangle(self.geom.bbox)
            self.geom_id = self.geom.hashval
        else:
            self.geom_id = hashlib.md5("".encode()).hexdigest()[-32:]
        self._files = []
        self._profiles = []
        self._data = []
        self._file_ids = []
        self.processed_files = []
        self._datetime = None
        self.title = None
        self.metadata_files = []
        self.compress = True
        self.s3_client_profile = s3_client_profile
        self.verbose = verbose

    def __repr__(self):
        if self.title is not None:
            return f"Sentinel Data: {self.title}"
        else:
            return f"Sentinel Data: {self.image}"
        

    def _download_EE(self, out_folder, bands, satellite, metadata_bands=None):
        self.out_folder = out_folder
        out_dir = copy.deepcopy(out_folder)
        if out_folder[-1] != "/":
            out_dir += "/"
        out_dir += f"{satellite}/{self.geom_id}"
        self.out_folder = out_dir
        os.makedirs(out_dir, exist_ok=True)
        for b in bands:
            img = self.image.select(b)
            img_id = self.image.getInfo()['properties']['system:index']
            fn = f"{self.out_folder}/{img_id}_{b}.tif"  # sentinel-1a and 1c
            self._files.append(fn)
            self._file_ids.append(img_id)
            if os.path.exists(fn):
                if self.verbose:
                    print(f"Skipping downloading {fn}")
                continue
            if self.verbose:
                print(f"Downloading {fn}")
            if self.geom is not None:
                region = self.ee_geom
                epsg = f"EPSG:{self.geom.utm.epsg}"
            else:
                region = None
                epsg = 'EPSG:4326'
            geemap.ee_export_image(
                img,
                filename=fn,
                scale=10,
                region=region,          # bbox
                file_per_band=False,  # VV+VH single GeoTIFF
                crs=epsg
            )
        return False

    def _download_AWS(self, out_folder, bands, satellite, metadata_bands=[]):
        # TODO: Fix multiple datasets for same date and tile.
        session = boto3.Session(profile_name=AWS_EODATA_PROFILE)
        client = session.client('s3')
        s3_files = [None] * len(bands)
        s3_key = None
        if satellite == "Sentinel-1":
            # create sentinel 1 aws s3 path
            bucket = EODATA_BUCKET
            #date = datetime.fromisoformat(self.metadata["properties"]["startDate"])
            date = datetime.strptime(self.metadata["properties"]["datetime"], "%Y-%m-%dT%H:%M:%S.%fZ")
            title = self.metadata["id"]  # S1A_IW_GRDH_1SDV_20250503T044344_20250503T044409_059025_075225_1C05.SAFE
            self.title = title

            for i, b in enumerate(bands):
                band_key = S1_BANDS_TO_STAC_BANDS[b]
                asset = self.metadata["assets"][band_key]
                s3_file = "/".join(asset["href"].split("//")[-1].split("/")[1:])
                s3_files[i] = s3_file

        if satellite == "Sentinel-2":
            bucket = EODATA_BUCKET
            # date = datetime.fromisoformat(self.metadata["properties"]["startDate"])
            date = datetime.strptime(self.metadata["properties"]["datetime"], "%Y-%m-%dT%H:%M:%S.%fZ")
            title = self.metadata['id']  # S2B_MSIL2A_20250508T094029_N0511_R036_T35ULA_20250508T115702.SAFE
            self.title = title
            tile = None
            for pt in title.split("_"):
                if pt[0] == "T" and len(pt) == 6:   # T35ULA
                    tile = pt
                    break
            assert tile is not None, f"Error tile not found in {title}"
            for i, b in enumerate(bands):
                if b in S2_GENERATED_BANDS:  # skip generated bands
                    continue
                band_key = S2_BANDS_TO_STAC_BANDS[b]
                asset = self.metadata["assets"][band_key]
                s3_file = "/".join(asset["href"].split("//")[-1].split("/")[1:])
                s3_files[i] = s3_file
        
        # download files
        self.s3_files = s3_files
        out_dir = copy.deepcopy(out_folder)
        if out_folder[-1] != "/":
            out_dir += "/"
        out_dir += title + "/"
        self.out_folder = out_dir
        os.makedirs(out_dir, exist_ok=True)
        for i, fl in enumerate(s3_files):
            if fl is None:  # skip generated bands
                continue
            if satellite == "Sentinel-2":
                fn = f"{out_dir}" + fl.split("/")[-1]
            if satellite == "Sentinel-1":
                fn = f"{out_dir}" + "measurement/" + fl.split("/")[-1]
                os.makedirs(out_dir + "measurement/", exist_ok=True)
            self._files.append(fn)
            self._file_ids.append(fl.split("/")[-1])
            if os.path.exists(fn):
                if self.verbose:
                    print(f"Skipping downloading {fl} to {fn}")
                continue
            if self.verbose:
                print(f"Downloading {fl} to {fn}")
            try:
                client.download_file(bucket, fl, fn)
            except Exception as e:
                print(e)
        if len(metadata_bands) > 0:
            max_idx = 0
            max_val = 10000
            for i, fl in enumerate(self._files):
                with rasterio.open(fl) as f:
                    prof = f.profile
                    val = prof["transform"].a
                    if val < max_val:
                        max_val = val
                        max_idx = i
            with rasterio.open(self._files[max_idx]) as f:
                prof = f.profile
            ret_val = {"new_bands": []}
            for mb in metadata_bands:  # file to get zenith and azimuth data from (for STAC API = granule_metadata)
                if satellite == "Sentinel-2":
                    fn = f"{self.out_folder}" + mb
                    asset = self.metadata["assets"][mb]
                    fl = "/".join(asset["href"].split("//")[-1].split("/")[1:])
                    self.metadata_files.append(fn)
                    if os.path.exists(fn):
                        if self.verbose:
                            print(f"Skipping downloading {fl} to {fn}")
                    else:
                        if self.verbose:
                            print(f"Downloading {fl} to {fn}")
                        try:
                            client.download_file(bucket, fl, fn)
                        except Exception as e:
                            print(e)
                    if mb == "granule_metadata":
                        zenith_path = f"{self.out_folder}zenith.tiff"
                        azimuth_path = f"{self.out_folder}azimuth.tiff"
                        if os.path.exists(zenith_path) and os.path.exists(azimuth_path):
                            pass
                        else:
                            zenith, azimuth = parse_metadata(fn)
                            matrix_to_tiff(zenith, prof, zenith_path, self.compress)
                        
                            matrix_to_tiff(azimuth, prof, azimuth_path, self.compress)
                        self._files.append(zenith_path)
                        self._files.append(azimuth_path)
                        ret_val = {"new_bands": ret_val["new_bands"].extend(["zenith", "azimuth"])}
            return {"new_bands": ["zenith", "azimuth"]}
        return False

    def download(self, out_folder, bands, satellite, metadata_bands=[]):
        if self.image is not None:
            res = self._download_EE(out_folder, bands, satellite, metadata_bands)
        if self.metadata is not None:
            res = self._download_AWS(out_folder, bands, satellite, metadata_bands)
        return res

    def sentinel_1_mass_download(self):
        # use for sentinel1 only, to download all data for postprocessing purposes
        s3 = boto3.resource("s3")
        new_key = "/".join(self.s3_key.split("/")[:-2])
        if os.path.exists(self.out_folder + "manifest.safe"):
            if self.verbose:
                print(f"SAFE file exists skipping download of folder: {new_key}")
        else:
            if self.verbose:
                print(f"Downloading folder: {new_key}")
            download_s3_folder(s3, S1_S3_URL, new_key, self.out_folder)

    @property
    def folder(self):
        return self.out_folder

    @property
    def files(self):
        return self._files

    @property
    def profiles(self):
        return self._profiles

    @property
    def data(self):
        return self._data

    @property
    def datetime(self):
        if self._datetime is None:
            if self.image is not None:
                self._datetime = datetime.fromtimestamp(self.image.get('system:time_start').getInfo() / 1000)
            else:
                self._datetime = datetime.fromisoformat(self.metadata["properties"]["datetime"])
        return self._datetime

    def get_file_id(self, index):
        fid = self._file_ids[index]
        fid = fid.split(".")[0]
        return fid

    def get_file_band(self, index):
        fname = self._files[index]
        bnd = fname.split("_")[-1]
        bnd = bnd.split(".")[0]
        return bnd

    def _upscale(self):
        assert len(self._data) > 0
        assert len(self._data[0].shape) == 3
        maxx = 0
        for i, dt in enumerate(self._data):
            if dt.shape[1] > maxx:
                maxx = dt.shape[1]
        for i, dt in enumerate(self._data):
            if dt.shape[1] < maxx:
                diffx = round(maxx /dt.shape[1])
                assert diffx in [2, 3, 6], f"wrong image scale difference, original shape {dt.shape[1]} - max shape {maxx}"
                self._data[i] = _upscaler(dt, diffx)
                self.profiles[i].update({"height": self._data[i].shape[1], "width": self._data[i].shape[2]})

    def _write_processed_raster(self, combine_bands=True, mask=None):
        if combine_bands:
            assert len(self._data.shape) == 3, "Input data has to be 3D of shape: bands, X, Y"
            assert np.argmin(self._data.shape) == 0, "Input data has to be 3D of shape: bands, X, Y for rasterio"
            if self._data.dtype == np.dtype("float"):  # reduce the float accuracy for image saving
                X = self._data.astype("float32")
            else:
                X = self._data
            meta = self._profiles[0]
            meta['count'] = X.shape[0]
            meta['dtype'] = X.dtype
            meta['driver'] = "GTiff"
            meta["interleave"] = "band"
            if self.compress:
                meta["compress"] = "lzw"
                meta["predictor"] = 2
            pth = "/".join(self.files[0].split("/")[:-1])
            pth += "processed.tiff"
            self.processed_files.append(pth)
            with rasterio.open(pth, "w", **meta) as f:
                f.write(X)
                if mask is not None:
                    f.write_mask(mask)
        else:
            for i, fl in enumerate(self.files):
                pass

    def clip_raster(self, f, geom):  # raster file
        data = f.read()
        assert len(data.shape) == 3
        if f.profile["crs"].to_epsg() != geom.crs.epsg:  # reproject geometry
            geom = geom.transform(CRS(f.profile["crs"].to_epsg()))
        polygon = geom.geometry
        with mem_raster(data, f.profile) as src:
            out_image, out_transform = mask(src, [polygon], crop=True, all_touched=True)
            out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
        return out_image, out_meta

    def read(self, clip_to_shape: Geometry | None, profiles_only=False, force=False):
        if len(self._data) == 0 or force:
            self._data = []
            self._profiles = []
            fls = self._files
            if len(self.processed_files) > 0:
                fls = self.processed_files
            for fl in fls:
                with rasterio.open(fl) as f:
                    if profiles_only:
                        self._profiles.append(f.profile)
                        continue
                    else:
                        if clip_to_shape is not None:
                            dt, prof = self.clip_raster(f, clip_to_shape)
                            self._data.append(dt)
                            self._profiles.append(prof)
                        else:
                            self._data.append(f.read())
                            self._profiles.append(f.profile)

    def delete(self, orig_only=False):
        """
        Delete the files of the tiles.

        orig_only: bool
            If True, only the original files are deleted.
        """
        if orig_only:
            for fl in self._files:
                os.remove(fl)
        else:
            try:
                shutil.rmtree(self.out_folder)
            except Exception as e:
                print(e)
                pass

    def clear(self, orig_only=True):
        """
        Delete the files and clear the data.
        """
        self.delete(orig_only)
        self._data = None
        self._files = None

    def save(self, path=None):
        save_dict = {}
        for i, val in self._files:
            save_dict.update({i: val})
        if path is None:
            return save_dict
        else:
            pass


class MultitileDataset:
    def __init__(self, collection: DataCollection, main_epsg: int, subfolder: str = "Results", compress = True, separate_bands = True, verbose: bool = True):
        self.collection = collection
        self.subfolder = subfolder
        self.out_dir = os.path.join(self.collection.out_folder, subfolder)
        self.compress = compress
        self.separate_bands = separate_bands
        self.main_epsg = main_epsg
        self.verbose = verbose
        os.makedirs(self.out_dir, exist_ok=True)
        self._groups = []
        self._vrts = []
        self._tiffs = []
        self._create_groups()
        self._epsgs = [None] * len(self._groups)
        self._read_profiles()
        self._get_bands()

    def _get_bands(self):
        new = []
        for b in self.collection.bands:
            if "/" in b:
                new.append(b.split("/")[-1])
            else:
                new.append(b)
        self._bands = new
        print(self._bands)

    def _create_groups(self):
        dates = []
        for dt in self.collection.data:
            dates.append(dt.datetime.date())
        dates = np.array(dates)
        self._dates = np.unique(dates)
        for un in np.unique(dates):
            tmp = []
            for i, dt in enumerate(dates):
                if dt == un:
                    tmp.append(self.collection.data[i])
            self._groups.append(tmp)

    def _vrt_to_tiff(self, vrt_file, out_file, out_epsg: int):
        if os.path.exists(out_file):
            print(f"Tiff file {out_file} exists, skipping generation")
            return True
        out_srs = rasterio.crs.CRS.from_epsg(out_epsg)
        cmd = f'gdal_translate -q -of GTiff -a_srs {out_srs} '
        if self.verbose:
            cmd = f'gdal_translate -of GTiff -a_srs {out_srs} '
        if self.compress:
            cmd += f'-co compress=lzw -co predictor=2 '
        cmd += f'{vrt_file} {out_file}'
        proc = subprocess.Popen(shlex.split(cmd))
        print(f"Converting vrt to tiff: {out_file}")
        proc.wait()

    def _merge_tiffs_1(self, files, out_file):
        all_files = " ".join(files)
        if self.separate_bands:
            cmd = f'gdal_merge.py -q -of GTiff -separate -a_nodata 0 -o {out_file} '
            if self.verbose:
                cmd = f'gdal_merge.py -of GTiff -separate -a_nodata 0 -o {out_file} '
        else:
            cmd = f'gdal_merge.py -q -of GTiff -a_nodata 0 -o {out_file} '
            if self.verbose:
                cmd = f'gdal_merge.py -of GTiff -a_nodata 0 -o {out_file} '
        if self.compress:
            cmd += f'-co compress=lzw -co predictor=2 '
        cmd += f'{all_files}'
        proc = subprocess.Popen(shlex.split(cmd))
        print(f"Merging {len(files)} tiffs to file: {out_file}")
        proc.wait()

    def _merge_tiffs(self, files, out_file, block_size=128):
        all_files = " ".join(files)
        tmp_file = out_file[:-4] + "vrt"
        if self.separate_bands:
            cmd = f'gdalbuildvrt -q -separate -srcnodata 0 -vrtnodata 0  {tmp_file} {all_files}'
            if self.verbose:
                cmd = f'gdalbuildvrt -separate -srcnodata 0 -vrtnodata 0  {tmp_file} {all_files}'
        else:
            cmd = f'gdalbuildvrt -q -srcnodata 0 -vrtnodata 0  {tmp_file} {all_files}'
            if self.verbose:
                cmd = f'gdalbuildvrt -srcnodata 0 -vrtnodata 0  {tmp_file} {all_files}'
        proc = subprocess.Popen(shlex.split(cmd))
        proc.wait()
 
        cmd = f'gdal_translate -q -of GTiff -co BIGTIFF=YES -co BLOCKXSIZE={block_size} -co BLOCKYSIZE={block_size} -co TILED=YES -co NUM_THREADS=ALL_CPUS '
        if self.verbose:
            cmd = f'gdal_translate -of GTiff -co BIGTIFF=YES -co BLOCKXSIZE={block_size} -co BLOCKYSIZE={block_size} -co TILED=YES -co NUM_THREADS=ALL_CPUS '
        if self.compress:
            cmd += f'-co compress=lzw -co predictor=2 '
        cmd += f'{tmp_file} {out_file}'
        proc = subprocess.Popen(shlex.split(cmd))
        print(f"Merging {len(files)} tiffs to file: {out_file}")
        proc.wait()

    def _clip_tiff(self, in_file, out_file, path_to_shp):
        try:
            os.remove(out_file)
        except:
            pass
        cmd = f'gdalwarp -q -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -crop_to_cutline -cutline {path_to_shp} {in_file} {out_file} '
        if self.verbose:
            cmd = f'gdalwarp -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -crop_to_cutline -cutline {path_to_shp} {in_file} {out_file} '
        if self.compress:
            cmd += f'-co compress=lzw -co predictor=2 '
        proc = subprocess.Popen(shlex.split(cmd))
        print(f"Cropping tiff using shapefile {path_to_shp} to file {out_file}")
        proc.wait()

    def _read_profiles(self):
        for dt in self.collection.data:
            dt.read(clip_to_shape=None, profiles_only=True)

    def _tiff_transform(self, file_in, file_out, epsg_in: int, epsg_out: int, force = False):
        if os.path.exists(file_out):
            if not force:
                print(f"File {file_out} exists skipping transformation")
                return True
        crs_in = rasterio.crs.CRS.from_epsg(epsg_in)
        crs_out = rasterio.crs.CRS.from_epsg(epsg_out)
        cmd = f'gdalwarp -q -s_srs {crs_in} -t_srs {crs_out} {file_in} {file_out}'  # source srs, target srs
        if self.verbose:
            cmd = f'gdalwarp -s_srs {crs_in} -t_srs {crs_out} {file_in} {file_out}'  # source srs, target srs
            print(f"Transforming tiff to: {file_out}")
        proc = subprocess.Popen(shlex.split(cmd))
        proc.wait()

    def _tiff_transform_th(self, i, fl, epsgs):
        pass
    
    def _tiffs_to_vrt(self, out_path, files_list, profile_list, epsg=None):
        assert len(files_list) == len(profile_list)
        epsgs = []
        for pfl in profile_list:
            epsgs.append(pfl["crs"].to_epsg())
        epsgs = np.array(epsgs)
        if epsg is None:
            vals, cts = np.unique(epsgs, return_counts=True)
            epsg = vals[np.argmax(cts)]
        new_files = []
        for i, fl in enumerate(files_list):
            if epsgs[i] != epsg:
                # contvert file to uniform CRS
                tmp_fl_ext = fl.split(".")[-1]
                tmp_fl = fl[:-(len(tmp_fl_ext) + 1)]
                tmp_fl += f"_{epsg}.{tmp_fl_ext}"
                self._tiff_transform(fl, tmp_fl, epsgs[i], epsg)
                new_files.append(tmp_fl)
            else:
                new_files.append(fl)
        all_files = " ".join(new_files)
        if self.verbose:
            cmd = f'gdalbuildvrt -srcnodata 0 -vrtnodata 0  {out_path} {all_files}'
            # cmd = f'gdalbuildvrt  {path_out} {all_files}'
            proc = subprocess.Popen(shlex.split(cmd))
            print(f"combining tiffs to path: {out_path}")
        else:
            cmd = f'gdalbuildvrt -q -srcnodata 0 -vrtnodata 0  {out_path} {all_files}'
            # cmd = f'gdalbuildvrt  {path_out} {all_files}'
            proc = subprocess.Popen(shlex.split(cmd))
        proc.wait()
        return epsg

    def combine_tiffs(self):
        self._get_bands()
        for i, gp in enumerate(self._groups):
            vrt_tmp = []
            out_folder = os.path.join(self.out_dir, self._dates[i].strftime('%Y-%m-%d'))
            os.makedirs(out_folder, exist_ok=True)
            for b in self._bands:
                files = []
                profiles = []
                for sd in gp:  # sentinel data in group
                    for l, fl in enumerate(sd.files):
                        if b in fl:
                            files.append(fl)
                            profiles.append(sd.profiles[l])
                            break
                out_path = os.path.join(out_folder, f"{b}.vrt")
                epsg_out = self._tiffs_to_vrt(out_path, files, profiles, self.main_epsg)
                self._epsgs[i] = epsg_out
                vrt_tmp.append(out_path)
            self._vrts.append(vrt_tmp)

    def export_tiffs(self):
        for i, vrt in enumerate(self._vrts):
            grp = []
            for fl in vrt:
                fn = fl[:-3] + "tiff"
                self._vrt_to_tiff(fl, fn, self.main_epsg)
                grp.append(fn)
            self._tiffs.append(grp)

    def merge_all(self, geom_path, bands: list = [], force = False):  # add date ranges
        processed_files = []
        if len(bands) == 0:
            bands = self._bands
        for b in self._bands:
            if b in bands:
                files = []
                for i, tiff in enumerate(self._tiffs):
                    for fl in tiff:
                        if b in fl:
                            files.append(fl)
                # combine bands
                out_dir = os.path.join(self.out_dir, "merged")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{b}.tiff")
                out_path_clip = os.path.join(out_dir, f"{b}_clip.tiff")
                processed_files.append(out_path_clip)
                if os.path.exists(out_path_clip):
                    if not force:
                        if self.verbose:
                            print(f"Processsed file {out_path_clip} exists, skipping merge and clip.")
                        continue
                self._merge_tiffs(files, out_path)
                self._clip_tiff(out_path, out_path_clip, geom_path)
                os.remove(out_path)  # delete merged non-clipped file to save space
        self.processed_files = processed_files


class MergedDataset:
    def __init__(self, md: MultitileDataset, read_window = 4000, verbose: bool = True):
        self.md = md
        self._profiles = []
        self._collect_profiles()
        self._compute_scale()
        self.read_window = read_window
        self.bands = md._bands
        self.mask = None
        self.verbose = verbose
        self._processed_files = [""] * len(self._profiles)
        self.out_dir = os.path.join(self.md.out_dir, "merged")
        self.cloud_percentage = 0

    def _collect_profiles(self):
        for fl in self.md.processed_files:
            with rasterio.open(fl) as f:
                self._profiles.append(f.profile)

    def _compute_scale(self):
        w = []
        h = []
        for pf in self._profiles:
            w.append(pf["width"])
            h.append(pf["height"])
        self.w = np.array(w)
        self.h = np.array(h)
        scale = np.round(self.w.max()/self.w)
        self._scale = scale

    def _process_mask(self, data: np.ndarray, scale: int, count_clouds: bool=False):
        msk = np.zeros_like(data, dtype="uint8")
        # layer = np.full((1, data.shape[1], data.shape[2]), 1)
        msk[data > 0] = 2
        msk[np.logical_and(data > 3, data < 7)] = 3
        msk = np.reshape(msk, tuple(data.shape))
        if count_clouds:
            msk = np.max(msk, axis=0)
        else:
            msk = np.argmax(msk, axis=0)
        if scale >1:
            msk = msk.repeat(scale,axis=0).repeat(scale,axis=1)
        return msk

    def _process_data(self):
        start = time.time()
        for i, b in enumerate(self.bands):
            if  b != "SCL":
                out_filename = os.path.join(self.out_dir, f"{b}_merged.tiff")
                if os.path.exists(out_filename):
                    print(f"File {out_filename} exists, skipping generation")
                    self._processed_files[i] = out_filename
                    continue
                if type(self.mask) == str:
                    with rasterio.open(self.mask) as f:
                        self.mask = f.read(1)
                wd = self.read_window//self._scale[i]
                ws = np.linspace(0, self.w[i], np.round(self.w[i]/wd).astype(int)).astype(int)
                hs = np.linspace(0, self.h[i], np.round(self.h[i]/wd).astype(int)).astype(int)
                tiff = None
                for x in range(1, len(ws)):
                    part = None
                    for y in range(1, len(hs)):
                        with rasterio.open(self.md.processed_files[i]) as f:
                            data = f.read(window=Window(ws[x-1], hs[y-1], ws[x] - ws[x-1], hs[y] - hs[y-1]))
                            # data shape -> bands, x, y
                            if self._scale[i] >1:
                                data = data.repeat(self._scale[i],axis=1).repeat(self._scale[i],axis=2)
                        sc = int(self._scale[i])
                        # print(ws[x-1], hs[y-1], ws[x], hs[y])
                        mask_part = self.mask[hs[y-1]*sc:hs[y]*sc, ws[x-1]*sc:ws[x]*sc]
                        # print(mask_part.shape, data.shape)
                        if data.shape[1] != mask_part.shape[0]:
                            if data.shape[1] > mask_part.shape[0]:
                                data = data[:, :mask_part.shape[0], :]
                            else:
                                mask_part = mask_part[:, :data.shape[1], :]
                        if data.shape[2] != mask_part.shape[1]:
                            if data.shape[2] > mask_part.shape[1]:
                                data = data[:, :, :mask_part.shape[1]]
                            else:
                                mask_part = mask_part[:, :, :data.shape[2]]
                        # TODO: Recompute mask when save_scl used and not all files generated
                        if b in S2_GENERATED_BANDS:
                            dt = process_data_ext(data, mask_part, use_float=True)
                        else:
                            dt = process_data_ext(data, mask_part)
                        
                        # print(ws[x-1], hs[y-1], ws[x], hs[y], data.shape, dt.shape)
                        if part is None:
                            part = dt
                        else:
                            part = np.append(part, dt, axis=0)
                    if tiff is None:
                        tiff = part
                    else:
                        tiff = np.append(tiff, part, axis=1)
                    print(tiff.shape)
                self._processed_files[i] = out_filename
                self._save_tiff(tiff, out_filename)
                del tiff
        end = time.time()
        # print(end - start)

    def _generate_mask(self, count_clouds=False, geom_path=None, save_scl=False):
        for i, b in enumerate(self.bands):
            if  b == "SCL":
                out_filename = os.path.join(self.out_dir, f"{b}_merged.tiff")
                if not count_clouds:
                    if os.path.exists(out_filename):
                        print(f"File {out_filename} exists, skipping generation")
                        self.mask = out_filename
                        self._processed_files[i] = out_filename
                        break
                    wd = self.read_window//self._scale[i]
                    ws = np.linspace(0, self.w[i], np.round(self.w[i]/wd).astype(int)).astype(int)
                    hs = np.linspace(0, self.h[i], np.round(self.h[i]/wd).astype(int)).astype(int)
                    msk = None
                    if save_scl:
                        scl_data = None
                        for x in range(1, len(ws)):
                            part = None
                            for y in range(1, len(hs)):
                                with rasterio.open(self.md.processed_files[i]) as f:
                                    data = f.read(window=Window(ws[x-1], hs[y-1], ws[x] - ws[x-1], hs[y] - hs[y-1]))
                                    if self._scale[i] >1:
                                        scl_dt = data.repeat(self._scale[i],axis=1).repeat(self._scale[i],axis=2).squeeze()
                                dt = self._process_mask(data, self._scale[i])
                                # print(ws[x-1], hs[y-1], ws[x], hs[y], data.shape, dt.shape)
                                if part is None:
                                    scl_part = scl_dt
                                    part = dt
                                else:
                                    if part.shape[0] < dt.shape[0]:
                                        dt = dt[:part.shape[0], :]
                                    if scl_part.shape[0] < scl_dt.shape[0]:
                                        scl_dt = scl_dt[:scl_part.shape[0], :]
                                    if part.shape[1] < dt.shape[1]:
                                        dt = dt[:, :part.shape[1]]
                                    if scl_part.shape[1] < scl_dt.shape[1]:
                                        scl_dt = scl_dt[:, :scl_part.shape[1]]
                                    part = np.append(part, dt, axis=0)
                                    scl_part = np.append(scl_part, scl_dt, axis=0)
                            if msk is None:
                                msk = part
                                scl_data = scl_part
                            else:
                                if msk.shape[1] < part.shape[1]:
                                    part = part[:, :msk.shape[1]]
                                if scl_data.shape[1] < scl_part.shape[1]:
                                    scl_part = scl_part[:, :scl_data.shape[1]]
                                if scl_data.shape[2] < scl_part.shape[2]:
                                    scl_part = scl_part[:, :, :scl_data.shape[2]]
                                msk = np.append(msk, part, axis=1)
                                scl_data = np.append(scl_data, scl_part, axis=1)
                            print(msk.shape, scl_data.shape)
                    else:
                        for x in range(1, len(ws)):
                            part = None
                            for y in range(1, len(hs)):
                                with rasterio.open(self.md.processed_files[i]) as f:
                                    data = f.read(window=Window(ws[x-1], hs[y-1], ws[x] - ws[x-1], hs[y] - hs[y-1]))
                                dt = self._process_mask(data, self._scale[i])
                                # print(ws[x-1], hs[y-1], ws[x], hs[y], data.shape, dt.shape)
                                if part is None:
                                    part = dt
                                else:
                                    part = np.append(part, dt, axis=0)
                            if msk is None:
                                msk = part
                            else:
                                msk = np.append(msk, part, axis=1)
                            print(msk.shape)
                    self._processed_files[i] = out_filename
                    self.mask = msk
                    if save_scl:
                        self._save_tiff(scl_data, out_filename)
                    else:
                        self._save_tiff(self.mask, out_filename)
                    del msk
                else:
                    wd = self.read_window
                    ws = np.linspace(0, self.w[i], np.round(self.w[i]/wd).astype(int)).astype(int)
                    hs = np.linspace(0, self.h[i], np.round(self.h[i]/wd).astype(int)).astype(int)
                    msk = None
                    for x in range(1, len(ws)):
                        part = None
                        for y in range(1, len(hs)):
                            with rasterio.open(self.md.processed_files[i]) as f:
                                data = f.read(window=Window(ws[x-1], hs[y-1], ws[x] - ws[x-1], hs[y] - hs[y-1]))
                            dt = self._process_mask(data, 1, True)
                            # print(ws[x-1], hs[y-1], ws[x], hs[y], data.shape, dt.shape)
                            if part is None:
                                part = dt
                            else:
                                part = np.append(part, dt, axis=0)
                        if msk is None:
                            msk = part
                        else:
                            msk = np.append(msk, part, axis=1)
                    msk += 1
                    if geom_path is not None:
                        out_filename = os.path.join(self.out_dir, f"{b}_clouds.tiff")
                        self._save_tiff(msk, out_filename)
                        del msk
                        geom = Geometry.from_shapefile(geom_path)
                        with rasterio.open(out_filename) as f:
                            if f.profile["crs"].to_epsg() != geom.crs.epsg:  # reproject geometry
                                geom = geom.transform(CRS(f.profile["crs"].to_epsg()))
                            polygon = geom.geometry
                            msk, out_transform = mask(f, [polygon], crop=True)
                        os.remove(out_filename)
                    uq = np.unique(msk, return_counts=True)  # counts of 0, 1, 3, 4
                    print(uq)
                    idxs = []
                    for i, val in enumerate(uq[0]):
                        if val > 0:
                            idxs.append(i)
                    uq = uq[1]
                    self.cloud_percentage = uq[-1] / np.sum(uq[idxs])
                    print(self.cloud_percentage)
                    del msk

    def _save_tiff(self, data, filename):
        prof = None
        for i, sc in enumerate(self._scale):
            if int(sc) == 1:
                prof = self._profiles[i]
                break
        if len(data.shape) == 2:
            data = data[np.newaxis, :, :]
        meta = prof
        meta['count'] = data.shape[0]
        # TODO: possible error may appear creating geotiffs from envi files
        meta['dtype'] = data.dtype
        meta['driver'] = "GTiff"
        meta["interleave"] = "band"
        meta["compress"] = "lzw"
        # meta["predictor"] = 2
        print(meta)
        with rasterio.open(filename, "w", **meta) as f:
            f.write(data)

    def _upscale(self, inpath, outpath, sizex, sizey, block_size=128, compress=True):
        cmd = f'gdal_translate -of GTiff -outsize {sizex} {sizey} -r bilinear -co BLOCKXSIZE={block_size} -co BLOCKYSIZE={block_size} -co TILED=YES -co NUM_THREADS=ALL_CPUS '
        if compress:
            cmd += f'-co compress=lzw -co predictor=2 '
        cmd += f'{inpath} {outpath}'
        proc = subprocess.Popen(shlex.split(cmd))
        print(f"Upscaling {inpath} tiff to file: {outpath}")
        proc.wait()

    def clouds(self, geom_path=None):
        if self.cloud_percentage == 0:
            if "SCL" in self.bands:
                print("Cloud processing")
                self._generate_mask(True, geom_path)
            else:
                print("No clouds band downloaded")
        return self.cloud_percentage

    def combine_data(self, count_clouds=False, save_scl=False):
        if self.md.separate_bands == False:
            for i, scl in enumerate(self._scale):
                if scl == 1:
                    break
            for l, b in enumerate(self.bands):
                out_filename = os.path.join(self.out_dir, f"{b}_merged.tiff")
                if os.path.exists(out_filename):
                    print(f"File {out_filename} exists, skipping generation")
                    self._processed_files[l] = out_filename
                    continue
                if self._scale[l] == 1:
                    shutil.copyfile(self.md.processed_files[l], out_filename)
                else:
                    sizex = self._profiles[i]["width"]
                    sizey = self._profiles[i]["height"]
                    self._upscale(self.md.processed_files[l], out_filename, sizex, sizey)
                self._processed_files[l] = out_filename
        else:
            if "SCL" in self.bands:
                print("Cloud processing")
                self._generate_mask(count_clouds=count_clouds, save_scl=save_scl)
                print("Data processing")
                self._process_data()
                self.mask = None
            else:
                pass


class MergedProcessing:  # for processing data to indecies
    def __init__(self, mgd: MergedDataset, read_window=None):
        self.mgd = mgd
        self.read_window = read_window
        if read_window is None:
            self.read_window = self.mgd.read_window

    def _read_profile(self, band):
        for i, fl in enumerate(self.mgd._processed_files):
            if band in fl:
                return self.mgd._profiles[i]

    def _compress_tiff(self, in_file, out_file):
        cmd = f'gdalwarp -co NUM_THREADS=ALL_CPUS -co compress=lzw -co predictor=2 {in_file} {out_file} '
        proc = subprocess.Popen(shlex.split(cmd))
        print(f"Compressing tiff {in_file}")
        proc.wait()

    def _window_read(self, band, profile):
        for i, fl in enumerate(self.mgd._processed_files):
            if band in fl:
                wd = self.read_window
                ws = np.linspace(0, profile["width"], np.round(profile["width"]/wd).astype(int)).astype(int)
                hs = np.linspace(0, profile["height"], np.round(profile["height"]/wd).astype(int)).astype(int)
                msk = None
                for x in range(1, len(ws)):
                    part = None
                    for y in range(1, len(hs)):
                        with rasterio.open(fl) as f:
                            data = f.read(window=Window(ws[x-1], hs[y-1], ws[x] - ws[x-1], hs[y] - hs[y-1]))
                        yield data.squeeze(), x, y

    def SCL(self):
        self.scl_path = os.path.join(self.mgd.out_dir, "SCL_merged.tiff")
        return self.scl_path

    def NDVI(self):
        self.ndvi_path = os.path.join(self.mgd.out_dir, "NDVI.tiff")
        if os.path.exists(self.ndvi_path):
            print(f"File already exists {self.ndvi_path}")
            return self.ndvi_path
        pf = self._read_profile("B04")
        ndvi = np.zeros((pf["height"], pf["width"]), dtype="float32")
        wd = self.read_window
        ws = np.linspace(0, pf["width"], np.round(pf["width"]/wd).astype(int)).astype(int)
        hs = np.linspace(0, pf["height"], np.round(pf["height"]/wd).astype(int)).astype(int)
        b4 = self._window_read("B04", pf)
        b8 = self._window_read("B08", pf)
        while True:
            try:
                b4_data, x, y = next(b4)
                b8_data, x, y = next(b8)
                ndvi[hs[y-1]:hs[y], ws[x-1]:ws[x]] = (b8_data - b4_data) / (b8_data + b4_data)
            except StopIteration:
                break
        ndvi[ndvi > 1] = 1
        ndvi[ndvi < 0] = 0
        ndvi *= 255
        ndvi = np.astype(ndvi, "uint8")
        self.mgd._save_tiff(ndvi, self.ndvi_path)
        return self.ndvi_path
        # self._compress_tiff(os.path.join(self.mgd.out_dir, "NDVI.tiff"), os.path.join(self.mgd.out_dir, "NDVI_compressed.tiff"))


    def LAI(self):
        self.lai_path = os.path.join(self.mgd.out_dir, "LAI.tiff")
        if os.path.exists(self.lai_path):
            print(f"File already exists {self.lai_path}")
            return self.lai_path
        pf = self._read_profile("B04")
        lai = np.zeros((pf["height"], pf["width"]), dtype="float32")
        msk = np.zeros((pf["height"], pf["width"]), dtype="float32")
        wd = self.read_window
        ws = np.linspace(0, pf["width"], np.round(pf["width"]/wd).astype(int)).astype(int)
        hs = np.linspace(0, pf["height"], np.round(pf["height"]/wd).astype(int)).astype(int)
        b3 = self._window_read("B03", pf)
        b4 = self._window_read("B04", pf)
        b5 = self._window_read("B05", pf)
        b6 = self._window_read("B06", pf)
        b7 = self._window_read("B07", pf)
        b8a = self._window_read("B8A", pf)
        b11 = self._window_read("B11", pf)
        b12 = self._window_read("B12", pf)
        azim = self._window_read("azimuth", pf)
        zen = self._window_read("zenith", pf)
        print("start calc")
        while True:
            try:
                b3_data, x, y = next(b3)
                b4_data, x, y = next(b4)
                b5_data, x, y = next(b5)
                b6_data, x, y = next(b6)
                b7_data, x, y = next(b7)
                b8a_data, x, y = next(b8a)
                b11_data, x, y = next(b11)
                b12_data, x, y = next(b12)
                azim_data, x, y = next(azim)
                zen_data, x, y = next(zen)
                lai[hs[y-1]:hs[y], ws[x-1]:ws[x]] = LAI_calc(b3_data, b4_data, b5_data, b6_data, b7_data, b8a_data, b11_data, b12_data, zen_data, azim_data)
                msk[hs[y-1]:hs[y], ws[x-1]:ws[x]][b3_data != 0] = 1 
            except StopIteration:
                print("stop iter")
                break
        print("scale")
        print(np.average(lai))
        lai[lai > 3] = 3
        lai[lai < 0] = 0
        lai *= (255/3)
        lai = np.astype(lai, "uint8")
        lai[msk == 0] = 0
        print(lai.min(), lai.max())
        self.mgd._save_tiff(lai, self.lai_path)
        return self.lai_path

    def NBRP(self):  # NBR+ https://www.mdpi.com/2072-4292/14/7/1727
        self.nbrp_path = os.path.join(self.mgd.out_dir, "NBRP.tiff")
        if os.path.exists(self.nbrp_path):
            print(f"File already exists {self.nbrp_path}")
            return self.nbrp_path
        pf = self._read_profile("B08")
        nbr = np.zeros((pf["height"], pf["width"]), dtype="float32")
        wd = self.read_window
        ws = np.linspace(0, pf["width"], np.round(pf["width"]/wd).astype(int)).astype(int)
        hs = np.linspace(0, pf["height"], np.round(pf["height"]/wd).astype(int)).astype(int)
        b8 = self._window_read("B08", pf)
        b12 = self._window_read("B12", pf)
        # print(self.mgd._processed_files)
        while True:
            try:
                b8_data, x, y = next(b8)
                b12_data, x, y = next(b12)
                # nbr[hs[y-1]:hs[y], ws[x-1]:ws[x]] = (b12_data - b8a_data - b3_data - b2_data) / (b12_data + b8a_data + b3_data + b2_data)
                nbr[hs[y-1]:hs[y], ws[x-1]:ws[x]] = (b8_data - b12_data) / (b8_data + b12_data)
            except StopIteration:
                print("stop iter")
                break

        nbr[nbr > 1] = 1
        nbr[nbr < 0] = 0
        nbr *= 255
        nbr = np.astype(nbr, "uint8")
        self.mgd._save_tiff(nbr, self.nbrp_path)
        return self.nbrp_path

    def MSI(self):
        self.msi_path = os.path.join(self.mgd.out_dir, "MSI.tiff")
        if os.path.exists(self.msi_path):
            print(f"File already exists {self.msi_path}")
            return self.msi_path
        pf = self._read_profile("B08")
        msi = np.zeros((pf["height"], pf["width"]), dtype="float32")
        wd = self.read_window
        ws = np.linspace(0, pf["width"], np.round(pf["width"]/wd).astype(int)).astype(int)
        hs = np.linspace(0, pf["height"], np.round(pf["height"]/wd).astype(int)).astype(int)
        b8 = self._window_read("B08", pf)
        b11 = self._window_read("B11", pf)
        while True:
            try:
                b8_data, x, y = next(b8)
                b11_data, x, y = next(b11)
                msi[hs[y-1]:hs[y], ws[x-1]:ws[x]] = b11_data / b8_data
            except StopIteration:
                print("stop iter")
                break
        msi[msi > 3] = 3
        msi[msi < 0] = 0
        msi *= 255
        msi = np.astype(msi, "uint8")
        self.mgd._save_tiff(msi, self.msi_path)
        return self.msi_path