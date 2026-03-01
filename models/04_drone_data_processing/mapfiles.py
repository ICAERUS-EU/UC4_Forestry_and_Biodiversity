import os
import json
import shutil
import argparse
import requests
import rasterio
import math
import numpy as np
from rasterio.features import shapes
from shapely.geometry import shape
from shapely import geometry
from shapely.ops import transform
import shapely 
import pyproj


EPSG = "__EPSG__"
EXTENT_1 = "__minx__"
EXTENT_2 = "__miny__"
EXTENT_3 = "__maxx__"
EXTENT_4 = "__maxy__"
SIZE_X = "__sizex__ "
SIZE_Y = "__sizey__"
MAXSIZE = "__maxsize__"
RGB = "__RGB_PATH__"

verify=True


def generate_geometry(tiff):
    """
    Generates convex hull and center point of the tiff, converts to epsg 4326, and return both as geojson
    """
    with rasterio.open(tiff) as src:
        img = src.read()
        mask = src.read_masks()
        mask[np.isnan(img)] = 0
        img[~np.isnan(img)] = 1
        crs = src.crs
        shape_gen = (shape(s) for s, v in shapes(img, mask=mask, transform=src.transform))
    mpl = geometry.MultiPolygon(list(shape_gen))
    mpl = mpl.convex_hull
    center = mpl.centroid

    project = pyproj.Transformer.from_crs(pyproj.CRS(crs), pyproj.CRS('EPSG:4326'), always_xy=True).transform
    return shapely.to_geojson(transform(project, mpl)), transform(project, center)


def generate_mapfile(tiff, mapserver_tiff_path, tiff_path):
    # generate mapfile from the tiffs provided and the default template
    dir_path = os.path.dirname(os.path.realpath(__file__))
    template_path = os.path.join(dir_path, "templates/mapfile-hyper.map")
    out_dir = os.path.join(dir_path, "mapfiles/Hyper")
 
    out_path = os.path.join(out_dir, tiff_path)
    out_path = out_path.replace(".tiff", ".map")
    # create folder tree if non existant
    end_dir = os.path.dirname(out_path)
    os.makedirs(end_dir, exist_ok=True)
    # create changes dict (k - template string, value - replace string)
    with open(template_path) as f:
        lines = f.readlines()

    dataset = rasterio.open(tiff)

    # fix tiff file paths, from local to AWS s3 bucket key
    changes = {EXTENT_1: str(math.floor(dataset.bounds.left)),
               EXTENT_2: str(math.floor(dataset.bounds.bottom)),
               EXTENT_3: str(math.ceil(dataset.bounds.right)),
               EXTENT_4: str(math.ceil(dataset.bounds.top)),
               SIZE_X: str(dataset.shape[1]) + " ",
               SIZE_Y: str(dataset.shape[0]),
               MAXSIZE: str(max(dataset.shape)),
               RGB: mapserver_tiff_path,  # neeed to be changed if the files is moved. Shows the path for mapserver where to get the file.
               EPSG: dataset.read_crs().to_string().lower()}

    newlines = lines
    for k, v in changes.items():
        tmp = []
        for ln in newlines:
            if k in ln:
                tmp.append(ln.replace(k, v))
            else:
                tmp.append(ln)
        newlines = tmp

    with open(out_path, "w") as f:
        for ln in newlines:
            f.write(ln)
    return out_path, changes[EPSG], [changes[EXTENT_1], changes[EXTENT_2], changes[EXTENT_3], changes[EXTENT_4]], dataset.shape[1], dataset.shape[0]


def main(params):
    """
    Mapfile generation process
    """
    
    tiff_path = params["path"]
    out_path = params["outpath"]
    
    if not os.path.isfile(tiff_path):
        print(f"{tiff_path} original tiff file doesnt exist, Error")
        return False

    if not params["rewrite"]:
        if os.path.isfile(tiff_dest):
            print(f"{tiff_dest} tiff exists in destination, skipping creation")
            return True
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Create mapfile
    mapfile_path, epsg, extent, w, h = generate_mapfile(tiff_path, tiff_path, out_path)
    # gather parameters for loading wms mapserver layer
    hull, center = generate_geometry(tiff_path)
    load_parameters = {"1": {"format": "PNG", "layers": "RGB"}, 
                       "crs": epsg.upper() , 
                       "bbox": ",".join(extent), 
                       "width": w, 
                       "height": h, 
                       "styles": "", 
                       "request": "GetMap", 
                       "service": "WMS", 
                       "version": "1.3", 
                       "transparent": "true"}
    load_parameters = json.dumps(load_parameters)
    print(load_parameters, hull, center)
    
    return load_parameters, hull, center


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to processed hyperspectral tiff file')
    parser.add_argument('--outpath', type=str, required=True, help='Path to output generated mapfile')
    parser.add_argument('--rewrite', action='store_true', help='Enable existing file overwriting')

    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    main(vars(opt))

