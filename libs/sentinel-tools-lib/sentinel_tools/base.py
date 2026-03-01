from abc import ABC, abstractmethod
from .crs import CRS
from typing import Optional, List
import shutil
import os
import rasterio
import numpy as np
import math
from shapely.geometry import MultiPolygon, Polygon, shape, mapping, Point, MultiPoint, box
from shapely.geometry.base import BaseGeometry
import pyproj
from shapely.ops import transform
import fiona
from rasterio.transform import rowcol
from rasterio.windows import Window
from fiona.crs import CRS as FionaCRS
import json
from shapely import intersects, intersection, union, from_geojson
from baselib.base import ProcessorMixin
import hashlib


class BaseSentinelReader(ABC):
    """
    Base spectral data reader class for different, hyperspectral and other types of spectra, readers

    Requires the wavelenght and metadata to be read.
    """

    def __init__(self):
        self._wavelength = None
        self._metadata = None

    @abstractmethod
    def _load_metadata(self):
        pass

    @property
    @abstractmethod
    def metadata(self):
        """
        Base metadata property

        Return metadata about the spectral data read
        """
        return self._metadata
    

class BaseSentinelLoader(ABC):
    """
    Base spectral data loader.

    Implements the load function to read the data to memory using different methods.

    Gets metadata about the required files to be read from BaseSentinelReader classes.
    """

    def __init__(self):
        self._data = None
    
    @abstractmethod
    def _loader(self):
        """
        Data loader using the method implemented (eg. Full load at once, partial pixel load, partial band load, single file load) 

        Implement different methods depending on the need and the metadata provided from the BaseSentinelReader class and internal state

        return read data
        """
        pass

    @abstractmethod
    def load(self, metadata):
        """
        Loader function that read required files unsing the metadata provided

        """
        yield self._loader(metadata)

    @property
    @abstractmethod
    def data(self):
        return self._data


class BaseSentinelProcessor(ABC):
    """
    Base spectral data processing class

    Takes data from BaseSentinelLoader

    return processed data, agnostic to the BaseSentinelLoader state, uses only wavelength if required (eg. for rgb conversion).
    """
    @abstractmethod
    def process(self, X, y=None, wavelength=None):
        pass


class BaseSentinelWriter(ABC):
    """
    Base spectral data writer

    Takes data from BaseSentinelProcessor, status data from BaseSentinelLoader and metadata from BaseSentinelReader
    """

    def __init__(self, metatada):
        self.metadata = metatada
        self._store = None

    @abstractmethod
    def store(self, X, status):
        """
        Temporary data storage function, if the data has to be written all at once (eg. Geotiff raster or png image)

        Stores data in memory according to the status and metadata provided.

        Is used by the write funtion to finalize writing
        """
        pass

    @abstractmethod
    def write(self, X, status, parameters):
        """
        Data writer function

        Implement data store check if data store is used

        Uses metadata from BaseSentinelReader to write files
        
        Additional parameters required (output path, data types or other)
        """
        if self._store is None:
            pass
        else:
            pass


class BaseIndexProcessor(ABC):
    """
    Base spectral index calculation class

    Takes input data as separate bands required be the index.
    """

    @abstractmethod
    def process(self, band1, band2):
        pass


class BaseSentinelModel(ABC):
    """
    Base model class for creating hyperspectral data processing models

    Based on sklearn exmaples. Implements predict function to run in the processing pipeline.
    """

    @abstractmethod
    def __init__(self, parameters=None):
        """
        List parameter separately or as a dictionary
        """
        self.parameters = parameters

    @abstractmethod
    def predict(self, X, y=None):
        pass


class BasePipeline(ABC):
    """
    Base pipeline class to describe/create and run the pipeline.
    """
    @abstractmethod
    def __init__(self, in_path, out_path, overwrite=False) -> None:
        super().__init__()

    @abstractmethod
    def run(self, calibration=None):
        pass


class APIBase():
    """
    Base API class to use as wrapper for data read and write to an API (hyperlabeling or datastore)
    """

    def __init__(self, ip, token=None):
        self.ip = ip
        if token is None:
            self.token = "WIP" 
        else:
            self.token = token
        self.headers = {'Authorization': 'Token ' + self.token}


class Geometry():
    """
    Base geometry class to describe/create and run the geometry.

    Parameters
    ----------
    geometry : BaseGeometry
        Geometry object to be used for the pipeline.
    crs : CRS
        Coordinate reference system to be used for the pipeline.
    """
    def __init__(self, geometry: BaseGeometry, crs: CRS = CRS.WGS84, prop = None, geom_type = None):
        self.geometry = geometry  # one or more collections of lon,lat points
        self.geom_type = geom_type
        self.prop = prop
        self.crs = crs
        self.utm = None
        self._hashval = None
        self._bbox = None
        self._utm()

    def __repr__(self):
        return f"Geometry: {self.geometry.geom_type} - {self.crs} - {self.utm}"

    def _utm(self):
        if str(self.crs.epsg)[:3] == "326" or str(self.crs.epsg)[:3] == "327":
            self.utm = self.crs
        elif str(self.crs.epsg) == "4326":
            center = self.geometry.centroid
            self.utm = self.convert_wgs_to_utm(center.x, center.y)
        else:
            self.utm = None

    def _gen_bbox(self):
        if self.crs != CRS.WGS84:
            project = pyproj.Transformer.from_crs(self.crs.epsg, CRS.WGS84.epsg, always_xy=True).transform
            bbox = transform(project, self.geometry)
            bbox = bbox.bounds
        else:
            bbox = self.geometry.bounds
        self._bbox = bbox

    @property
    def hashval(self):
        if self._hashval is None:
            if self._bbox is None:
                self._gen_bbox()
            self._hashval = hashlib.md5(''.join([str(x) for x in self._bbox]).encode()).hexdigest()[-32:]
        return self._hashval

    @property
    def bbox_polygon(self):
        if self._bbox is None:
            self._gen_bbox()
        bounds_pgon = box(self._bbox[0], self._bbox[1], self._bbox[2], self._bbox[3])
        return bounds_pgon

    @property
    def bbox(self):
        if self._bbox is None:
            self._gen_bbox()
        return self._bbox

    @classmethod
    def from_shapefile(cls, shp, geom_type="polygon"):
        """
        Create geometry from shapefile directly
        """
        with fiona.open(shp, "r") as shapefile:
            crs = shapefile.crs
            if geom_type == "polygon":
                if len(shapefile) > 1:  # multipolygon (multipoint)
                    geometry = MultiPolygon([shape(pol['geometry']) for pol in shapefile])
                    prop = [pol['properties'] for pol in shapefile]
                    geom_type = "multipolygon"
                else:
                    geometry = Polygon(shape(shapefile[0]['geometry'])) 
                    prop = shapefile[0]['properties']
            if geom_type == "point":
                if len(shapefile) > 1:  # multipolygon (multipoint)
                    geometry = MultiPoint([shape(pol['geometry']) for pol in shapefile])
                    prop = [pol['properties'] for pol in shapefile]
                    geom_type = "multipoint"
                else:
                    geometry = Point(shape(shapefile[0]['geometry'])) 
                    prop = shapefile[0]['properties']
        return cls(geometry, CRS(crs), prop, geom_type)

    @classmethod
    def from_geojson(cls, geojson, crs = CRS.WGS84):
        geom_type = None
        if isinstance(crs, int):
            crs = CRS(crs)
        # convert to dict
        if isinstance(geojson, str):
            try:
                geojson = json.loads(geojson)
            except:
                pass
        # extract type and prop
        geom_type = geojson["type"].lower()
        assert len(geojson["features"]) == 1, "Assertion error, geojson has more than one feature"
        part = geojson["features"][0]
        if "prop" in part:
            prop = part["prop"]
        else:
            prop = None
        geometry = from_geojson(json.dumps(part))
        return cls(geometry, crs, prop, geom_type)

    @staticmethod
    def convert_wgs_to_utm(lon: float, lat: float):
        return CRS.get_utm_from_wgs84(lon, lat)

    @staticmethod
    def combine(geoms: list):
        """
        Combine multiple geometries into one.
        :param geoms: List of geometries
        :return:
        """
        assert len(geoms) > 0, "No geometries to combine"
        assert all([geom.crs == geoms[0].crs for geom in geoms]), "Geometries must be in same CRS"
        assert all([geom.geometry.geom_type == 'Polygon' for geom in geoms]), "Geometries must be polygons"
        return Geometry(MultiPolygon([geom.geometry for geom in geoms]), geoms[0].crs, geoms[0].prop)

    def intersect(self, other: BaseGeometry, threshold: float = 0):
        """
        Intersect two geometries together to 1 geometry if possible and if intersection area is bigger than threshold.
        :param other: Geometry to intersect with
        :param threshold: Minimum intersection area compared to smallest geometry.
        :return:
        """
        assert self.crs == other.crs, "Geometries must be in same CRS"
        if intersects(self.geometry, other.geometry):
            if not self.geometry.is_valid:
                self.geometry = self.geometry.buffer(0)
            if not other.geometry.is_valid:
                other.geometry = other.geometry.buffer(0)
            intersec = intersection(self.geometry, other.geometry)
            a1 = self.geometry.area
            if other.geometry.area < a1:  # get smaller area
                a1 = other.geometry.area
            if threshold > 0 and intersec.area < threshold * a1:
                return False
            else:
                return Geometry(union(self.geometry, other.geometry), self.crs, self.prop)
        return False

    def buffer(self, distance, cap_style="round", return_cls=False):
        if self.utm != self.crs:
            project = pyproj.Transformer.from_crs(self.crs.epsg, self.utm.epsg, always_xy=True).transform
            new_geo = transform(project, self.geometry)
            new_geo = new_geo.buffer(distance, cap_style=cap_style)
        else:
            new_geo = self.geometry.buffer(distance, cap_style=cap_style)
        if return_cls:
            gm = Geometry(new_geo, self.utm, prop=self.prop)
            return gm
        else:
            return new_geo

    def transform(self, new_crs: CRS):
        if isinstance(new_crs, int):
            new_crs = CRS(new_crs)
        if self.crs != new_crs:
            project = pyproj.Transformer.from_crs(self.crs.epsg, new_crs.epsg, always_xy=True).transform
            new_geo = transform(project, self.geometry)
            return Geometry(new_geo, new_crs, self.prop)
        else:
            print("Transformation skipped CRS is already ", self.crs)
            return self

    def export_shapefile(self, path, schema):
        """
        WIP: Extensions required for multiple properties
        Export the geometry to a shapefile.

        Parameters
        ----------
        path : str
            Path to the shapefile.
        schema : dict
            Dictionary with the schema of the shapefile.
        """
        if hasattr(self.geometry, '__len__'):
            pass
        else:
            self.geometry = [self.geometry]

        assert "id" in schema["properties"], "Schema must contain an id field"
        with fiona.open(path, "w", crs=FionaCRS.from_epsg(self.crs.epsg), driver='ESRI Shapefile', schema=schema) as c:
            for i in range(len(self.geometry)):
                c.write({
                    "geometry": mapping(self.geometry[i]),
                    "properties": {"id": i}
                })

    def export_geojson(self, path):
        """
        Export the geometry to a GeoJSON file.

        Parameters
        ----------
        path : str
            Path to the GeoJSON file. If None return geojson dict
        """
        if hasattr(self.geometry, '__len__'):
            pass
        else:
            self.geometry = [self.geometry]
        ft = "FeatureCollection"
        if self.geom_type == "polygon":
            ft = "Polygon"
        if self.geom_type == "multipolygon":
            ft = "MultiPolygon"
        if self.geom_type == "point":
            ft = "Point"
        if self.geom_type == "multipoint":
            ft = "MultiPoint"

        data = {"type": ft, "features": []}
        feats = []
        for i in range(len(self.geometry)):
            feats.append({"type": "Feature", "geometry": mapping(self.geometry[i]), "properties": {"id": i}})
        data.update({"features": feats})
        if path is None:
            return data
        else:
            with open(path, "w") as f:
                json.dump(data, f)


class Tile():
    """
    Class for storing metadata of a tile.

    Parameters
    ----------
    tile : str
        Sentinel-2 unique tile identifier.
    date : datetime
        Date of the tile.
    index : int
        AWS tile index.
    cloud_cover : float
        Cloud cover of the tile.
    data_dir : str
        Path to the directory where the data is stored.
    status : int
        Status of the tile. 0 - generated, 1 - prepared, 2 - downloaded
    bands : list
        List of bands.
    """
    def __init__(self, tile, date, index, cloud_cover=None, data_dir: Optional[str]=None, status=0, bands=None):
        from sentinelhub.aws.request import AwsTileRequest
        from sentinelhub.data_collections import DataCollection
        self.tile = tile
        self.date = date
        self.index = index
        self.cloud_cover = cloud_cover
        self.data_dir = data_dir
        self.status = status
        self.bands = bands
        self.files = list()
        self._data = None
        self._profiles = None
        utm = tile[:2]
        lat = tile[2]
        if lat in ["X", "W", "V", "U", "T", "S", "R", "Q", "P", "N"]:
            lat = "326"
        else:
            lat = "327"
        self.epsg = CRS(lat + utm)

    def _request(self, bands=None, force=False):
        if self.data_dir is None:
            raise ValueError(f"Output directory required to download files for tile {self}")
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not hasattr(self, "request"):
            self.request = AwsTileRequest(tile=self.tile, time=self.date, aws_index=self.index, bands=self.bands, data_folder=self.data_dir, data_collection=DataCollection.SENTINEL2_L2A)
        else:
            if force:
                self.request = AwsTileRequest(tile=self.tile, time=self.date, aws_index=self.index, bands=self.bands, data_folder=self.data_dir, data_collection=DataCollection.SENTINEL2_L2A)
        if bands is not None:  # create request with specified bands
            self.request = AwsTileRequest(tile=self.tile, time=self.date, aws_index=self.index, bands=bands, data_folder=self.data_dir, data_collection=DataCollection.SENTINEL2_L2A)

    def download(self, bands=None, force=False):  # specify bands to download if not all initial bands are required
        self._request(bands, force)
        if force:
            self.files = list()
        self.request.save_data()
        for fl in self.request.get_filename_list():
            self.files.append(fl)

    def _read(self, profiles_only=False):
        if self._data is None:
            self._data = []
            self._profiles = []
            for fl in self.files:
                with rasterio.open(os.path.join(self.data_dir, fl)) as f:
                    self._profiles.append(f.profile)
                    if profiles_only:
                        continue
                    else:
                        self._data.append(f.read())

    def sample(self, points: list):
        data = []
        for fl in self.files:
            tmp = []
            with rasterio.open(os.path.join(self.data_dir, fl)) as src:
                x, y = rowcol(src.transform, *zip(*points))
                x = np.array(x)
                xmin = np.min(x)
                xwidth = np.max(x) - xmin 
                y = np.array(y)
                ymin = np.min(y)
                ywidth = np.max(y) - ymin
                x -= xmin
                y -= ymin
                if xmin <= 0 or ymin <= 0:  # check if row,col coords are in the raster (not negative)
                    samples = np.full(len(points), (src.nodata or 0),  dtype=src.dtypes[0])
                elif xmin > src.profile["width"] or ymin > src.profile["height"]:  # check if row,col in raster (not outside the raster scope)
                    samples = np.full(len(points), (src.nodata or 0),  dtype=src.dtypes[0])
                # use windowed read 
                else:
                    all_data = src.read(1, window=Window(ymin, xmin, ywidth+1, xwidth+1))
                    if np.max(x) > all_data.shape[0]:
                        x[x >= all_data.shape[0]] = 0
                        y[x >= all_data.shape[0]] = 0
                    if np.max(y) > all_data.shape[1]:
                        x[y >= all_data.shape[1]] = 0
                        y[y >= all_data.shape[1]] = 0

                    samples = all_data[x, y].T
            data.append(samples)
        return data

    def sample_all(self, points: list):
        data = []
        for fl in self.files:
            with rasterio.open(os.path.join(self.data_dir, fl)) as src:
                x, y = rowcol(src.transform, *zip(*points))
                x = np.array(x)
                xmin = np.min(x)
                xwidth = np.max(x) - xmin 
                y = np.array(y)
                ymin = np.min(y)
                ywidth = np.max(y) - ymin
                x -= xmin
                y -= ymin
                all_data = src.read(1, window=Window(ymin, xmin, ywidth+1, xwidth+1))
                if np.max(x) > all_data.shape[0]:
                    x[x >= all_data.shape[0]] = 0
                    y[x >= all_data.shape[0]] = 0
                if np.max(y) > all_data.shape[1]:
                    x[y >= all_data.shape[1]] = 0
                    y[y >= all_data.shape[1]] = 0

                samples = all_data[x, y].T
            data.append(samples)
        return data

    @property
    def clouds(self) -> np.ndarray:
        data = np.array([0])
        for i, band in enumerate(self.bands):
            if "SCL" in band:
                if self._data is None:
                    self._read()
                data = self._data[i]
        return data

    @property
    def data(self) -> List[np.ndarray]:
        if self._data is None:
            self._read()
        return self._data

    @property
    def profiles(self):
        if self._profiles is None:
            self._read(profiles_only=True)
        return self._profiles

    @property
    def ID(self):
        # return Tile unique identification
        return f"{self.tile}_{self.date}_{self.index}"

    def delete(self, orig_only=False):
        """
        Delete the files of the tiles.

        orig_only: bool
            If True, only the original files are deleted.
        """
        if orig_only:
            for fl in self.files:
                os.remove(os.path.join(self.data_dir, fl))
        else:
            shutil.rmtree(self.data_dir)

    def clear(self, orig_only=True):
        """
        Delete the files and clear the data.
        """
        self.delete(orig_only)
        self._data = None
        self.files = None
        self._profiles = None

    def __repr__(self):
        return f'Tile(tile={self.tile}, date={self.date}, index={self.index})'
