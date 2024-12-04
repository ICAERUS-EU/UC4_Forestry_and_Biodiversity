from abc import ABC, abstractmethod
from sentinelhub.aws.request import AwsTileRequest
from sentinelhub.constants import CRS
from sentinelhub.data_collections import DataCollection
from typing import Optional, List
import shutil
import os
import rasterio
import numpy as np
import math
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
import pyproj
from shapely.ops import transform
import fiona
from rasterio.transform import rowcol
from rasterio.windows import Window


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
    def __init__(self, geometry: BaseGeometry, crs: CRS = CRS.WGS84, prop = None):
        self.geometry = geometry  # one or more collections of lon,lat points
        self.prop = prop
        self.crs = crs
        self.utm = None 
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
    def bbox(self):
        if self._bbox is None:
            self._gen_bbox()
        return self._bbox

    @classmethod
    def from_shapefile(cls, shp):
        """
        Create geometry from shapefile directly
        """
        with fiona.open(shp, "r") as shapefile:
            crs = shapefile.crs
            if len(shapefile) > 1:  # multipolygon (multipoint)
                geometry = MultiPolygon([shape(pol['geometry']) for pol in shapefile])
                prop = [pol['properties'] for pol in shapefile]
            else:
                geometry = Polygon(shape(shapefile[0]['geometry'])) 
                prop = shapefile[0]['properties']
        return cls(geometry, CRS(crs), prop)

    @staticmethod
    def convert_wgs_to_utm(lon: float, lat: float):
        return CRS.get_utm_from_wgs84(lon, lat)

    def buffer(self, distance, cap_style="round"):
        if self.utm != self.crs:
            project = pyproj.Transformer.from_crs(self.crs.epsg, self.utm.epsg, always_xy=True).transform
            new_geo = transform(project, self.geometry)
            new_geo = new_geo.buffer(distance, cap_style=cap_style)
        else:
            new_geo = self.geometry.buffer(distance, cap_style=cap_style)
        return new_geo

    def transform(self, new_crs: CRS):
        if self.crs != new_crs:
            project = pyproj.Transformer.from_crs(self.crs.epsg, new_crs.epsg, always_xy=True).transform
            new_geo = transform(project, self.geometry)
            return Geometry(new_geo, new_crs, self.prop)
        else:
            print("Transformation skipped CRS is already ", self.crs)
            return self


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

    def clear(self):
        """
        Delete the files and clear the data.
        """
        self.delete(True)
        self._data = None
        self.files = None
        self._profiles = None

    def __repr__(self):
        return f'Tile(tile={self.tile}, date={self.date}, index={self.index})'

