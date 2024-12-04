from contextlib import contextmanager
from typing import Optional
import numpy as np
from .base import BaseSentinelProcessor, Geometry, Tile
from rasterio.io import MemoryFile
import rasterio
import rasterio.mask
import geopandas as gpd


@contextmanager
def mem_raster(data, profile):

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset: # Open as DatasetWriter
            dataset.write(data)
            del data

        with memfile.open() as dataset:  # Reopen as DatasetReader
            yield dataset  # Note yield not return


class CombineProcessor(BaseSentinelProcessor):
    """
    Stack files from the Tile into a matrix.

    Parameters
    ----------
    metadata : Tile
        The metadata of the tile.
    order : list, optional
        The order of the files to stack. Default is None. List of indices of the files to stack.
    """
    def __init__(self, transpose=False):
        self.transpose = transpose

    def process(self, metadata: Tile, profile: dict, order: Optional[list]=None):
        data = None
        if order is not None:
            for o in order:
                if data is None:
                    data = metadata.data[o]
                    assert len(data.shape) == 3  
                    assert np.argmin(data.shape) == 0  # assert that dim 0 is bands dim
                else:
                    data = np.concatenate((data, metadata.data[o]), axis=0)
        else:
            for dt in metadata.data:
                if data is None:
                    data = dt
                    assert len(data.shape) == 3  
                    assert np.argmin(data.shape) == 0  # assert that dim 0 is bands dim
                else:
                    data = np.concatenate((data, dt), axis=0)
        if self.transpose:
            data = np.transpose(data, (1, 2, 0))
        profile["count"] = data.shape[np.argmin(data.shape)]
        return data, profile


class TransposeProcessor(BaseSentinelProcessor):
    def process(self, data: np.ndarray):
        assert len(data.shape) == 3
        if np.argmin(data.shape) == 0:
            data = np.transpose(data, (1, 2, 0))
            return data
        if np.argmin(data.shape) == 2:
            data = np.transpose(data, (2, 0, 1))
            return data


class BufferProcessor(BaseSentinelProcessor):
    """
    Buffer the image by a given distance.

    Parameters
    ----------
    distance : float
        The distance of buffer.
    metric : bool
        Whether the distance is in metric or not.
    """
    def __init__(self, quad_segs = 8):
        self.quad_segs = quad_segs

    def process(self, polygon: Geometry, distance: float, metric: bool = True):
        if polygon.utm is None and metric:
            raise AttributeError("BufferProcessor only works for metric CRS.")
        new_poly = polygon.geometry.buffer(distance, quad_segs = self.quad_segs)
        if polygon.utm is not None:
            return Geometry(new_poly, polygon.utm)
        else:
            return Geometry(new_poly, polygon.crs)


class RGBProcessor(BaseSentinelProcessor):
    """
    Check for RGB tiles in Tile and return RGB image matrix (unprocessed).
    """
    def process(self, data: Tile):
        red_idx = -1
        green_idx = -1
        blue_idx = -1
        for i, fl in enumerate(data.files):
            if "B04" in fl:
                red_idx = i
            if "B03" in fl:
                green_idx = i
            if "B02" in fl:
                blue_idx = i 
        if red_idx == -1 or green_idx == -1 or blue_idx == -1:
            raise ValueError("RGBProcessor requires B04, B03, and B02 bands (R G B bands).")
        d_red = data.data[red_idx]
        d_green = data.data[green_idx]
        d_blue = data.data[blue_idx]
        if d_red.shape!= d_green.shape or d_green.shape!= d_blue.shape:
            raise ValueError("RGBProcessor requires B04, B03, and B02 bands to have the same shape.")
        if np.argmin(d_red.shape) == 0:
            mat = np.stack([d_red, d_green, d_blue], axis=0).squeeze()
            # transpose to x, y, 3
            mat = np.transpose(mat, (1, 2, 0))
        else:
            mat = np.stack([d_red, d_green, d_blue], axis=-1).squeeze()
        return mat
        # save to file (in RGBwriter)


class CropProcessor(BaseSentinelProcessor):
    """
    Crop the image to the extent of the polygon. Polygon and image must be in the same CRS.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon to crop the image to.
    """
    def process(self, data: np.ndarray, profile: dict, polygon: Geometry):
        assert len(data.shape) == 3
        with mem_raster(data, profile) as src:
            out_image, out_transform = rasterio.mask.mask(src, polygon, crop=True, all_touched=True)
            out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
        return out_image, out_meta


class RGBCropProcessor(BaseSentinelProcessor):
    """
    Crop the image to the extent of the polygon. Polygon and image must be in the same CRS.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        The polygon to crop the image to.
    """
    def process(self, data: np.ndarray, profile: dict, polygon: Geometry):
        assert len(data.shape) == 3
        assert data.shape[-1] == 3
        data = np.transpose(data, (2, 0, 1))
        profile['count'] = 3
        with mem_raster(data, profile) as src:
            out_image, out_transform = rasterio.mask.mask(src, polygon, crop=True, all_touched=True)
            out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "count": 3,
                             "transform": out_transform})
        out_image = np.transpose(out_image, (1, 2, 0))
        return out_image, out_meta


class U16U8Processor(BaseSentinelProcessor):
    """
    Convert U16 to U8
    """
    def __init__(self, scale=False):
        self.scale = scale

    def process(self, X):
        assert X.dtype == np.uint16
        X = X / 10000 
        X = X * 255
        if self.scale:
            X += (255 - X.max())
        return (np.rint(X)).astype(np.uint8)


class APIProcessor(BaseSentinelProcessor):

    def __init__(self, ip):
        self.ip = ip


class GridProcessor(BaseSentinelProcessor):
    """
    Grid processor. Create point grid over a geometry. Use UTM geometries for accurate pixel sizes

    Parameters
       geo: Geometry
           Base geometry to create the grid from
       spacing: int, default 10 
           Grid spacing aka resulting pixel size. Use 10 m for Sentinel

    Returns
       Geopandas GeoDataFrame
    """
    def process(self, geo: Geometry, spacing = 10):
        assert geo.crs == geo.utm, "Geometry must be in UTM projection"
        
        xmin, ymin, xmax, ymax = geo.geometry.bounds
        xcoords = [c for c in np.arange(xmin, xmax, spacing)] #Create x coordinates
        ycoords = [c for c in np.arange(ymin, ymax, spacing)] #And y

        coordinate_pairs = np.array(np.meshgrid(xcoords, ycoords)).T.reshape(-1, 2) #Create all combinations of xy coordinates
        geometries = gpd.points_from_xy(coordinate_pairs[:,0], coordinate_pairs[:,1]) #Create a list of shapely points

        pointdf = gpd.GeoDataFrame(geometry=geometries, crs=geo.crs.epsg) #Create the point df

        return pointdf, len(xcoords), len(ycoords)


class CloudProcessor(BaseSentinelProcessor):
    """
    Generate cloud filter mask array
    """

    def process(self, clouds: np.ndarray):
        mask = np.zeros_like(clouds)
        mask[np.logical_and(clouds > 3, clouds < 7)] = 1  # default vegetation and water filter
        return mask


class UpscaleProcessor(BaseSentinelProcessor):
    """
    Upscale the image by integer scale factor 
    """
    def __init__(self, axis=[0, 1]):
        self.axis = axis
    
    def process(self, image: np.ndarray, scale_factor: int):
        assert type(scale_factor) == int, "Scale factor must be an integer"
        assert scale_factor > 0, "Scale factor must be greater than 0"
        assert max(self.axis) < len(image.shape), "Selected axis outside matrix range"
        for a in self.axis:
            image = image.repeat(scale_factor, axis=a)
        return image

    
class ProfileUpscaleProcessor(BaseSentinelProcessor):
    """
    Upsacle tiff profile by integer factor
    """

    def process(self, profile: dict, scale_factor: int):
        assert type(scale_factor) == int, "Scale factor must be an integer"
        assert scale_factor > 0, "Scale factor must be greater than 0"
        affine = profile['transform']
        a, b, c, d, e, f = affine.a, affine.b, affine.c, affine.d, affine.e, affine.f
        a = a / scale_factor
        e = e / scale_factor
        new = rasterio.Affine(a, b, c, d, e, f)
        profile['transform'] = new
        profile['width'] = profile['width'] * scale_factor
        profile['height'] = profile['height'] * scale_factor
        return profile
