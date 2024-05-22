import numpy as np
from .base import BaseSentinelWriter
import rasterio
from PIL import Image


class FullTiffWriter(BaseSentinelWriter):
    """
    Raster (Geotiff) writer using rasterio.

    Writes the data using the same dtype of input data. Uses rasterio profile from metadata
    """
    def __init__(self, out_path, compress=True):
        self.out_path = out_path
        self.compress = compress

    def store(self):
        pass

    def write(self, X, profile,  mask=None):
        assert len(X.shape) == 3, "Input data has to be 3D of shape: X, Y, bands"
        assert np.argmin(X.shape) == 0, "Input data has to be 3D of shape: bands, X, Y for rasterio"
        if X.dtype == np.dtype("float"):  # reduce the float accuracy for image saving
            X = X.astype("float32")
        meta = profile
        meta['count'] = X.shape[0]
        # TODO: possible error may appear creating geotiffs from envi files
        meta['dtype'] = X.dtype
        meta['driver'] = "GTiff"
        meta["interleave"] = "band"
        if self.compress:
            meta["compress"] = "lzw"
            meta["predictor"] = 2
        with rasterio.open(self.out_path, "w", **meta) as f:
            f.write(X)
            if mask is not None:
                f.write_mask(mask)


class RGBWriter(BaseSentinelWriter):
    """
    PNG/JPG writer using 3 band numpy array. Using PIL and uint8 array
    
    Parameters:
        path: str
            output file full path (.png or .jpg)
    """
    def __init__(self, path, scale=1):
        self.path = path
        self.scale = scale

    def store(self):
        pass

    def write(self, X):
        assert len(X.shape) == 3, "Input data has to be 3D of shape: X, Y, 3"
        assert np.argmin(X.shape) == 2, "Input data has to be 3D of shape: X, Y, 3 for rgb image creation"
        assert X.shape[-1] == 3
        assert X.dtype == np.uint8
        img = Image.fromarray(X, 'RGB')
        if self.scale!= 1:
            img = img.resize((int(img.size[0] * self.scale), int(img.size[1] * self.scale)), Image.Resampling.LANCZOS)
        img.save(self.path)
        return True


