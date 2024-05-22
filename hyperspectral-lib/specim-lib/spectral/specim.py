import rasterio
from rasterio.windows import Window
import numpy as np
import os
from spectral.utils import spectral_calculator
import time
from PIL import Image
import copy
from .base import BaseIndexProcessor, BaseSpectralLoader, BaseSpectralReader, BaseSpectralWriter, BaseSpectralProcessor
import colour
import math


SPECIM_WAVELENGTHS_2X1 = np.array([397.66, 400.28, 402.9, 405.52, 408.13, 410.75, 413.37, 416, 418.62, 421.24, 423.86, 426.49, 429.12, 431.74, 434.37, 437, 439.63, 442.26, 444.89, 447.52, 450.16,
                                   452.79, 455.43, 458.06, 460.7, 463.34, 465.98, 468.62, 471.26, 473.9, 476.54, 479.18, 481.83, 484.47, 487.12, 489.77, 492.42, 495.07, 497.72, 500.37, 503.02, 505.67,
                                   508.32, 510.98, 513.63, 516.29, 518.95, 521.61, 524.27, 526.93, 529.59, 532.25, 534.91, 537.57, 540.24, 542.91, 545.57, 548.24, 550.91, 553.58, 556.25, 558.92,
                                   561.59, 564.26, 566.94, 569.61, 572.29, 574.96, 577.64, 580.32, 583, 585.68, 588.36, 591.04, 593.73, 596.41, 599.1, 601.78, 604.47, 607.16, 609.85, 612.53, 615.23,
                                   617.92, 620.61, 623.3, 626, 628.69, 631.39, 634.08, 636.78, 639.48, 642.18, 644.88, 647.58, 650.29, 652.99, 655.69, 658.4, 661.1, 663.81, 666.52, 669.23, 671.94,
                                   674.65, 677.36, 680.07, 682.79, 685.5, 688.22, 690.93, 693.65, 696.37, 699.09, 701.81, 704.53, 707.25, 709.97, 712.7, 715.42, 718.15, 720.87, 723.6, 726.33, 729.06,
                                   731.79, 734.52, 737.25, 739.98, 742.72, 745.45, 748.19, 750.93, 753.66, 756.4, 759.14, 761.88, 764.62, 767.36, 770.11, 772.85, 775.6, 778.34, 781.09, 783.84, 786.58,
                                   789.33, 792.08, 794.84, 797.59, 800.34, 803.1, 805.85, 808.61, 811.36, 814.12, 816.88, 819.64, 822.4, 825.16, 827.92, 830.69, 833.45, 836.22, 838.98, 841.75, 844.52,
                                   847.29, 850.06, 852.83, 855.6, 858.37, 861.14, 863.92, 866.69, 869.47, 872.25, 875.03, 877.8, 880.58, 883.37, 886.15, 888.93, 891.71, 894.5, 897.28, 900.07, 902.86,
                                   905.64, 908.43, 911.22, 914.02, 916.81, 919.6, 922.39, 925.19, 927.98, 930.78, 933.58, 936.38, 939.18, 941.98, 944.78, 947.58, 950.38, 953.19, 955.99, 958.8, 961.6,
                                   964.41, 967.22, 970.03, 972.84, 975.65, 978.46, 981.27, 984.09, 986.9, 989.72, 992.54, 995.35, 998.17, 1000.99, 1003.81])


class SpecimCubeReader(BaseSpectralReader):
    """
    A class to work with hyperspectral cube data created by specim camera

    """

    def __init__(self, cube_path, wavelength=None, lazy_load=True, cube_extension=".dat", verbose=False):
        """
        Parameters

            cube_path : str
                absolute path to cube header or data file
            wavlengths : np.array
                numpy array of wavelenghts in each spectral index
            lazy_load : boolean
                setting to determine if data is loaded on cube initialization or on data access, if on data access set to True
            cube_extension : str
                parameter to determine the extension that is used for hyperspectral cube data file
        """
        self.cube_path = cube_path
        self.cube_extension = cube_extension
        self.verbose = verbose
        if self.verbose:
            print(f"Cube {self.cube_path} read, metadata: \n {self.metadata}")
        super().__init__()

    def _parse_metadata(self):
        """
        Parse metadata to required format for use in other processing pipeline classes

        Structure:
            {"path": path to cube,
             "header_path": path to the cube header file
             "wavelength": hypersepctral wavelenghts used
             "image_dimensions": [x, y, z]
             "rasterio_profile": profile
        """
        if self._wavelength is None:
            self._load_wavelength()
        tmp = {"path": self.cube_path,
               "header_path": self.cube_header_path,
               "wavelength": self._wavelength,
               "image_dimensions": [self._metadata["width"], self._metadata["height"], self._metadata["count"]],
               "rasterio_profile": self._metadata}
        self._metadata = tmp

    def _load_metadata(self):
        """
        Reads medatada from data file using rasterio and set correct file paths
        """
        path = self.cube_path
        type = os.path.splitext(path)[1][1:].lower()
        if type == "hdr":
            self.cube_header_path = self.cube_path
            self.cube_path = os.path.splitext(path)[0] + self.cube_extension
        else:
            self.cube_header_path = os.path.splitext(path)[0] + ".hdr"
        with rasterio.open(self.cube_path, "r") as fl:
            self._metadata = fl.profile
        self._parse_metadata()

    def _load_wavelength(self):
        """
        Loads the wavelengths for this data cube, uses default if none given
        """
        if self._wavelength is None:
            self._wavelength = SPECIM_WAVELENGTHS_2X1

    @property
    def metadata(self):
        if self._metadata is None:
            self._load_metadata()
        return self._metadata

    @property
    def wavelength(self):
        if self._wavelength is None:
            return self._load_wavelength()
        return self._wavelength


"""
-------------------------- Loaders ---------------------------------
"""


class SpecimFullLoader(BaseSpectralLoader):
    """
    Class loads the whole data cube at one time.

    state: boolean
        Parameter indicates if data is loaded
    """
    
    def __init__(self, metadata=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self._state = False
        self.metadata_keys_required = ["path", "header_path", "rasterio_profile"]
        self._metadata = metadata
        if self._metadata is not None:
            assert all(k in metadata for k in self.metadata_keys_required), f"Metadata key assertion error, not all required keys are set, required keys are: {self.metadata_keys_required}"

    def _loader(self):
        if self.verbose:
            start = time.time()
            print("Loading cube data")
        if self._data is None:
            with rasterio.open(self.cube_path, "r") as fl:
                self._data = fl.read()
            self._data = np.transpose(self._data, (1, 2, 0))
            self._state = True
        if self.verbose:
            print("Cube data loaded in ", time.time() - start)

    def load(self, metadata=None):
        """
        Data loader using the metadata structure from SpecimCubeReader
        Checks the provided metadata with the keys required
        """
        if metadata is not None:
            assert all(k in metadata for k in self.metadata_keys_required), f"Metadata key assertion error, not all required keys are set, required keys are: {self.metadata_keys_required}"
            self._metadata = metadata
        self.cube_path = self._metadata["path"]
        self.cube_header = self._metadata["header_path"]
        self.profile = self._metadata["rasterio_profile"]
        self._loader()

    @property
    def data(self):
        if self._data is not None:
            return self._data
        else:
            if self.verbose:
                print("Data return error, data not loaded. Loading data")
            self.load()
            return self._data

    @property
    def state(self):
        return self._state


class SpecimPartLoader(BaseSpectralLoader):
    """
    Class loads a part of data cube specified by the given list of point pairs [[x_min, x_max, y_min, y_max], ...].
    Uses rasterio windowed reader to read only part of the data

    state: boolean
        Parameter indicates if data is loaded
    """
    
    def __init__(self, metadata=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self._state = False
        self.metadata_keys_required = ["path", "header_path", "rasterio_profile"]
        self._metadata = metadata
        if metadata is not None:
            assert all(k in metadata for k in self.metadata_keys_required), f"Metadata key assertion error, not all required keys are set, required keys are: {self.metadata_keys_required}"

    def _loader(self):
        if self.verbose:
            start = time.time()
            print("Loading cube data")
        if self._data is None:
            with rasterio.open(self.cube_path, "r") as fl:
                self._data = fl.read(window=self.window)
            self._data = np.transpose(self._data, (1, 2, 0))
            self._state = True
        if self.verbose:
            print("Cube data loaded in ", time.time() - start)

    def _window(self):
        assert type(self._cuts) == list or type(self._cuts) == np.ndarray
        if type(self._cuts) == list:
            self._cuts = np.array(self._cuts)
        self._xmin = np.min(self._cuts[:, 0])
        self._xmax = np.max(self._cuts[:, 1])
        self._ymin = np.min(self._cuts[:, 2])
        self._ymax = np.max(self._cuts[:, 3])
        self.window = Window(self._ymin, self._xmin, self._ymax - self._ymin, self._xmax - self._xmin)
        self._cuts[:, 0] -= self._xmin
        self._cuts[:, 1] -= self._xmin
        self._cuts[:, 2] -= self._ymin
        self._cuts[:, 3] -= self._ymin

    def _cutter(self):
        assert type(self._cuts) == list or type(self._cuts) == np.ndarray
        tmp = []
        for ct in self._cuts:
            tmp.append(self._data[ct[0]:ct[1], ct[2]:ct[3], ...])
        self._data = tmp

    def load(self, cuts, metadata=None):
        """
        Data loader using the metadata structure from SpecimCubeReader
        Checks the provided metadata with the keys required
        """
        if metadata is not None:
            assert all(k in metadata for k in self.metadata_keys_required), f"Metadata key assertion error, not all required keys are set, required keys are: {self.metadata_keys_required}"
            self._metadata = metadata
        self._cuts = cuts
        self.cube_path = self._metadata["path"]
        self.cube_header = self._metadata["header_path"]
        self.profile = self._metadata["rasterio_profile"]
        self._window()
        self._loader()
        self._cutter()

    @property
    def data(self):
        if self._data is not None:
            return self._data
        else:
            if self.verbose:
                print("Data return error, data not loaded")
            return False

    @property
    def state(self):
        return self._state


class SpecimBlockLoader(BaseSpectralLoader):
    """
    Class to read specim hyperspectral cube file in blocks, using rasterio Window. Spatial partitioning only, reads all bands at once

    By manual testing discovered that specim cameras save data in blocks of (1, width) (films line by line). Read data in rows.
    """

    def __init__(self, metadata):
        super().__init__()
        self.n = None
        self.verbose = False
        self.metadata = metadata
        self.metadata_keys_required = ["path", "header_path", "image_dimensions", "rasterio_profile"]
        # metadata init check
        assert all(k in metadata for k in self.metadata_keys_required), f"Metadata key assertion error, not all required keys are set, required keys are: {self.metadata_keys_required}"
        self._block_init()

    def _block_init(self):
        with rasterio.open(self.metadata['path']) as src:
            blk = src.block_shapes[0]
        if blk[0] == self.metadata["rasterio_profile"]["height"]:
            blk_h_step = 0
        else:
            blk_h_step = blk[0]

        if blk[1] == self.metadata["rasterio_profile"]["width"]:
            blk_w_step = 0
        else:
            blk_w_step = blk[1]
        # Start with negative position, because iteration updates state first, then data
        self._state = {"block_width": blk[1], "block_height": blk[0], "current_width": 0 - blk_w_step, "current_height": 0 - blk_h_step, "height_step": blk_h_step, "width_step": blk_w_step}

    def __iter__(self):
        self.n = 0
        # iterate over rows and columns
        # TODO: if step width or height is indivisible by step check for errors
        self.iter_h = int(self.metadata["rasterio_profile"]["height"] / self._state['height_step']) if self._state['height_step'] > 0 else 1
        self.iter_w = int(self.metadata["rasterio_profile"]["width"] / self._state['width_step']) if self._state['width_step'] > 0 else 1
        # find max number of iterations and the number of iteration for w and h
        self.max_iter = self.iter_h * self.iter_w
        return self

    def __next__(self):
        """
        Advance iteration, gather new data, return data
        """
        new_state = {}
        if self.n < self.max_iter:
            # iterate width first (if block size == width) then this will be most efficient
            if self.iter_w == 1:
                new_state["current_height"] = self._state['current_height'] + self._state['height_step']
            else:
                if self.n % self.iter_w == 0:  # start of the new row, increment row number
                    new_state["current_height"] = self._state['current_height'] + self._state['height_step']
                    new_state["current_width"] = 0
                else:
                    new_state["current_width"] = self._state['current_width'] + self._state['width_step']
            self.n += 1
            # Update state and clear data
            self._state.update(new_state)
            self._data = None
            # load and return new data
            return self.data, self.state
        else:
            raise StopIteration

    def _loader(self):
        """
        Return data of the current iteration
        """
        if self.verbose:
            start = time.time()
            print("Loading cube data")
        if self._data is None:
            self.window = Window(self._state["current_width"], self._state["current_height"], self._state["block_width"], self._state['block_height'])
            with rasterio.open(self.metadata["path"], "r") as fl:
                self._data = fl.read(window=self.window)
            self._data = np.transpose(self._data, (1, 2, 0))
        if self.verbose:
            print("Cube data loaded in ", time.time() - start)

    def load(self):
        if self.n is None:
            if self.verbose:
                print("Iterator was not initialized, returning iterator")
            return self.__iter__()
        else:
            self._loader()
    
    @property
    def data(self):
        if self._data is None:
            self._loader()
        return self._data

    @property
    def state(self):
        if self._state is None:
            if self.verbose:
                print("State not initialized error creating object")
            return False
        else:
            return self._state


class SpecimFullBandLoader(BaseSpectralLoader):
    """
    Loads the required bands of spectral data cube
    """

    def __init__(self, metadata, bands=None, wavelength_numbers=None):
        super().__init__()
        self.verbose = False
        self.metadata = metadata
        self.metadata_keys_required = ["path", "header_path", "wavelength"]
        self.bands = bands
        self.wavelength_numbers = wavelength_numbers
        # metadata init check
        assert all(k in metadata for k in self.metadata_keys_required), f"Metadata key assertion error, not all required keys are set, required keys are: {self.metadata_keys_required}"
        self._band_init()

    def _wavelength_match(self, wavelength):
        """
        Finds the closes matching hyperspectral index to the wavelenght provided

        Parameters
            wavelength : float
                wavelength that needs to be matched
        """

        return (np.abs(self.metadata["wavelength"] - wavelength)).argmin()

    def _index_generation(self):
        if self.bands is not None:
            self.bands = np.array(self.bands)
        else:
            bnds = []
            for num in self.wavelength_numbers:
                bnds.append(self._wavelength_match(num))
            self.bands = np.array(bnds)
        if self.bands.min() < 0 or self.bands.max() > len(self.metadata["wavelength"]):
            raise AttributeError("The band index range is outside of wavelength count")
        u, counts = np.unique(self.bands, return_counts=True)
        if counts.max() > 1 and self.verbose:
            print("Warning a band (or bands) were selected multiple times. Bands selected: ", self.bands)

    def _band_init(self):
        if self.bands is None and self.wavelength_numbers is None:
            raise AttributeError("For band loader a list of band indicies or list of wavelength numbers (of the same units) is required.")
        self._index_generation()

    def _loader(self):
        """
        Return data of the current iteration
        """
        if self.verbose:
            start = time.time()
            print("Loading cube data")
        if self._data is None:
            with rasterio.open(self.metadata["path"], "r") as fl:
                for index in self.bands:
                    tmp = fl.read(int(index) + 1)  # increment by 1 because rasterio (gdal) band list is 1 indexed
                    if self._data is None:
                        self._data = tmp[:, :, np.newaxis]
                    else:
                        self._data = np.append(self._data, tmp[:, :, np.newaxis], axis=-1)
        if self.verbose:
            print("Cube data loaded in ", time.time() - start)

    def load(self):
        self._loader()

    @property
    def wavelength(self):
        # return band indicies
        return self.bands
    
    @property
    def data(self):
        if self._data is None:
            self._loader()
        return self._data

    @property
    def state(self):
        if self._state is None:
            if self.verbose:
                print("State not initialized error creating object")
            return False
        else:
            return self._state


"""
-------------------------- Processors ---------------------------------
"""


class MaskProcessor(BaseSpectralProcessor):
    def __init__(self, value=0):
        self.value = value

    def process(self, X, y=None, wavelength=None):
        assert len(X.shape) == 3, "Processing error, wrong number of cube dimensions"
        total = np.sum(X, axis=-1)
        return total > self.value



class RGBProcessor(BaseSpectralProcessor):
    """
    Spectral data converter to RGB

    Parameters:
        spectral_axis: int
            Sets the axis that holds spectral data, default last axis -1.
    """

    def __init__(self, wavelength, spectral_axis=-1, max_size=200_000_000, normalize=False):
        self.spectral_axis = spectral_axis
        self.wavelength = wavelength
        self.max_size = max_size
        self.normalize = normalize
        self._init_conversion()

    def _init_conversion(self):
        self.bands = self.wavelength[np.logical_and(380 < self.wavelength, self.wavelength < 780)]
        self.x_cmf = colour.colorimetry.wavelength_to_XYZ(self.bands)[:, 0]
        self.y_cmf = colour.colorimetry.wavelength_to_XYZ(self.bands)[:, 1]
        self.z_cmf = colour.colorimetry.wavelength_to_XYZ(self.bands)[:, 2]

    def process(self, data):
        assert len(data.shape) > 1, "Processing error, provided data is 1D, has to be at least 2D"
        assert data.shape[self.spectral_axis] == len(self.wavelength), "Spectral data and wavelength data size missmatch"
        return colour.XYZ_to_sRGB(self._rgb(data))

    def _rgb(self, data):
        """
        max_size (int): maximum size of array to integrate. if size > max_size cut array to pieces. 200Mb limit default
        """
        sh = data.shape
        if len(data.shape) > 2:
            data = data[..., np.isin(self.wavelength, self.bands)].reshape((sh[0] * sh[1], len(self.bands)))
        else:
            data = data[:, np.isin(self.wavelength, self.bands)]
            
        size = np.prod(data.shape)
        if size > self.max_size:
            image = None
            for arr in np.array_split(data, math.ceil(size / self.max_size)):

                if image is None:
                    image = self._integrate(arr)
                else:
                    image = np.append(image, self._integrate(arr), axis=0)
            if self.normalize:
                image[:, 0] = (image[:, 0] - image[:, 0].min()) / (image[:, 0].max() - image[:, 0].min())
                image[:, 1] = (image[:, 1] - image[:, 1].min()) / (image[:, 1].max() - image[:, 1].min())
                image[:, 2] = (image[:, 2] - image[:, 2].min()) / (image[:, 2].max() - image[:, 2].min())
        else:
            image = self._integrate(data)
            
        if len(sh) > 2:
            image = image.reshape((sh[0], sh[1], 3))
        
        return image

    def _integrate(self, data):
        X = np.trapz(data * self.x_cmf, self.bands.reshape((1, -1)), axis=1)
        if self.normalize:
            image = (X - X.min()) / (X.max() - X.min())
        else:
            image = X
        X = None

        # Y = np.trapz(data * self.y_cmf * 1.2, self.bands.reshape((1, -1)), axis=1)
        Y = np.trapz(data * self.y_cmf, self.bands.reshape((1, -1)), axis=1)
        if self.normalize:
            image = np.vstack((image, (Y - Y.min()) / (Y.max() - Y.min())))
        else:
            image = np.vstack((image, Y))
        Y = None

        Z = np.trapz(data * self.z_cmf, self.bands.reshape((1, -1)), axis=1)
        if self.normalize:
            image = np.vstack((image, (Z - Z.min()) / (Z.max() - Z.min())))
        else:
            image = np.vstack((image, Z))
        Z = None
        return image.T


class SpecimBandExtractor(BaseSpectralProcessor):
    """
    Band extraction processor, returns selected bands in the order provided from input data
    """

    def __init__(self, metadata, bands=None, wavelength_numbers=None):
        super().__init__()
        self.verbose = False
        self.metadata = metadata
        self.metadata_keys_required = ["path", "header_path", "wavelength"]
        self.bands = bands
        self.wavelength_numbers = wavelength_numbers
        # metadata init check
        assert all(k in metadata for k in self.metadata_keys_required), f"Metadata key assertion error, not all required keys are set, required keys are: {self.metadata_keys_required}"
        self._band_init()

    def _wavelength_match(self, wavelength):
        """
        Finds the closes matching hyperspectral index to the wavelenght provided

        Parameters
            wavelength : float
                wavelength that needs to be matched
        """

        return (np.abs(self.metadata["wavelength"] - wavelength)).argmin()

    def _index_generation(self):
        if self.bands is not None:
            self.bands = np.array(self.bands)
        else:
            bnds = []
            for num in self.wavelength_numbers:
                bnds.append(self._wavelength_match(num))
            self.bands = np.array(bnds)
        if self.bands.min() < 0 or self.bands.max() > len(self.metadata["wavelength"]):
            raise AttributeError("The band index range is outside of wavelength count")
        u, counts = np.unique(self.bands, return_counts=True)
        if counts.max() > 1 and self.verbose:
            print("Warning a band (or bands) were selected multiple times. Bands selected: ", self.bands)

    def _band_init(self):
        if self.bands is None and self.wavelength_numbers is None:
            raise AttributeError("For band loader a list of band indicies or list of wavelength numbers (of the same units) is required.")
        self._index_generation()

    def process(self, X):
        assert len(X.shape) == 3, "Input data needs to be 3D."
        return X[:, :, self.bands]


class NDVIProcessor(BaseIndexProcessor):
    """
    NDVI index calculator.

    NDVI formula:
        (nir - red) / (nir + red)
        red = 670 nm, nir = 800 nm
    """

    def process(self, band1, band2, threshold=None):
        """
        Parameters:
            band1: numpy ndarray (X, Y)
                red band (670 nm)

            band2: numpy ndarray (X, Y)
                nir band (800 nm)

            threshold: float
                minimum index value threshold.
        """
        data = (band2 - band1) / (band2 + band1)
        if threshold is None:
            data[data < threshold] = 0
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
        return data


class MinMaxScaler(BaseSpectralProcessor):
    def __init__(self, feature_range=(0, 1), percentiles=None, forced_min=None, forced_max=None):
        """
        Scales the data to specified range

        Parameters:
            feature_range: tuple of ints
                Set the range to which the data should be saceld to, default 0-1; First element < second element

            percentiles: list of int
                Set the wanted percentiles instead of min, max for data scaling, to avoid outlier influence. First element < second element

            forced_min, forced_max: int
                set a min or max value to use, and not get from data
        """
        self.f_min = forced_min
        self.f_max = forced_max
        self._min = None
        self._max = None
        self.feature_range = feature_range
        self.percentiles = percentiles

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    def process(self, X):
        if self.feature_range[0] >= self.feature_range[1]:  # taken from scikit learn library :D
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(self.feature_range)
            )
        if np.isnan(X.max()) or np.isnan(X.min()):
            X = np.nan_to_num(X)
        if self.percentiles is not None:
            self._min = np.percentile(X, self.percentiles[0])
            self._max = np.percentile(X, self.percentiles[1])
        else:
            self._min = X.min()
            self._max = X.max()
        if self.f_min is not None:
            self._min = self.f_min
        if self.f_max is not None:
            self._max = self.f_max
        scaled = (X - self._min) / (self._max - self._min)
        if self.feature_range[0] == 0 and self.feature_range[1] == 1:
            return scaled
        return scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]


class FloatUint8Converter(BaseSpectralProcessor):
    """
    Converts scaled float data to Uint8.
    """

    def process(self, X, verbose=False):
        if X.dtype == np.dtype("float") or X.dtype == np.dtype("float32"):
            if (X.min() < 0 or X.max() > 1) and verbose:
                print("Warning input data is outside of range from 0 to 1. Conversion may be inaccurate")
            return (X * 255).astype("uint8")
        else:
            if verbose:
                print("Conversion error, input data is not of type float64 or float32")


class SpecimCalibrationProcessor(BaseSpectralProcessor):
    """
    Spectral calibration processor.

    Calibrates input data by using the given calibration spectra and reflectance values
    """

    def __init__(self, calibration_spectra, reflectance_percentages, wavelength, calibration_curve_degree=1):
        """
        Calculate cube reflectance from given DRS (Diffuse Reflectance Standard) spectra.
        As a base this was used: https://d-nb.info/1216632197/34
        Main formula R = (raw - dark) / (white - dark)

        Parameters
            calibration_spectra : np.array shape -> (N, WL), (N, M, WL)
                input of calibration spectra in singular array
            reflectance_percentages : np.array shape -> (N), (N, WL)
                the reflectance percentager for each of calibration spectra, maybe different for each wavelength
            calibration_curve_degree : int
                degree of polynomial to fit to the calibration curve. Use 1 for black and white reflectance only, with more calibration curves higher degree may be more accurate.
        """
        self.calibration_spectra = calibration_spectra  # input from SpecimPartLoader list of numpy arrays

        self.reflectance_percentages = np.array(reflectance_percentages)  # same length list of reflectance values (float) for the cut arrays.
        self.wavelength = wavelength
        self.calibration_curve_degree = calibration_curve_degree  # integer value for the polynomial degree. Anything over 1 is WIP :D
        self.wl_axis = -1
        try:
            self.calibration_spectra = np.array(self.calibration_spectra)
        except ValueError:
            cals = []
            for cl in self.calibration_spectra:
                if len(cl.shape) == 2:
                    ax = 0
                else:
                    ax = (0, 1)
                cals.append(np.average(cl, axis=ax))
            self.calibration_spectra = np.array(cals)
        self._calculate_calibration_curves()

    def _calculate_calibration_curves(self):
        shp1 = self.calibration_spectra.shape
        shp2 = self.reflectance_percentages.shape

        if self.calibration_curve_degree > 2:
            # print("Higher degree polynomials are WIP")
            return False

        assert len(shp1) == 2 or len(shp1) == 3, "calibration_spectra shape is out range"
        assert len(shp2) == 1 or len(shp2) == 2, "reflection_percentages shape is out range"
        
        if len(shp2) == 2:
            assert len(shp2[0]) == len(self.wavelength) or len(shp2[1]) == len(self.wavelength), "mismatch between wavelenght counts in cube and relection_precentages data"

        assert shp1[self.wl_axis] == len(self.wavelength), "mismatch between wavelength of calibration spectra and data cube"
        # dark check
        # assert 0 in reflection_percentages, "Dark calibration spectra is required (0% reflectance)"

        # get reflectance calibration functions for each wavelength
        calib = []
        for wl in range(shp1[self.wl_axis]):
            cal_spectra = np.take(self.calibration_spectra, wl, axis=self.wl_axis)
            if len(cal_spectra.shape) > 1:
                cal_spectra = spectral_calculator(cal_spectra, self.wl_axis - 1 if self.wl_axis > 0 else self.wl_axis, "average")
            if len(cal_spectra.shape) == 2:
                z = np.polyfit(cal_spectra[:, wl], self.reflectance_percentages[:, wl], self.calibration_curve_degree)
            else:
                z = np.polyfit(cal_spectra, self.reflectance_percentages, self.calibration_curve_degree)
            calib.append(z.copy())
        calib = np.array(calib)
        self.calibration_functions = calib

        # remove calibration spectra to reduce memory usage
        self.calibration_spectra = None
        self.reflectance_precentages = None

    def process(self, X, wavelength_override=None):
        """
        Run the calibration calculation using the created calibration curves (functions)

        Parameters:
            X: numpy ndarray shape (X, Y, bands)
             
            wavelength_override: list
                Used to override the wavelengths of input data, if partial data is used (not all wavelenghts of the original file).
        """
        if X.dtype != np.dtype("float"):
            X = X.astype("float")
        if wavelength_override is None:
            length = self.wavelength.shape[0]
        else:
            length = len(wavelength_override)
        # calibrate data
        if wavelength_override is not None:
            if self.calibration_functions.shape[-1] == 3:  # quadratic
                for wl in range(length):
                    tmp = np.copy(X[..., wl])
                    X[..., wl] = tmp * tmp * self.calibration_functions[wavelength_override[wl], 0] + tmp * self.calibration_functions[wavelength_override[wl], 1] \
                                + self.calibration_functions[wavelength_override[wl], 2]
            elif self.calibration_functions.shape[-1] == 2:  # linear
                for wl in range(length):
                    tmp = np.copy(X[..., wl])
                    X[..., wl] = tmp * self.calibration_functions[wavelength_override[wl], 0] + self.calibration_functions[wavelength_override[wl], 1]
        else:
            if self.calibration_functions.shape[-1] == 3:  # quadratic
                for wl in range(length):
                    tmp = np.copy(X[..., wl])
                    X[..., wl] = tmp * tmp * self.calibration_functions[wl, 0] + tmp * self.calibration_functions[wl, 1] + self.calibration_functions[wl, 2]
            elif self.calibration_functions.shape[-1] == 2:  # linear
                for wl in range(length):
                    tmp = np.copy(X[..., wl])
                    X[..., wl] = tmp * self.calibration_functions[wl, 0] + self.calibration_functions[wl, 1]
        return X


class ProcessingPipeline(BaseSpectralProcessor):
    def __init__(self, steps):
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        for stp in self.steps:
            if not hasattr(stp, "process"):
                raise TypeError("All processing pipeline steps need process method implemented")

    def process(self, X):
        for stp in self.steps:
            X = stp.process(X)
        return X


class ClassificationProcessor(BaseSpectralProcessor):

    def __init__(self, model=None):
        self.model = model
        
    def process(self, X, y=None):
        return self.model.process(X, y)


"""
-------------------------- Writers ---------------------------------
"""


class SpecimFullImageWriter(BaseSpectralWriter):
    """
    Image writer uses rgb or grayscale dataset and write the to given file (png or jpg) using PIL library.
    
    Pil save images of dtype uint8 only. If float is given, converts to uint8 by multiplying. For accurate results float needs to be in the range of 0-1.
    """

    def __init__(self, metadata, mode=None):
        super().__init__(metadata)
        self.verbose = False
        self.mode = mode  # override image converion method
        self.required_parameters = ["full_path"]

    def store(self):
        pass

    def write(self, X, parameters):
        assert all(k in parameters for k in self.required_parameters), f"Metadata key assertion error, not all required keys are set, required keys are: {self.required_parameters}"
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data has to be 3D of shape X, Y, 3 (RGB) or X, Y (grayscale)"
        if self.mode is None:
            mode = "RGB"
            if len(X.shape) == 2:  # assume grayscale image
                mode = "L"
                if X.dtype == np.dtype("float") or X.dtype == np.dtype("float32"):
                    mode = "F"
                    if X.dtype == np.dtype("float"):  # convert to float32
                        X = X.astype("float32")
        else:
            mode = self.mode
        im = Image.fromarray(X, mode=mode)
        if mode == "F" and parameters["full_path"].split(".")[-1] != "tiff":
            tmp_path = parameters["full_path"].split(".")[:-1]
            tmp_path.append("tiff")
            tmp_path = ".".join(tmp_path)
            if self.verbose:
                print("Warning grayscale float image will be writen as a tiff file: ", tmp_path)
            im.save(tmp_path)
        else:
            im.save(parameters["full_path"])


class SpecimFullTiffWriter(BaseSpectralWriter):
    """
    Raster (Geotiff) writer using rasterio.

    Writes the data using the same dtype of input data. Uses rasterio profile from metadata
    """

    def __init__(self, metadata):
        super().__init__(metadata)
        self.required_parameters = ["full_path"]

    def store(self):
        pass

    def write(self, X, parameters, mask=None):
        assert all(k in parameters for k in self.required_parameters), f"Parameter key assertion error, not all required keys are set, required keys are: {self.required_parameters}"
        assert len(X.shape) == 3, "Input data has to be 3D of shape: X, Y, bands"
        if X.dtype == np.dtype("float"):  # reduce the float accuracy for image saving
            X = X.astype("float32")
        meta = copy.deepcopy(self.metadata["rasterio_profile"])
        meta['count'] = X.shape[-1]
        # TODO: possible error may appear creating geotiffs from envi files
        meta['dtype'] = X.dtype
        meta['driver'] = "GTiff"
        meta["interleave"] = "band"
        with rasterio.open(parameters["full_path"], "w", **meta) as f:
            f.write(np.transpose(X, (2, 0, 1)))
            if mask is not None:
                f.write_mask(mask)


class SpecimBlockWriter(BaseSpectralWriter):
    """
    Writes blocks of data to raster file using rasterio window and state data from SpecimBlockLoader.

    Window writer does not work, using the store to write all of the data at once.
    """

    def __init__(self, metadata, parameters):
        super().__init__(metadata)
        self.required_parameters = ["full_path", "dtype", "count"]  # requires the full path of output raster and raster dtype that will be written, and number of bands to write
        assert all(k in parameters for k in self.required_parameters), f"Metadata key assertion error, not all required keys are set, required keys are: {self.required_parameters}"
        self.parameters = parameters
        self._init_store()

    def _init_store(self):
        self._store = np.zeros((self.metadata["image_dimensions"][1], self.metadata["image_dimensions"][0], self.parameters["count"]), dtype=self.parameters["dtype"])

    def store(self, X, state):
        assert len(X.shape) == 3, "Input data has to be 3D of shape: X, Y, bands"
        if X.dtype == np.dtype("float") and self.parameters["dtype"] != np.dtype("float"):  # reduce the float accuracy for image saving
            X = X.astype("float32")
        cw = state["current_width"]
        ch = state["current_height"]
        dw = state["width_step"]
        dh = state["height_step"]
        if dh == 0:
            sh = slice(None)
        else:
            sh = slice(ch, ch + dh)
        if dw == 0:
            sw = slice(None)
        else:
            sw = slice(cw, cw + dw)
        self._store[sh, sw, :] = X

    def write(self, X, state, write=False, processor=None):
        """
        Store data in memory or write the data to raster.

        Use write = True  to write stored data to raster
        """

        if write:
            if processor is not None:
                self._store = processor.process(self._store)  # Use processor before writing dataset. One processor or a pipeline of processors (WIP), eg. to scale whole dataset
                if self._store.dtype == np.dtype("float"):  # reduce the float accuracy for image saving
                    self._store = self._store.astype("float32")
            # convert (x, y, bands) to (bands, x, y) for writing raster
            self._store = np.transpose(self._store, (2, 0, 1))
            meta = copy.deepcopy(self.metadata["rasterio_profile"])
            # change raster metadata to that of store, if processor transformed the data
            meta['count'] = self._store.shape[0]
            meta['dtype'] = self._store.dtype
            # TODO: possible error may appear creating geotiffs from envi files
            meta['driver'] = "GTiff"
            meta["interleave"] = "band"
            with rasterio.open(self.parameters["full_path"], "w", **meta) as f:
                f.write(self._store)
            del self._store
            self._init_store()
        else:
            self.store(X, state)


class SpecimBlockImageWriter(BaseSpectralWriter):
    """
    Image writer uses rgb or grayscale dataset and write the to given file (png or jpg) using PIL library.
    
    Pil save images of dtype uint8 only. For accurate results float needs to be in the range of 0-1.

    Using the data store to collect blocks of data then write them to the image.
    """

    def __init__(self, metadata, parameters):
        super().__init__(metadata)
        self.verbose = False
        self.required_parameters = ["full_path", "dtype", "count"]
        assert all(k in parameters for k in self.required_parameters), f"Metadata key assertion error, not all required keys are set, required keys are: {self.required_parameters}"
        self.parameters = parameters
        if "mode" in self.parameters:
            self.mode = self.parameters["mode"]  # override image converion method
        else:
            self.mode = None
        self._init_store()
        if self.verbose:
            # parameter sanity check
            if self.parameters["count"] == 3 and (self.parameters["dtype"] == np.dtype("float") or self.parameters["dtype"] == np.dtype("float32")):
                print("Warning PIL cannot write 3 channel float type data, convert before writing")

    def _init_store(self):
        self._store = np.zeros((self.metadata["image_dimensions"][1], self.metadata["image_dimensions"][0], self.parameters["count"]), dtype=self.parameters["dtype"])

    def store(self, X, state):
        assert len(X.shape) == 3 or len(X.shape) == 2, "Input data has to be 3D of shape X, Y, 3 (RGB) or X, Y (grayscale)"
        if X.dtype == np.dtype("float") and self.parameters["dtype"] != np.dtype("float"):  # reduce the float accuracy for image saving
            X = X.astype("float32")
        cw = state["current_width"]
        ch = state["current_height"]
        dw = state["width_step"]
        dh = state["height_step"]
        if dh == 0:
            sh = slice(None)
        else:
            sh = slice(ch, ch + dh)
        if dw == 0:
            sw = slice(None)
        else:
            sw = slice(cw, cw + dw)
        self._store[sh, sw, :] = X

    def write(self, X, state, write=False, processor=None):
        if write:  # Final writing step. invoke once and only after pipeline completion.
            if processor is not None:
                self._store = processor.process(self._store)
            if self.mode is None:
                mode = "RGB"
                if len(self._store.shape) == 2:  # assume grayscale image
                    mode = "L"
                    if self._store.dtype == np.dtype("float") or self._store.dtype == np.dtype("float32"):
                        mode = "F"
                        if self._store.dtype == np.dtype("float"):  # convert to float32
                            self._store = self._store.astype("float32")
            else:
                mode = self.mode
            im = Image.fromarray(self._store, mode=mode)
            if mode == "F" and self.parameters["full_path"].split(".")[-1] != "tiff":
                tmp_path = self.parameters["full_path"].split(".")[:-1]
                tmp_path.append("tiff")
                tmp_path = ".".join(tmp_path)
                if self.verbose:
                    print("Warning grayscale float image will be writen as a tiff file: ", tmp_path)
                im.save(tmp_path)
            else:
                im.save(self.parameters["full_path"])
        else:
            self.store(X, state)


