from baselib.base import ReaderBase, LoaderBase, WriterBase, ProcessorMixin
from abc import ABC, abstractmethod


class BaseSpectralReader(ReaderBase):
    """
    Base spectral data reader class for different, hyperspectral and other types of spectra, readers

    Requires the wavelenght and metadata to be read.
    """

    def __init__(self):
        self._wavelength = None
        self._metadata = None

    def _load_metadata(self):
        raise NotImplementedError

    @property
    def metadata(self):
        """
        Base metadata property

        Return metadata about the spectral data read
        """
        return self._metadata

    def _load_wavelength(self):
        raise NotImplementedError

    @property
    def wavelength(self):
        """
        Base metadata property

        Return metadata about the spectral data read
        """
        return self._wavelength


class BaseSpectralLoader(LoaderBase):
    """
    Base spectral data loader.

    Implements the load function to read the data to memory using different methods.

    Gets metadata about the required files to be read from BaseSpectralReader classes.
    """

    def __init__(self):
        self._state = None
        self._data = None

    def _loader(self):
        """
        Data loader using the method implemented (eg. Full load at once, partial pixel load, partial band load, single file load) 

        Implement different methods depending on the need and the metadata provided from the BaseSpectralReader class and internal state

        return read data
        """
        raise NotImplementedError

    def load(self, metadata):
        """
        Loader function that read required files unsing the metadata provided

        Implements internal state to track the data read

        Can be implemented as a generator to return part of data and the current state 
        """
        yield self._loader(metadata)

    @property
    def data(self):
        return self._data

    @property
    def state(self):
        return self._state


class BaseSpectralProcessor(ProcessorMixin):
    """
    Base spectral data processing class

    Takes data from BaseSpectralLoader

    return processed data, agnostic to the BaseSpectralLoader state, uses only wavelength if required (eg. for rgb conversion).
    """

    def process(self, X, y=None, wavelength=None):
        raise NotImplementedError


class BaseSpectralWriter(WriterBase):
    """
    Base spectral data writer

    Takes data from BaseSpectralProcessor, status data from BaseSpectralLoader and metadata from BaseSpectralReader
    """

    def __init__(self, metatada):
        self.metadata = metatada
        self._store = None

    def store(self, X, status):
        """
        Temporary data storage function, if the data has to be written all at once (eg. Geotiff raster or png image)

        Stores data in memory according to the status and metadata provided.

        Is used by the write funtion to finalize writing
        """
        raise NotImplementedError

    def write(self, X, status, parameters):
        """
        Data writer function

        Implement data store check if data store is used

        Uses metadata from BaseSpectralReader to write files

        Additional parameters required (output path, data types or other)
        """
        raise NotImplementedError


class BaseIndexProcessor(ProcessorMixin):
    """
    Base spectral index calculation class

    Takes input data as separate bands required be the index.
    """

    def process(self, X, y, **kwargs):  # X- band1, y- band2, kwargs- other parameters
        raise NotImplementedError


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
