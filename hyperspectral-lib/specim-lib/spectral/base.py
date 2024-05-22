from abc import ABC, abstractmethod


class BaseSpectralReader(ABC):
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
    
    @abstractmethod
    def _load_wavelength(self):
        pass

    @property
    @abstractmethod
    def wavelength(self):
        """
        Base metadata property

        Return metadata about the spectral data read
        """
        return self._wavelength


class BaseSpectralLoader(ABC):
    """
    Base spectral data loader.

    Implements the load function to read the data to memory using different methods.

    Gets metadata about the required files to be read from BaseSpectralReader classes.
    """

    def __init__(self):
        self._state = None
        self._data = None
    
    @abstractmethod
    def _loader(self):
        """
        Data loader using the method implemented (eg. Full load at once, partial pixel load, partial band load, single file load) 

        Implement different methods depending on the need and the metadata provided from the BaseSpectralReader class and internal state

        return read data
        """
        pass

    @abstractmethod
    def load(self, metadata):
        """
        Loader function that read required files unsing the metadata provided

        Implements internal state to track the data read

        Can be implemented as a generator to return part of data and the current state 
        """
        yield self._loader(metadata)

    @property
    @abstractmethod
    def data(self):
        return self._data
    
    @property
    @abstractmethod
    def state(self):
        return self._state


class BaseSpectralProcessor(ABC):
    """
    Base spectral data processing class
    
    Takes data from BaseSpectralLoader

    return processed data, agnostic to the BaseSpectralLoader state, uses only wavelength if required (eg. for rgb conversion).
    """
    
    @abstractmethod
    def process(self, X, y=None, wavelength=None):
        pass


class BaseSpectralWriter(ABC):
    """
    Base spectral data writer

    Takes data from BaseSpectralProcessor, status data from BaseSpectralLoader and metadata from BaseSpectralReader
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

        Uses metadata from BaseSpectralReader to write files
        
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


class BaseSpectralModel(ABC):
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

