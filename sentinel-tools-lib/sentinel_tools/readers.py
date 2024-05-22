from .base import BaseSentinelLoader, BaseSentinelReader, Tile
from sentinelhub.geometry import BBox
from datetime import datetime
from sentinelhub.api.opensearch import get_area_info
from sentinelhub.aws.data import AwsData


ALL_BANDS = ['R10m/B08', 'R10m/B04', 'R10m/B02', 'R20m/B05', 'R10m/B03', 'R20m/B06', 'R20m/B07', 'R20m/B8A', 'R20m/B11', 'R20m/B12', 'R20m/B01', 'R60m/B09', 'R20m/SCL']
ALL_BANDS_SORTED = ['R60m/B01', 'R10m/B02', 'R10m/B03', 'R10m/B04', 'R20m/B05', 'R20m/B06', 'R20m/B07', 'R10m/B08', 'R20m/B8A', 'R60m/B09', 'R20m/B11', 'R20m/B12', 'R20m/SCL']
RGB_BANDS = ["R10m/B04", "R10m/B03", "R10m/B02"]
INDEX_BANDS = {'SCL': ['R20m/SCL'], 'MSAVI': ['R10m/B04', 'R20m/B8A'], 'LAI': ['R10m/B04', 'R10m/B08'], 'LAI3': ['R10m/B03', 'R10m/B02'],
                'NDVI': ['R10m/B04', 'R10m/B08'], 'EVI': ['R10m/B02', 'R10m/B04', 'R10m/B08'], 'MCARI':['R10m/B03', 'R10m/B04', 'R20m/B05'],
                'SOC': ['R20m/B05', 'R20m/B06', 'R20m/B11', 'R20m/B12', 'R10m/B02', 'R10m/B03', 'R10m/B04', 'R10m/B08'],
                'LAI2': ['R10m/B02', 'R10m/B03', 'R10m/B04', 'R20m/B05', 'R20m/B06', 'R20m/B11', 'R20m/B12','R20m/B07', 'R20m/B8A']}
CLOUD_BANDS = ["R60m/SCL"]



class CSVReader(BaseSentinelReader):
    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path, 'r') as f:
            return f.read()


class TileReader(BaseSentinelReader):
    """
    Class for reading metadata of tiles for a given bounding box.

    bbox: BBox
        Bounding box of the area for which the tiles are to be seached.
    date_from: datetime
        Start date of the time interval for which the tiles are to be searched.
    date_to: datetime
        End date of the time interval for which the tiles are to be searched.
    maxcc: float
        Maximum cloud coverage of the tiles to be searched.
    """
    def __init__(self, bbox: BBox, date_from: datetime, date_to: datetime, maxcc: float):
        self.bbox = bbox
        self.date_from = date_from
        self.date_to = date_to
        self.maxcc = maxcc
        self._metadata = None

    def _load_metadata(self):
        """
        Gathers metadata for the collected tiles givent the parameters.

        _metadata: list
            List of Tile objects.
        """
        self._metadata = []
        results = get_area_info(self.bbox, (self.date_from, self.date_to), maxcc=self.maxcc)
        for res in results:
            data = AwsData.url_to_tile(res['properties']['s3Path'])
            self._metadata.append(Tile(data[0], data[1], data[2], res['properties']['cloudCover']))

    @property
    def metadata(self):
        if self._metadata is None:
            self._load_metadata()
        return self._metadata


class TileDownloader(BaseSentinelLoader):
    """
    Downloads tiles from AWS. Updates tile metadata with out_dir and bands.

    Parameters
    ----------
    out_dir : str
        Output directory.
    bands : list
        List of bands to download.
    lazy_load : bool
        If True, only prepares Tile for download and doesnt download data until Tile.data() is called.
    verbose : bool
        If True, prints status messages.
    """
    def __init__(self, out_dir: str, bands: list, lazy_load=True, verbose=False):
        self.verbose = verbose
        self.outdir = out_dir
        self.bands = bands
        self.lazy_load = lazy_load

    def _loader(self):
        for i in range(len(self._data)):
            self._data[i].data_dir = self.outdir
            self._data[i].bands = self.bands
            if not self.lazy_load:
                self._data[i].download()

    def load(self, metadata):
        self._data = metadata
        self._loader()

    @property
    def data(self):  # returns metadata
        return self._data
    

