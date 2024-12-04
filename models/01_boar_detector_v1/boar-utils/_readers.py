from collections.abc import Iterator
from typing import Any
from xxlimited import new
from boars.io.utils import turning_points, interpolate_points
import numpy as np


class AbstractReader(Iterator):
    """Abstract class for readers. Concrete reader methods have to implement methods for _read(i), _set_positional and _set_metadata. 

    Positional data should be read to an array, that is bijective to image data. Image data should be lazily loaded and accessed only via iterator, 
    positional data (self.positional) and metadata(self.meta) should be always available.
    """

    def __init__(self, data_root: str, **reader_params: Any):
        """
        Args:
            data_root (str): data directory.
            reader_params (dict): extra parameters that may be passed to concrete readers
        """
        self.data_root = data_root
        self.reader_params = reader_params
        self.meta = dict()
        self.positional = list()
        self._set_meta()
        self._set_positional()
        self._interpolate_position()

    def _interpolate_position(self):
        """
        Interpolates positional data after its set
        """
        lat = []
        long = []
        for pos in self.positional:
            lat.append(pos['GPSLatitude'])
            long.append(pos['GPSLongitude'])

        points = np.array([[lt, lng] for lt, lng in zip(lat, long)])

        tp = turning_points(points, return_all=False)
        indexes_tp = np.where(tp != 0)[0]
        pairwise_indexes = np.zeros((2, len(indexes_tp)+1), dtype=int)
        pairwise_indexes[0, 0] = 0
        pairwise_indexes[1, -1] = len(points)-1
        pairwise_indexes[0, 1:] = indexes_tp
        pairwise_indexes[1, :-1] = indexes_tp
        pairwise_indexes = pairwise_indexes.T

        new_points = interpolate_points(
            points, pairwise_indexes, drop_neighbours=True)

        for newp, pos in zip(new_points, self.positional):
            pos['GPSLatitude'] = newp[0]
            pos['GPSLongitude'] = newp[1]

    def _set_positional(self):
        """Sets self.positional

        Raises:
            NotImplemented: _description_
        """
        raise NotImplementedError

    def _set_meta(self):
        """Sets self.meta.
        Required meta data: FocalLenght (mm), FocalPlaneXResolution(mm), FocalPlaneYResolution(mm), ImageWidth, ImageHeight

        Raises:
            NotImplemented: _description_
        """
        raise NotImplementedError

    def _read(self, index):
        """Reads reader data by index

        Args:
            index (int): index to read data from

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.positional)

    def __iter__(self):
        self.idx = 0
        self.len = self.__len__()
        return self

    def __next__(self):
        i = self.idx
        self.idx += 1
        if self.idx > self.len:
            raise StopIteration
        return self._read(i)
