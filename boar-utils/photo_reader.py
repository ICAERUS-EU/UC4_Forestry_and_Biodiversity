from ._readers import AbstractReader
import os
from PIL import Image
from numpy import asarray
from .utils import get_exif, photo_files, get_meta_by_model
import datetime


class PhotoReader(AbstractReader):

    def __init__(self, data_root, file_filter=None, **kwargs):
        self.default_filter = photo_files
        self.files = None
        if file_filter is not None:
            self.set_file_filter(file_filter)
        super().__init__(data_root, **kwargs)

    def set_file_filter(self, new_filter):
        """Set a custom photo file extention filter (or file name substring filters)

        Args:
            new_filter (list): list of file extention filters
        """
        if isinstance(new_filter, list):
            self.default_filter = new_filter
        else:
            print('Input filter type error, skipping operation')

    def parse_folder(self):
        """Reads all files (files only) from data_root folder, non recursive.
        """
        self.files = sorted([os.path.join(self.data_root, f) for f in os.listdir(
            self.data_root) if os.path.isfile(os.path.join(self.data_root, f))])
        self._filter_files()

    def _filter_files(self):
        """Filter gathered files from the folder, to remove any non photo file.
        Checks if any filter string is in filename
        """
        new_files = []
        for fl in self.files:
            check = False
            for ext in self.default_filter:
                if ext in fl:
                    check = True
                    break
            if check:
                new_files.append(fl)
        self.files = new_files

    def _get_camera_model(self, meta):
        """Get the used camera model from metadata

        Args:
            meta (dict): image exif metadata
        """
        if 'Model' in meta:
            self.camera_model = str(meta['Model']).replace('\x00', '')
        else:
            self.camera_model = None

    def _set_meta(self):
        """Super override, reads file types from folder, generates meta from camera model
        """
        if self.files is None:
            self.parse_folder()
        meta = get_exif(self.files[0])
        self._get_camera_model(meta)
        self.meta = get_meta_by_model(self.camera_model.replace("\x00", ""))

    def _set_positional(self):
        """Generate positional data from images and EXIF information
        """
        info = []
        for fl in self.files:
            meta = get_exif(fl)
            data = {'Path': fl,
                    'GPSLatitude': meta['GPSLatitude'],
                    'GPSLongitude': meta['GPSLongitude'],
                    'RelativeAltitude': meta['RelativeAltitude'],
                    'ImageWidth': meta['ImageWidth'],
                    'ImageHeight': meta['ImageHeight'],
                    'timestamp': float(datetime.datetime.strptime(meta['DateTime'], '%Y:%m:%d %H:%M:%S').strftime("%s"))}
            info.append(data)
        self.positional = info

    def _read(self, index):
        return asarray(Image.open(self.files[index]))
