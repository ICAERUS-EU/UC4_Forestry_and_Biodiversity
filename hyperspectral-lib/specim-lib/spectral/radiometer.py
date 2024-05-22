import os
import numpy as np


class Radiometer:
    def __init__(self, path, spectral_file_extensions=["txt"]):
        """
        Parameters
            
            path : str
                absolute path to file or folder of radiometer files
        """
        self.path = path
        self.extensions = spectral_file_extensions
        self.delimiter_string = ">>>>>Begin Spectral Data<<<<<"
        self.wavelenght_column = 0
        self.files = []
        self._data = None
        self._parse_path()
        self._read_data()

    def _check_extension(self, file_path):
        filename, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.split(".")[-1].lower()
        if file_extension in self.extensions:
            return True
        else:
            return False

    def __len__(self):
        return len(self.files)

    def _parse_path(self):
        assert type(self.path) == str, "Given path is not a string, please provide a path string"
        self.isdir = os.path.isdir(self.path)
        if self.isdir:
            all_files = [os.path.join(self.path, f) for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f))]
            for fl in all_files:
                if self._check_extension(fl):
                    self.files.append(fl)
        else:
            if self._check_extension(self.path):
                self.files.append(self.path)

    def _read_data(self):
        for fl in self.files:
            if self._data is None:
                self._data = self.read_file(fl, self.delimiter_string)[np.newaxis, :]
            else:
                self._data = np.append(self._data, self.read_file(fl, self.delimiter_string)[np.newaxis, :], axis=0)

    @property
    def data(self):
        if self._data is None:
            self._read_data()
        return self._data
        
    @staticmethod
    def read_file(path, delimiter_string):
        with open(path, "r") as fr:
            data = fr.readlines()
        values = []
        delimiter = False
        for ln in data:
            if delimiter:  # start reading after the delimiter line was read
                values.append([float(x) for x in ln.rstrip().split("\t")])
            else:
                if delimiter_string in ln:
                    delimiter = True
        return np.array(values)

