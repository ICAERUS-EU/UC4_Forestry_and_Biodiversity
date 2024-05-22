import rasterio
import numpy as np
import os
from spectral.utils import spectral_calculator
import time
from spectral.rgb import RGB
from PIL import Image
import copy
from rasterio.mask import mask
from shapely import geometry
from shapely.ops import transform
import pyproj
import requests
import utils
import json


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


class Cube:
    """
    A class to work with hyperspectral cube data created by specim camera

    """

    def __init__(self, cube_path, wavelengths=None, lazy_load=True, cube_extension=".dat", verbose=False):
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
        self.wavelengths = wavelengths
        self.cube_extension = cube_extension
        self.lazy_load = lazy_load
        self.verbose = verbose
        self._data = None
        self._rgb = None
        self._wavelength_load()
        self._load_metadata()
        if self.verbose:
            print(f"Cube {self.cube_path} read, metadata: \n {self.metadata}")

    def _wavelength_load(self):
        """
        Loads the wavelengths for this data cube, uses default if none given
        """
        if self.wavelengths is None:
            self.wavelengths = SPECIM_WAVELENGTHS_2X1

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
            self.metadata = fl.profile
        if not self.lazy_load:
            self._load_data()

    def _load_data(self):
        """
        Loads cube data into memory. Transposes to shape x, y, wl
        """
        if self.verbose:
            start = time.time()
            print("Loading cube data")
        if self._data is None:
            with rasterio.open(self.cube_path, "r") as fl:
                self._data = fl.read()
            self._data = np.transpose(self._data, (1, 2, 0))
        if self.verbose:
            print("Cube data loaded in ", time.time() - start)

    @property
    def data(self):
        if self._data is None:
            self._load_data()
        return self._data

    @property
    def rgb(self):
        if self._data is None:
            self._load_data()
        if self._rgb is None:
            if self.verbose:
                start = time.time()
                print("Generating RGB image from HSI")
            self._rgb = RGB(self._data, self.wavelengths)
            if self.verbose:
                print("RGB generation completed in ", time.time() - start)
        return self._rgb

    @property
    def shape(self):
        if self._data is None:
            self._load_data()
        return self._data.shape

    def save_rgb(self, out_path=None, overwrite=False):
        crs = self.metadata['crs']
        if crs is not None:  # georeference exist
            out_filename = out_path if out_path is not None else ".".join(self.cube_path.split(".")[:-1]) + "-rgb.tiff"
            if os.path.exists(out_filename) and not overwrite:
                print(f"File {out_filename} already exists skipping generation")
                return True
            meta = copy.deepcopy(self.metadata)
            meta['count'] = 3
            meta['dtype'] = 'float32'
            if self.rgb.shape[0] == 3:
                with rasterio.open(out_filename, "w", **meta) as f:
                    f.write(self.rgb)
            else:
                # convert to gtiff for better compatibility.
                meta['driver'] = 'GTiff'
                del meta['interleave']
                with rasterio.open(out_filename, "w", **meta) as f:
                    f.write(np.transpose(self.rgb, (2, 0, 1)))
        else:
            out_filename = out_path if out_path is not None else ".".join(self.cube_path.split(".")[:-1]) + "-rgb.png"
            if os.path.exists(out_filename) and not overwrite:
                print(f"File {out_filename} already exists skipping generation")
                return True
            im = Image.fromarray((self.rgb * 255).astype(np.uint8))
            im.save(out_filename)
        print("saved image to path: ", out_filename)

    def save_index(self, data, index_name="ndvi", out_path=None):
        if out_path is None:
            if len(data.shape) == 2:  # grayscale image
                name_extension = f"-{index_name}" + "-gray"
            else:
                name_extension = f"-{index_name}" + "-rgb"
        crs = self.metadata['crs']
        if crs is not None:  # georeference exist
            if len(data.shape) == 2:  # grayscale image
                data = data[np.newaxis, ...]
            else:  # rgb image
                if data.shape[-1] == 3:
                    data = np.transpose(data, (2, 0, 1))
            out_filename = out_path if out_path is not None else ".".join(self.cube_path.split(".")[:-1]) + name_extension + ".tiff"
            meta = self.metadata
            meta['count'] = data.shape[0]
            meta['dtype'] = 'float32'
            with rasterio.open(out_filename, "w", **meta) as f:
                f.write(data)
        else:
            out_filename = out_path if out_path is not None else ".".join(self.cube_path.split(".")[:-1]) + name_extension + ".png"
            im = Image.fromarray((data * 255).astype(np.uint8))
            im.save(out_filename)
        print("saved image to path: ", out_filename)

    def reflectance(self, calibration_spectra, reflection_percentages, wl_axis=-1, method="average", calibration_curve_degree=1):
        """
        Calculate cube reflectance from given DRS (Diffuse Reflectance Standard) spectra.
        As a base this was used: https://d-nb.info/1216632197/34
        Main formula R = (raw - dark) / (white - dark)

        Parameters
            calibration_spectra : np.array shape -> (N, WL), (N, M, WL)
                input of calibration spectra in singular array
            reflectance_percentages : np.array shape -> (N), (N, WL)
                the reflectance percentager for each of calibration spectra, maybe different for each wavelength
            wl_axis : int
                which calibration spectra axis is the wavelenght axis
            method : str
                which method to use for calibraion spectra aggregation if required.
            calibration_curve_degree : int
                degree of polynomial to fit to the calibration curve. Use 1 for black and white reflectance only, with more calibration curves higher degree may be more accurate.
        """
        shp1 = calibration_spectra.shape
        shp2 = reflection_percentages.shape

        if calibration_curve_degree > 2:
            print("Higher degree polynomials are WIP")
            return False

        assert len(shp1) == 2 or len(shp1) == 3, "calibration_spectra shape is out range"
        assert len(shp2) == 1 or len(shp2) == 2, "reflection_percentages shape is out range"
        
        if len(shp2) == 2:
            assert len(shp2[0]) == len(self.wavelengths) or len(shp2[1]) == len(self.wavelengths), "mismatch between wavelenght counts in cube and relection_precentages data"

        assert type(wl_axis) == int, "wl_axis is not an integer"

        assert shp1[wl_axis] == len(self.wavelengths), "mismatch between wavelength of calibration spectra and data cube"
        # dark check
        # assert 0 in reflection_percentages, "Dark calibration spectra is required (0% reflectance)"

        # get reflectance calibration functions for each wavelength
        calib = []
        for wl in range(shp1[wl_axis]):
            cal_spectra = np.take(calibration_spectra, wl, axis=wl_axis)
            if len(cal_spectra.shape) > 1:
                cal_spectra = spectral_calculator(cal_spectra, wl_axis - 1 if wl_axis > 0 else wl_axis, method)
            if len(cal_spectra.shape) == 2:
                z = np.polyfit(cal_spectra[:, wl], reflection_percentages[:, wl], calibration_curve_degree)
            else:
                z = np.polyfit(cal_spectra, reflection_percentages, calibration_curve_degree)
            calib.append(z.copy())
        calib = np.array(calib)
        self.calibration_functions = calib
        # self.cal_degree = calibration_curve_degree

        # calibrate data
        self._data = self._data.astype("float")
        if calib.shape[-1] == 3:  # quadratic
            for wl in range(shp1[wl_axis]):
                tmp = self._data[..., wl]
                self._data[..., wl] = tmp * tmp * calib[wl, 0] + tmp * calib[wl, 1] + calib[wl, 2]
        elif calib.shape[-1] == 2:  # linear
            for wl in range(shp1[wl_axis]):
                tmp = self._data[..., wl]
                self._data[..., wl] = tmp * calib[wl, 0] + calib[wl, 1]


# WIP perdaryti pagal naujus calibrations is DataStore appso.
class Calibration:
    def __init__(self, cube, cal_cube_api_id, calibration_cube, dark_path=None) -> None:
        self.cube = cube
        self._data = None
        self.calibration_cube = calibration_cube
        self.cube_id = cal_cube_api_id
        self.server_ip = "https://hyperlabeling.art21.lt/"  # port of hyperlabeling server
        self.endpoint = "api/dataset"
        self.api_token = "e1cae176910d896162f17ffee397ed11b39bbdbe"
        self.dark_path = dark_path
        self._calibrations()
        self._load_data()
        self._calibrate()

    def _get_api(self, get_endpoint, params=None):
        post_url = self.server_ip + get_endpoint 
        headers = {'Authorization': 'Token ' + self.api_token}
        if params is None:
            r = requests.get(post_url, headers=headers, verify=False)
        else:
            r = requests.get(post_url, headers=headers, verify=False, params=params)
        return r

    def _get_dataset(self):
        response = self._get_api(self.endpoint + "/" + str(self.cube_id))
        # convert response to json two times XD
        response = json.loads(response.json())
        geoms = []
        # filter main cube geometry, get only samples and calibrations
        for ft in response["features"]:
            if "type" in ft["geometry"]["properties"]:
                if ft["geometry"]["properties"]["type"] == "sample" or ft["geometry"]["properties"]["type"] == "calibration":
                    geoms.append(ft["geometry"])
        return geoms 

    def _cut_tiff_with_poly(self, poly):
        """
        Cuts the tiff using provided polygon.
        
        Params:
            tiff: str
                path to .tiff file

            poly: geometry
                sample geojson
        """
        geom = geometry.Polygon(poly['coordinates'][0])

        # get tiff metadata
        with rasterio.open(self.calibration_cube) as src:
            crs = src.crs

        # transform polygons to raster crs
        project = pyproj.Transformer.from_crs(pyproj.CRS('EPSG:4326'), pyproj.CRS(crs), always_xy=True).transform
        geom = transform(project, geom).simplify(0.1)

        with rasterio.open(self.calibration_cube) as src:
            out_image, out_transform = mask(src, [geom], crop=True)
            out_meta = src.meta
        
        out_image = np.transpose(out_image, (1, 2, 0))
        return out_image

    @classmethod
    def gather_dark(cls, dark_raster):
        if type(dark_raster) == str:  # skip loading if not a path given
            with rasterio.open(dark_raster, "r") as fl:
                data = fl.read()
            data = np.average(data, axis=(1, 2))[np.newaxis, :]
        return data

    def _calibrations(self):
        poly = self._get_dataset()
        calibrations = []
        for pol in poly:
            if pol["properties"]["type"] != "sample":
                calibrations.append((pol['properties']['value'], self._cut_tiff_with_poly(pol)))
        self.cal_data = []
        self.cal_val = []
        for cal_file in calibrations:
            self.cal_data.append(np.average(cal_file[1], axis=(0, 1)))
            self.cal_val.append(float(cal_file[0]))
        self.cal_data = np.array(self.cal_data)
        self.cal_val = np.array(self.cal_val)
    
    def _load_data(self):
        """
        Loads cube data into memory. Transposes to shape x, y, wl
        """
        if self._data is None:
            with rasterio.open(self.cube, "r") as fl:
                self._data = fl.read()
                self.meta = fl.profile
            self._data = np.transpose(self._data, (1, 2, 0))
            self.data_mask = np.where(np.sum(self._data, axis=-1) == 0, 0, 1)

    @property
    def data(self):
        if self._data is None:
            self._load_data()
            # self_data = torch.from_numpy(self._data)
        return self._data

    def _calibrate(self):
        self._data = utils.calibrate(self._data, self.cal_data, self.cal_val, self.gather_dark(self.dark_path))

