import esa_snappy
from esa_snappy import ProductIO
from esa_snappy import HashMap
import os, gc   
from esa_snappy import GPF
from .sentinel import SentinelData
from .base import Geometry
import jpy
from baselib.base import ProcessorMixin


# snappy install :
# https://step.esa.int/main/download/snap-download/
# install using .sh file

# esa_snappy install:
# https://senbox.atlassian.net/wiki/spaces/SNAP/pages/3114106881/Installation+and+configuration+of+the+SNAP-Python+esa_snappy+interface+SNAP+version+12


# based on
# https://github.com/wajuqi/Sentinel-1-preprocessing-using-Snappy/blob/master/s1_preprocessing.py


class SnappyProcessor(ProcessorMixin):
    def process(self, X, y=None):  # SentinelData and Geometry
        self.data = X
        self.geometry = y
        self.data.processed_files = []
        safe_path = self.data.out_folder + "manifest.safe"
        terrain = self.data.files[0].split(".")[0] + "-processed.tif"
        if os.path.exists(terrain):
            print(f"processed file {terrain} exists, skipping processing pipeline")
            for i, fl in enumerate(self.data.files):  # tests/S1C_IW_GRDH_1SDV_20250509T044227_20250509T044252_002249_004C70_BF3A/measurement/iw-vv.tiff
                terrain = fl.split(".")[0] + "-processed"
                self.data.processed_files.append(terrain + ".tif")
            return True
        gc.enable()
        gc.collect()
        self.sentinel_1 = ProductIO.readProduct(safe_path)
        self._pipeline(self.sentinel_1)
        del self.sentinel_1

    def _pipeline(self, product):
        # applyt orbit file
        t0 = self._apply_orbit_file(product)

        # thermal noise removal
        t1 = self._thermal_noise_removal(t0)

        # specle filter
        # t1 = self._speckle_filtering(t1)

        # for each band
        for i, fl in enumerate(self.data.files):  # tests/S1C_IW_GRDH_1SDV_20250509T044227_20250509T044252_002249_004C70_BF3A/measurement/iw-vv.tiff
            polarization = str.upper(fl.split("/")[-1].split(".")[0].split("-")[-1])
            # calibration
            try:
                t2 = self._calibration(t1, polarization)
                sigma_bands = True
            except RuntimeError as e:
                print(e)
                t2 = t1
                sigma_bands = False

            # terrain correction
            try:
                t2 = self._terrain(t2, polarization, fl, sigma_bands)
            except RuntimeError as e:
                print(e)
                t2 = t2

            # subset 
            if self.geometry is not None:
                t2 = self._subset(t2)
            else:
                t2 = t2

            terrain = fl.split(".")[0] + "-processed"
            ProductIO.writeProduct(t2, terrain, 'GeoTIFF')
            self.data.processed_files.append(terrain + ".tif")

    def _calibration(self, product, polarization):
        parameters = HashMap()
        parameters.put('outputSigmaBand', True)
        parameters.put('sourceBands', 'Intensity_' + polarization)
        parameters.put('selectedPolarisations', polarization)
        parameters.put('outputImageScaleInDb', False)

        output = GPF.createProduct("Calibration", parameters, product)
        return output

    def _subset(self, product):
        parameters = HashMap()
        parameters.put('geoRegion', self.geometry.geometry.wkt)
        parameters.put('outputImageScaleInDb', False)

        output = GPF.createProduct("Subset", parameters, product)
        return output

    def _terrain(self, product, polarization, fn, sigma_bands):
        parameters = HashMap()
        parameters.put('demResamplingMethod', 'NEAREST_NEIGHBOUR')
        parameters.put('imgResamplingMethod', 'NEAREST_NEIGHBOUR')
        parameters.put('demName', 'SRTM 3Sec')
        parameters.put('pixelSpacingInMeter', 10.0)
        if sigma_bands:
            parameters.put('sourceBands', 'Sigma0_' + polarization)
        else:
            parameters.put('sourceBands', 'Intensity_' + polarization)

        output = GPF.createProduct("Terrain-Correction", parameters, product)
        return output

    def _apply_orbit_file(self, product):
        parameters = HashMap()
        parameters.put('Apply-Orbit-File', True)
        output = GPF.createProduct('Apply-Orbit-File', parameters, product)
        return output

    def _thermal_noise_removal(self, product):
        parameters = HashMap()
        parameters.put('removeThermalNoise', True)
        output = GPF.createProduct('ThermalNoiseRemoval', parameters, product)
        return output

    def _speckle_filtering(self, source):
        parameters = HashMap()
        Integer = jpy.get_type('java.lang.Integer')
        parameters.put('filter', 'Lee')
        parameters.put('filterSizeX', Integer(5))
        parameters.put('filterSizeY', Integer(5))
        output = GPF.createProduct('Speckle-Filter', parameters, source)
        return output
