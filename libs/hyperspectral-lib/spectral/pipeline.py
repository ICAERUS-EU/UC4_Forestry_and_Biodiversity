import os
from . import specim
from . import base
import numpy as np
from . import index


class DefaultRGB(base.BasePipeline):
    """
    Create RGB from nonrectified tiff file (*-radiance.dat). 

    Input:
        in_path: str - full path to input file to be used
        out_path: str - full path to output file to be created
    """
    def __init__(self, in_path, out_path, overwrite=False) -> None:
        self.in_path = in_path
        self.out_path = out_path
        self.overwrite = overwrite
        # describe pipeline parts
        self.reader = specim.SpecimCubeReader(self.in_path)
        self.loader = specim.SpecimFullLoader(self.reader.metadata)
        self.writer = specim.SpecimFullImageWriter(self.reader.metadata)
        self.rgb = specim.RGBProcessor(self.reader.wavelength)
        self.scaler = specim.MinMaxScaler()
        self.converter = specim.FloatUint8Converter()

    def run(self):
        data = self.loader.data
        data = self.rgb.process(data)
        data = self.scaler.process(data)
        data = self.converter.process(data)
        self.writer.write(data, {"full_path": self.out_path})
        return True


class DefaultRectRGB(base.BasePipeline):
    """
    Create RGB from georectified tiff file (*-rect.dat). 

    Input:
        in_path: str - full path to input file to be used
        out_path: str - full path to output file to be created
    """
    def __init__(self, in_path, out_path, overwrite=False) -> None:
        self.in_path = in_path
        self.out_path = out_path
        self.overwrite = overwrite
        # describe pipeline parts
        self.reader = specim.SpecimCubeReader(self.in_path)
        self.loader = specim.SpecimFullLoader(self.reader.metadata)
        self.writer = specim.SpecimFullTiffWriter(self.reader.metadata)
        self.rgb = specim.RGBProcessor(self.reader.wavelength)
        self.scaler = specim.MinMaxScaler()
        self.converter = specim.FloatUint8Converter()
        self.masker = specim.MaskProcessor()

    def run(self):
        data = self.loader.data
        data = self.rgb.process(data)
        mask = self.masker.process(data)
        data = self.scaler.process(data)
        data = self.converter.process(data)
        self.writer.write(data, {"full_path": self.out_path}, mask=mask)
        return True


class DefaultNDVI(base.BasePipeline):
    """
    Create RGB from nonrectified tiff file (*-radiance.dat). 

    Input:
        in_path: str - full path to input file to be used
        out_path: str - full path to output file to be created
    """
    def __init__(self, in_path, out_path, overwrite=False) -> None:
        self.in_path = in_path
        self.out_path = out_path
        self.overwrite = overwrite
        # describe pipeline parts
        self.reader = specim.SpecimCubeReader(self.in_path)
        self.loader = specim.SpecimFullLoader(self.reader.metadata)
        self.writer = specim.SpecimFullImageWriter(self.reader.metadata)
        self.ndvi = specim.NDVIProcessor()
        self.scaler = specim.MinMaxScaler()
        self.converter = specim.FloatUint8Converter()
        self.extractor = specim.SpecimBandExtractor(self.reader.metadata, wavelength_numbers=[670, 800])

    def run(self):
        data = self.loader.data
        bands = self.extractor.process(data)
        bands = self.scaler.process(bands)
        data = self.ndvi.process(bands[..., 0], bands[..., 1], 0)
        data = np.nan_to_num(data)
        data = self.scaler.process(data)
        data = self.converter.process(data)
        self.writer.write(np.repeat(data, 3, axis=-1), {"full_path": self.out_path})
        return True


class DefaultRectNDVI(base.BasePipeline):
    """
    Create RGB from nonrectified tiff file (*-rect.dat). 

    Input:
        in_path: str - full path to input file to be used
        out_path: str - full path to output file to be created
    """
    def __init__(self, in_path, out_path, overwrite=False) -> None:
        self.in_path = in_path
        self.out_path = out_path
        self.overwrite = overwrite
        # describe pipeline parts
        self.reader = specim.SpecimCubeReader(self.in_path)
        self.loader = specim.SpecimFullLoader(self.reader.metadata)
        self.writer = specim.SpecimFullTiffWriter(self.reader.metadata)
        self.scaler = specim.MinMaxScaler()
        self.extractor = specim.SpecimBandExtractor(self.reader.metadata, wavelength_numbers=[670, 800])
        self.ndvi = specim.NDVIProcessor()
        self.masker = specim.MaskProcessor()

    def run(self):
        data = self.loader.data
        mask = self.masker.process(data)
        bands = self.extractor.process(data)
        bands = self.scaler.process(bands)
        data = self.ndvi.process(bands[..., 0], bands[..., 1], 0)
        self.writer.write(data, {"full_path": self.out_path}, mask=mask)
        return True


class DefaultNDVIRGB(base.BasePipeline):
    """
    Create RGB from nonrectified tiff file (*-radiance.dat). 

    Input:
        in_path: str - full path to input file to be used
        out_path: str - full path to output file to be created
    """
    def __init__(self, in_path, out_path, overwrite=False) -> None:
        self.in_path = in_path
        self.out_path = out_path
        self.overwrite = overwrite
        # describe pipeline parts
        self.reader = specim.SpecimCubeReader(self.in_path)
        self.loader = specim.SpecimFullLoader(self.reader.metadata)
        self.writer = specim.SpecimFullImageWriter(self.reader.metadata)
        self.ndvi = specim.NDVIProcessor()
        self.scaler = specim.MinMaxScaler()
        self.converter = specim.FloatUint8Converter()
        self.extractor = specim.SpecimBandExtractor(self.reader.metadata, wavelength_numbers=[670, 800])
        self.painter = index.Painter(index.color_pallets["ndvi"][0], index.color_pallets["ndvi"][1], index.color_pallets["ndvi"][2])

    def run(self):
        data = self.loader.data
        bands = self.extractor.process(data)
        bands = self.scaler.process(bands)
        data = self.ndvi.process(bands[..., 0], bands[..., 1], 0)
        data = np.nan_to_num(data)
        data = self.scaler.process(data)
        data = self.painter.process(data)
        data = self.converter.process(data)
        self.writer.write(data, {"full_path": self.out_path})
        return True


class DefaultRectNDVIRGB(base.BasePipeline):
    """
    Create RGB from nonrectified tiff file (*-rect.dat). 

    Input:
        in_path: str - full path to input file to be used
        out_path: str - full path to output file to be created
    """
    def __init__(self, in_path, out_path, overwrite=False) -> None:
        self.in_path = in_path
        self.out_path = out_path
        self.overwrite = overwrite
        # describe pipeline parts
        self.reader = specim.SpecimCubeReader(self.in_path)
        self.loader = specim.SpecimFullLoader(self.reader.metadata)
        self.writer = specim.SpecimFullTiffWriter(self.reader.metadata)
        self.scaler = specim.MinMaxScaler()
        self.extractor = specim.SpecimBandExtractor(self.reader.metadata, wavelength_numbers=[670, 800])
        self.ndvi = specim.NDVIProcessor()
        self.converter = specim.FloatUint8Converter()
        self.painter = index.Painter(index.color_pallets["ndvi"][0], index.color_pallets["ndvi"][1], index.color_pallets["ndvi"][2])
        self.masker = specim.MaskProcessor()

    def run(self):
        data = self.loader.data
        mask = self.masker.process(data)
        bands = self.extractor.process(data)
        bands = self.scaler.process(bands)
        data = self.ndvi.process(bands[..., 0], bands[..., 1], 0)
        data = self.scaler.process(data)
        data = self.painter.process(data)
        data = self.converter.process(data)
        self.writer.write(data, {"full_path": self.out_path}, mask=mask)
        return True


