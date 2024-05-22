import numpy as np

from sentinel_tools.gis_utils import BoundingBox
from sentinel_tools.data_provider import DataProvider
from sentinel_tools.domain import Band

import rasterio
from dataclasses import dataclass
from datetime import datetime
from sentinelhub import CRS
from paths import BASE_DIR
from typing import Dict, List, Optional

DATE_FORMAT = "%Y-%m-%d"
BASE_PROFILE = {
    "driver": "GTiff",
    "nodata": None,
    "blockxsize": 1024,
    "blockysize": 1024,
    "tiled": True,
    "interleave": "pixel"
}


@dataclass
class DataEntry:
    date: datetime
    bbox: BoundingBox
    data_dict: Dict[Band, np.ndarray]
    data_type: np.dtype
    resolution_meters: int

    @staticmethod
    def parse_date(date: datetime | str) -> datetime:
        if isinstance(date, str):
            date = datetime.strptime(date, DATE_FORMAT)

        return date

    @staticmethod
    def parse_band(band: Band | str) -> Band:
        if isinstance(band, str):
            band = Band[band]

        return band

    @classmethod
    def read_from_file_old(cls, date: datetime | str, file_path: str) -> "DataEntry":
        date = cls.parse_date(date)

        with rasterio.open(file_path) as src:
            data_dict = src.read()
            bbox = BoundingBox.from_coordinates(*src.bounds, crs=src.crs)
            resolution, *_ = src.res

        return cls(date, bbox, data_dict, resolution)

    @classmethod
    def read_from_file(cls, date: datetime | str, file_path: str) -> "DataEntry":
        date = cls.parse_date(date)
        data_dict = {}

        with rasterio.open(file_path) as src:
            bbox = BoundingBox.from_coordinates(*src.bounds, crs=src.crs)
            resolution, *_ = src.res
            dtype = src.profile["dtype"]

            for bidx, description in enumerate(src.descriptions, start=1):
                band = Band.from_str(description)
                data_dict[band] = src.read(bidx)

        return cls(date, bbox, data_dict, dtype, resolution)

    @classmethod
    def generate_entry(
            cls,
            data_provider: DataProvider,
            bbox: BoundingBox,
            date: datetime | str,
            bands: Optional[List[Band]] = None
    ) -> "DataEntry":
        if bands is None:
            bands = Band.all_possible()

        date_str = cls.parse_date(date).strftime(DATE_FORMAT)
        tile_id = data_provider.get_tile_id(bbox)
        resolution_meters = data_provider.get_max_resolution(bands, date_str, tile_id=tile_id)

        data_dict = data_provider.load_multiple_bbox_bands(
            bands=bands,
            date_str=date_str,
            bbox=bbox,
            tile_id=tile_id,
            upscale_lower=True,
            suppress_errors=True
        )

        result_dtype = max(
            [data.dtype for data in data_dict.values() if data is not None], key=lambda x: np.iinfo(x).bits
        )
        for band, data in data_dict.items():
            if data is None:
                continue

            data_dict[band] = data.astype(result_dtype)

        return cls(date, bbox, data_dict, result_dtype, resolution_meters)

    def get_available_bands(self) -> List[Band]:
        return list(self.data_dict.keys())

    def get_band_data(self, band: Band | str):
        band = self.parse_band(band)
        return self.data_dict[band]

    def get_data_shape(self):
        data, *_ = self.data_dict.values()

        width, height = data.shape
        count = len(self.data_dict)

        return count, width, height

    def write_to_file(self, output_path: str):
        bbox_meters = self.bbox.transform_to_meters()

        count, width, height = self.get_data_shape()
        bbox_min_x, *_, bbox_max_y = bbox_meters.coordinates

        profile = {
            "dtype": self.data_type,
            "width": width,
            "height": height,
            "count": count,
            "crs": bbox_meters.crs,
            "transform": rasterio.transform.Affine(
                self.resolution_meters,
                0.0,
                bbox_min_x,
                0.0,
                -self.resolution_meters,
                bbox_max_y
            )
        }
        profile.update(BASE_PROFILE)

        with rasterio.open(output_path, "w", **profile) as out:
            for bidx, (band, data) in enumerate(self.data_dict.items(), start=1):
                out.write_band(bidx, data)
                out.set_band_description(bidx, band.to_str())


def main():
    provider = DataProvider(str(BASE_DIR / "downloads"))
    print(provider.get_max_resolution(
        Band.all_possible(),
        "2023-03-17",
        tile_id="35VMC"
    ))

    bbox = BoundingBox.from_coordinates(
        25.6436664842, 56.10128429925, 25.64494838877, 56.102000743504, CRS.WGS84.ogc_string()
    )
    bbox = BoundingBox.from_coordinates(
        25.402200,
        55.846949,
        27.159994,
        56.843709,
        "EPSG:4326"
    )

    # date_str = "2023-03-17"
    date_str = "2023-04-24"
    file_path = str(BASE_DIR / f"results/output_bbox_{date_str}_RGB.tiff")

    rgb_bands = [
        Band.B04,
        Band.B03,
        Band.B02
    ]

    data_entry = DataEntry.generate_entry(provider, bbox, date_str, rgb_bands)
    data_entry.write_to_file(file_path)
    print(data_entry.get_available_bands())

    data_entry = DataEntry.read_from_file(date_str, file_path)
    band = Band.B02
    # print(data_entry.data_dict)
    print(data_entry.get_available_bands())


    print(data_entry.data_dict[band].shape)
    print(data_entry.data_dict[band].dtype)
    print(data_entry.data_dict[band])
    # print(data_entry.get_band_data(Band.SCL).shape)


if __name__ == "__main__":
    main()
