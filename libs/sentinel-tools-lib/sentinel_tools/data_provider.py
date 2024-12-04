import sentinelhub

from sentinel_tools.gis_utils import BoundingBox
from sentinel_tools.image_generator import get_bbox, get_bbox_with_surroundings
from sentinel_tools.domain import Band
from typing import Optional, List

from datetime import datetime
import numpy as np
import pandas as pd
import rasterio
import re

import os


class MissingTile(Exception):
    pass


class MissingBoundingBox(Exception):
    pass


class DataProvider:
    def __init__(self, data_dir: str):
        if not data_dir.endswith("/"):
            data_dir += "/"

        if not os.path.exists(data_dir):
            raise ValueError(f"Path {data_dir} doesn't exists")

        self.data_dir = data_dir
        self.data_df: pd.DataFrame = self.construct_files_data()
        self.tiles_df: pd.DataFrame = self.construct_tiles_data()

    @staticmethod
    def list_files_in_directory(dir_to_list: str):
        file_list = []
        for root, _, files in os.walk(dir_to_list):
            for file in files:
                # directory = root.removeprefix(dir_to_list)
                file_list.append(os.path.join(root, file))

        return file_list

    @staticmethod
    def get_tile_metadata(file_path: str):
        with rasterio.open(file_path) as src:
            bbox = BoundingBox.from_coordinates(*src.bounds, crs=str(src.crs))

        return bbox.crs, *bbox.coordinates

    @staticmethod
    def get_default_crs() -> str:
        return sentinelhub.CRS.WGS84.ogc_string()

    def construct_files_data(self):
        parsed_data = []
        file_paths_list = self.list_files_in_directory(self.data_dir)

        for file_path in file_paths_list:
            local_file_path = file_path.removeprefix(self.data_dir)
            match = re.match(r"(\w+),(\d{4}-\d{2}-\d{2}),\d+/R(\d+)m/(\w+)\.jp2", local_file_path)

            if not match:
                raise ValueError(f"Invalid data file format {file_path}")

            with rasterio.open(file_path) as src:
                file_crs = str(src.crs)

            tile_id, date_str, resolution, band = match.groups()
            parsed_data.append({
                "tile_id": tile_id,
                "resolution": int(resolution),
                "band": band,
                "date": date_str,
                "crs": file_crs,
                "file_path": file_path,
            })

        if len(parsed_data) == 0:
            raise ValueError(f"No files were parsed in the {self.data_dir}")

        df = pd.DataFrame(parsed_data)
        df["date"] = pd.to_datetime(df["date"])

        return df

    def construct_tiles_data(self):
        tiles_df = self.data_df.drop_duplicates(subset=["tile_id"])
        tiles_df = tiles_df.assign(
            crs="",
            bbox_min_longitude=np.nan,
            bbox_min_latitude=np.nan,
            bbox_max_longitude=np.nan,
            bbox_max_latitude=np.nan,
        )

        metadata_cols = ["crs", "bbox_min_longitude", "bbox_min_latitude", "bbox_max_longitude", "bbox_max_latitude"]
        tiles_df[metadata_cols] = tiles_df.apply(
            lambda x: self.get_tile_metadata(x["file_path"]), axis=1, result_type="expand"
        )

        return tiles_df.reset_index().drop(columns=["file_path", "band", "date"])

    def get_tile_id(self, bbox: BoundingBox) -> Optional[str]:
        unique_crs_list = self.tiles_df.crs.unique()

        for crs in unique_crs_list:
            current_bbox = bbox.transform(crs)
            result = self.tiles_df[
                (self.tiles_df["crs"] == crs) &
                (self.tiles_df["bbox_min_longitude"] <= current_bbox.min_point.x) &
                (self.tiles_df["bbox_min_latitude"] <= current_bbox.min_point.y) &
                (self.tiles_df["bbox_max_longitude"] > current_bbox.max_point.x) &
                (self.tiles_df["bbox_max_latitude"] > current_bbox.max_point.y)
                ]

            if result.empty:
                continue

            row = result.iloc[0]
            return row["tile_id"]

        return None

    def filter_data_df(
            self,
            band: Optional[str | Band] = None,
            bands: Optional[List[str | Band]] = None,
            date: Optional[str | datetime] = None,
            bbox: Optional[BoundingBox] = None,
            tile_id: Optional[str] = None,
            resolution: Optional[int] = None,
            min_resolution: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Filter the self.data_df based on the provided parameters and return the corresponding results.
        :param band: A single band
        :param bands: List of bands. This parameter will be ignored if band is not None
        :param date: datetime or string in the %Y-%m-%d format
        :param bbox: bounding box object. This parameter will be ignored if tile_id is provided.
        :param tile_id:
        :param resolution:
        :param min_resolution: Whether to return the highest resolution possible. Ignored if the resolution
                               parameter is given.
        :return: filtered dataframe
        """
        result = self.data_df.copy()

        if band:
            if isinstance(band, Band):
                band = band.to_str()

            result = result[result["band"] == band]

        if bands and not band:
            bands = [str(b) for b in bands]
            result = result[result["band"].isin(bands)]

        if date:
            if isinstance(date, str):
                date = pd.to_datetime(date)

            result = result[result["date"] == date]

        if bbox and not tile_id:
            tile_id = self.get_tile_id(bbox)

        if tile_id:
            result = result[result["tile_id"] == tile_id]

        if min_resolution and not resolution:
            cols = ["tile_id", "date", "band"]
            temp_df = result[cols + ["resolution"]]
            min_indices = temp_df.groupby(by=cols).idxmin()["resolution"]
            result = result.loc[min_indices]

        if resolution:
            result = result[result["resolution"] == resolution]

        return result

    def get_bbox_file(
            self,
            band: str,
            date: datetime,
            bbox: BoundingBox,
            resolution: Optional[str] = None,
            surrounding_len: Optional[int] = None
    ) -> str:
        tile_id = self.get_tile_id(bbox)
        if tile_id:
            raise MissingTile(f"Tile was not found for bbox {bbox}")

        result = self.data_df[
            (self.data_df["tile_id"] == tile_id) &
            (self.data_df["date"] == date) &
            (self.data_df["band"] == band)
            ]

        if resolution:
            result = result[result["resolution"] == resolution]

        if result.empty:
            raise MissingBoundingBox(
                f"Requested file was not found. Received params: {band} / {date} / {bbox} / {resolution} / {tile_id}"
            )

        row = result.iloc[0]
        return row["file_path"]

    def __parse_tile_id(self, bbox: Optional[BoundingBox] = None, tile_id: Optional[str] = None) -> str:
        if bbox is None and tile_id is None:
            raise ValueError("Either bbox or tile_id must not be None.")

        tile_id = tile_id or self.get_tile_id(bbox)
        if tile_id is None:
            raise MissingTile(f"Tile was not found for bbox {bbox}")

        return tile_id

    def get_max_resolution(
            self,
            bands: List[str | Band],
            date_str: str,
            bbox: Optional[BoundingBox] = None,
            tile_id: Optional[str] = None,
    ) -> int:
        tile_id = self.__parse_tile_id(bbox, tile_id)
        filtered_df = self.filter_data_df(
            date=date_str,
            tile_id=tile_id,
            bands=bands
        )

        if filtered_df.empty:
            raise ValueError("Filtered Dataframe was empty")

        return filtered_df["resolution"].min().astype(int)

    def load_bbox_band_old(
            self,
            band: str,
            date_str: str,
            tile_id: str,
            bbox: BoundingBox,
            org_res_meters: Optional[int] = None,
            out_res_meters: Optional[int] = None,
            surrounding_len: Optional[int] = None,
            suppress_errors: Optional[bool] = None
    ) -> Optional[np.ndarray]:
        result = self.data_df[
            (self.data_df["tile_id"] == tile_id) &
            (self.data_df["date"] == pd.to_datetime(date_str)) &
            (self.data_df["band"] == band)
            ]

        if org_res_meters is None:
            org_res_meters = result["resolution"].min()

        result = result[result["resolution"] == org_res_meters]

        if result.empty:
            if suppress_errors:
                return None

            raise MissingBoundingBox(
                f"Requested file was not found. It was requested with the following params:\n"
                f"Band: {band}\n"
                f"Date: {date_str}\n"
                f"Bounding Box: {bbox} ({tile_id})\n"
                f"Resolution(m): {org_res_meters}"
            )

        row = result.iloc[0]
        bbox = bbox.transform(row["crs"])

        with rasterio.open(row["file_path"]) as src:
            if surrounding_len:
                bbox_data = get_bbox_with_surroundings(src, bbox, surrounding_len, out_res_meters)

            else:
                bbox_data = get_bbox(src, bbox, out_res_meters)

        return bbox_data

    @staticmethod
    def load_bbox_band(
            file_path: str,
            bbox: BoundingBox,
            out_res_meters: Optional[int] = None,
            surrounding_len: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        with rasterio.open(file_path) as src:
            bbox = bbox.transform(src.crs)

            if surrounding_len:
                bbox_data = get_bbox_with_surroundings(src, bbox, surrounding_len, out_res_meters)

            else:
                bbox_data = get_bbox(src, bbox, out_res_meters)

        return bbox_data

    def load_multiple_bbox_bands(
            self,
            bands: List[str | Band],
            date_str: str,
            bbox: BoundingBox,
            tile_id: Optional[str] = None,
            upscale_lower: Optional[bool] = None,
            surrounding_len: Optional[int] = None,
            suppress_errors: Optional[bool] = None
    ):
        tile_id = self.__parse_tile_id(bbox, tile_id)
        filtered_df = self.filter_data_df(
            bands=bands,
            date=date_str,
            tile_id=tile_id,
            min_resolution=True
        )

        if filtered_df.empty:
            raise MissingBoundingBox(
                f"No files were found. They were requested with the following params:\n"
                f"Bands: {bands}\n"
                f"Date: {date_str}\n"
                f"Bounding Box: {bbox} ({tile_id})\n"
            )

        upscale_res = None
        if upscale_lower:
            upscale_res = filtered_df["resolution"].min()

        results = {}
        for band in bands:
            row = filtered_df[filtered_df["band"] == str(band)]

            if len(row) > 1:
                raise ValueError("More than one row satisfies this condition.")

            if len(row) == 0:
                if suppress_errors:
                    results[band] = None
                    continue

                raise ValueError(f"Band file doesn't exist")

            row = row.iloc[0]
            file_path = row["file_path"]

            data = self.load_bbox_band(file_path, bbox, out_res_meters=upscale_res, surrounding_len=surrounding_len)
            if isinstance(band, str):
                band = Band.from_str(band)

            results[band] = data

        return results
