from sentinel_tools.domain import Band

import math

import pyproj
import geojson
import rasterio
import sentinelhub
import shapely
from shapely import Point, to_geojson, Polygon, ops
from typing import Tuple, Type, Union
from functools import partial
from dataclasses import dataclass
from pyproj import Transformer
from loguru import logger
import haversine

METERS_CRS = "EPSG:3346"
METERS_PROJ = pyproj.Proj(METERS_CRS)


@dataclass
class BoundingBox:
    min_point: Point # longitude, latitude
    max_point: Point
    crs: str

    @classmethod
    def from_coordinates(cls, min_x: float, min_y: float, max_x: float, max_y: float, crs: str):
        return BoundingBox(
            Point(min_x, min_y),
            Point(max_x, max_y),
            crs
        )

    @classmethod
    def from_sentinelhub_bbox(cls, bbox: sentinelhub.BBox) -> "BoundingBox":
        return BoundingBox(
            Point(bbox.lower_left),
            Point(bbox.upper_right),
            bbox.crs.ogc_string(),
        )

    def to_sentinelhub_bbox(self) -> sentinelhub.BBox:
        sentinelhub_crs_list = [crs.ogc_string() for crs in sentinelhub.CRS]
        org_bbox: BoundingBox = self

        if self.crs not in sentinelhub_crs_list:
            default_crs = sentinelhub.CRS.WGS84.ogc_string()
            logger.warning(
                f"{self.crs} is not supported in the sentinelhub.CRS. It will be converted to {default_crs}"
            )
            org_bbox = self.transform(default_crs)

        return sentinelhub.BBox(org_bbox.coordinates, sentinelhub.CRS(org_bbox.crs))

    def to_polygon(self) -> shapely.Polygon:
        min_x, min_y, max_x, max_y = self.coordinates
        return shapely.Polygon(
            [
                (min_x, min_y),
                (max_x, min_y),
                (max_x, max_y),
                (min_x, max_y),
                (min_x, min_y),
            ]
        )

    def transform(self, output_crs: str) -> "BoundingBox":
        """
        Transforms the BoundingBox to a different coordinate reference system (CRS).
        :param output_crs: CRS OGS string
        :return: BoundingBox transformed to the output_crs
        """
        transformer = pyproj.Transformer.from_crs(self.crs, output_crs, always_xy=True)
        return BoundingBox(
            transform_geometry(transformer, self.min_point),
            transform_geometry(transformer, self.max_point),
            output_crs
        )

    def transform_to_meters(self) -> "BoundingBox":
        return self.transform(METERS_CRS)

    @property
    def coordinates(self) -> Tuple[float, float, float, float]:
        return self.min_point.x, self.min_point.y, self.max_point.x, self.max_point.y

    def to_window(self, source: rasterio.io.DatasetReader) -> rasterio.windows.Window:
        if source.crs != self.crs:
            raise ValueError(f"Source CRS is different ({source.crs} vs {self.crs})")

        return rasterio.windows.from_bounds(*self.coordinates, transform=source.transform)

    def __str__(self):
        return f"({self.min_point.x}, {self.min_point.y}); ({self.max_point.x}, {self.max_point.y}) @ {self.crs}"


def __get_meters_transformers(source_crs: sentinelhub.CRS) -> Tuple[pyproj.Transformer, pyproj.Transformer]:
    """
    Create pyproj.Transformer objects for converting between the source CRS and METERS_CRS.
    :param source_crs: The source Coordinate Reference System (CRS)
    :return: A tuple of two pyproj.Transformer objects. The first transformer converts from source_crs to METERS_CRS,
             and the second transformer converts from METERS_CRS to source_crs.
    """
    org_to_meters = pyproj.Transformer.from_crs(source_crs.ogc_string(), METERS_CRS, always_xy=True)
    meters_to_org = pyproj.Transformer.from_crs(METERS_CRS, source_crs.pyproj_crs(), always_xy=True)
    # org_to_meters = pyproj.Transformer.from_proj(source_crs.projection(), METERS_PROJ)
    # meters_to_org = pyproj.Transformer.from_proj(METERS_PROJ, source_crs.projection())

    return org_to_meters, meters_to_org


def transform_geometry(transformer: pyproj.Transformer, geometry: Union[Point, Polygon]) -> Union[Point, Polygon]:
    result = ops.transform(transformer.transform, geometry)
    return type(geometry)(result)


def point_to_buffer(point: Point, crs: sentinelhub.CRS, radius: float) -> Polygon:
    transformer = pyproj.Transformer.from_crs(crs.ogc_string(), METERS_CRS, always_xy=True)
    point_meters = transform_geometry(transformer, point)
    buffer_meters: Polygon = point_meters.buffer(radius)

    return transform_geometry(
        pyproj.Transformer.from_crs(METERS_CRS, crs.ogc_string(), always_xy=True),
        buffer_meters
    )


def distance_in_meters(point_1: Point, point_2: Point, crs: sentinelhub.CRS):
    transformer = pyproj.Transformer.from_crs(crs.ogc_string(), METERS_CRS, always_xy=True)

    point_1_meters = transform_geometry(transformer, point_1)
    point_2_meters = transform_geometry(transformer, point_2)

    return point_1_meters.distance(point_2_meters)


def distance_in_meters_haversine(point_1, point_2, crs):
    pt1 = point_1.x, point_1.y
    pt2 = point_2.x, point_2.y

    return haversine.haversine(pt1, pt2, unit=haversine.Unit.METERS)


def flip_coords(geometry: Union[Point, Polygon]) -> Union[Point, Polygon]:
    def flip(x, y):
        return y, x

    return ops.transform(flip, geometry)


def main():
    center_point = Point(55.76808157739135, 25.50075645527998)
    buffer = point_to_buffer(center_point, CRS.WGS84, 100)

    print(polygon_to_bbox(buffer))

    for point_coords in buffer.exterior.coords:
        point = Point(*point_coords)

        print(
            point,
            distance_in_meters(center_point, point, CRS.WGS84),
            distance_in_meters_1(center_point, point, CRS.WGS84),
            distance_in_meters_2(center_point, point, CRS.WGS84),
            distance_in_meters_3(center_point, point, CRS.WGS84),
        )

    feature = geojson.Feature(geometry=buffer)

    feature_collection = geojson.FeatureCollection([feature])

    with open("out.geojson", "w") as f:
        geojson.dump(feature_collection, f)


if __name__ == "__main__":
    main()
