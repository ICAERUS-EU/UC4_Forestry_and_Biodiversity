import math

from sentinel_tools.gis_utils import distance_in_meters, point_to_buffer, distance_in_meters_haversine, flip_coords
from sentinelhub import CRS
from shapely import Point

VILNIUS_POINT = Point(25.2798, 54.6892)
KLAIPEDA_POINT = Point(21.1175, 55.7172)


def test_if_distance_between_two_points_is_correct():
    expected_distance_km = 289
    actual_distance_km = int(round(distance_in_meters(VILNIUS_POINT, KLAIPEDA_POINT, CRS.WGS84) / 1000))
    assert expected_distance_km == actual_distance_km


def test_if_distance_between_two_points_matches_using_different_methods():
    distance_km_org = int(round(distance_in_meters(VILNIUS_POINT, KLAIPEDA_POINT, CRS.WGS84) / 1000))

    vilnius_flipped_point = flip_coords(VILNIUS_POINT)
    klaipeda_flipped_point = flip_coords(KLAIPEDA_POINT)
    distance_km_haversine = int(
        round(
            distance_in_meters_haversine(vilnius_flipped_point, klaipeda_flipped_point, CRS.WGS84) / 1000
        )
    )

    assert math.isclose(distance_km_org, distance_km_haversine, rel_tol=1)


def test_if_all_points_around_the_polygon_are_equal_to_radius():
    center_point = Point(55.76808157739135, 25.50075645527998)
    crs = CRS.WGS84
    radius_meters = 100.0

    buffer = point_to_buffer(center_point, crs, radius_meters)
    buffer_points = [Point(*point_coords) for point_coords in buffer.exterior.coords]

    assert all(math.isclose(distance_in_meters(center_point, point, crs), radius_meters) for point in buffer_points)
