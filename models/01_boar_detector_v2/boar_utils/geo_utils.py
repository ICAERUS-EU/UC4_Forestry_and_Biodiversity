# src/geo_utils.py
from __future__ import annotations
import csv
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from PIL import Image, ExifTags
import bisect
import math

def conf_to_style(conf: Optional[float]) -> str:
    """
    Convert a detection confidence score (0-1) into a marker color for folium.
    Higher confidence -> greener markers; missing values fall back to blue.
    """
    if conf is None:
        return "#3388ff"  # folium's default
    try:
        val = float(conf)
    except (TypeError, ValueError):
        val = 0.0
    val = max(0.0, min(val, 1.0))
    if val >= 0.8:
        return "#2ecc71"  # green
    if val >= 0.5:
        return "#f1c40f"  # yellow
    return "#e74c3c"      # red

def _convert_to_degrees(value):
    d = float(value[0][0]) / float(value[0][1])
    m = float(value[1][0]) / float(value[1][1])
    s = float(value[2][0]) / float(value[2][1])
    return d + (m / 60.0) + (s / 3600.0)

def read_exif_gps(image_path: str) -> Optional[Tuple[float, float]]:
    try:
        img = Image.open(image_path)
        exif = img._getexif() or {}
        gps_tag = None
        for k, v in ExifTags.TAGS.items():
            if v == 'GPSInfo':
                gps_tag = k
                break
        if gps_tag not in exif:
            return None
        gps_info = {}
        for t in exif[gps_tag]:
            sub_tag = ExifTags.GPSTAGS.get(t, t)
            gps_info[sub_tag] = exif[gps_tag][t]
        lat = _convert_to_degrees(gps_info['GPSLatitude'])
        if gps_info.get('GPSLatitudeRef') == 'S':
            lat = -lat
        lon = _convert_to_degrees(gps_info['GPSLongitude'])
        if gps_info.get('GPSLongitudeRef') == 'W':
            lon = -lon
        return (lat, lon)
    except Exception:
        return None

class TelemetryIndex:
    def __init__(self, csv_path: str):
        self.rows: List[Tuple[float, float, float, float]] = []  # (ts, lat, lon, alt)
        with open(csv_path, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                ts = float(row['timestamp'])
                lat = float(row['lat'])
                lon = float(row['lon'])
                alt = float(row.get('alt', 0.0))
                self.rows.append((ts, lat, lon, alt))
        self.rows.sort(key=lambda x: x[0])
        self.ts = [x[0] for x in self.rows]

    def query(self, ts: float) -> Optional[Tuple[float, float, float]]:
        if not self.rows:
            return None
        i = bisect.bisect_left(self.ts, ts)
        if i == 0:
            _, lat, lon, alt = self.rows[0]
            return lat, lon, alt
        if i == len(self.rows):
            _, lat, lon, alt = self.rows[-1]
            return lat, lon, alt
        before = self.rows[i-1]
        after = self.rows[i]
        # choose closer
        if abs(before[0] - ts) <= abs(after[0] - ts):
            _, lat, lon, alt = before
        else:
            _, lat, lon, alt = after
        return lat, lon, alt


def _deg2rad(val: float) -> float:
    return val * math.pi / 180.0


def intrinsics_from_fov(
    width: int,
    height: int,
    hfov_deg: Optional[float] = None,
    vfov_deg: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    if hfov_deg is None and vfov_deg is None:
        raise ValueError("Provide hfov_deg or vfov_deg")
    if hfov_deg is not None:
        fx = (width / 2.0) / math.tan(_deg2rad(hfov_deg) / 2.0)
        fy = fx * (height / width)
    else:
        fy = (height / 2.0) / math.tan(_deg2rad(vfov_deg or 0.0) / 2.0)
        fx = fy * (width / height)
    cx, cy = width / 2.0, height / 2.0
    return fx, fy, cx, cy


def _matmul(A: Tuple[Tuple[float, float, float], ...], B: Tuple[Tuple[float, float, float], ...]) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(
        tuple(sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def rotation_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> Tuple[Tuple[float, float, float], ...]:
    """
    Construct a camera-to-world rotation matrix using a DJI-friendly convention:
    - Start with camera forward along +Z, right +X, down +Y.
    - Apply yaw around world Z, pitch around intermediate Y, roll around intermediate X.
    - Apply a 180° flip around X to align camera optical axis downward when pitch ~ -90°.
    """
    yaw = _deg2rad(yaw_deg)
    pitch = _deg2rad(pitch_deg + 90.0)
    roll = _deg2rad(roll_deg + 180.0)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    Rx = (
        (1.0, 0.0, 0.0),
        (0.0, cr, -sr),
        (0.0, sr, cr),
    )
    Ry = (
        (cp, 0.0, sp),
        (0.0, 1.0, 0.0),
        (-sp, 0.0, cp),
    )
    Rz = (
        (cy, -sy, 0.0),
        (sy, cy, 0.0),
        (0.0, 0.0, 1.0),
    )

    return _matmul(Rx, _matmul(Ry, Rz))


def ray_from_pixel(u: float, v: float, fx: float, fy: float, cx: float, cy: float) -> Tuple[float, float, float]:
    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        return (0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm)


def matvec(R: Tuple[Tuple[float, float, float], ...], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


def enu_to_lla(lat0_deg: float, lon0_deg: float, alt0: float, dx: float, dy: float, dz: float) -> Tuple[float, float, float]:
    earth_radius = 6_378_137.0
    dLat = (dy / earth_radius) * (180.0 / math.pi)
    dLon = (dx / (earth_radius * math.cos(_deg2rad(lat0_deg)))) * (180.0 / math.pi)
    return lat0_deg + dLat, lon0_deg + dLon, alt0 + dz


def project_to_ground(
    u: float,
    v: float,
    img_w: int,
    img_h: int,
    lat: float,
    lon: float,
    alt_m: float,
    yaw: float,
    pitch: float,
    roll: float,
    hfov: Optional[float] = None,
    vfov: Optional[float] = None,
    ground_alt_m: float = 0.0,
) -> Optional[Tuple[float, float]]:
    try:
        fx, fy, cx, cy = intrinsics_from_fov(img_w, img_h, hfov_deg=hfov, vfov_deg=vfov)
    except Exception:
        return None

    ray_cam = ray_from_pixel(u, v, fx, fy, cx, cy)
    R = rotation_from_ypr(yaw, pitch, roll)
    ray_world = matvec(R, ray_cam)

    if abs(ray_world[2]) < 1e-6:
        return None

    t = (ground_alt_m - alt_m) / ray_world[2]
    if t <= 0.0:
        return None

    dx = t * ray_world[0]
    dy = t * ray_world[1]
    dz = t * ray_world[2]

    lat_g, lon_g, _ = enu_to_lla(lat, lon, alt_m, dx, dy, dz)
    return lat_g, lon_g

# src/geo_utils.py
def _to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def _to_int(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

def normalize_detection(det: dict, class_names: dict | list | None = None) -> dict:
    """
    Ensure detection dict has numeric types and safe defaults.
    Expected input keys (any may be missing): x1,y1,x2,y2 or cx,cy,w,h; conf; cls; id/track_id; ts/utc.
    Returns a new dict with normalized fields.
    """
    d = dict(det)  # shallow copy

    # Confidence
    conf = _to_float(d.get("conf"), 0.0)
    if conf < 0.0 or conf > 1.0:
        # Some detectors output 0–100; rescale if it looks like a percentage.
        conf = conf / 100.0 if conf > 1.5 else max(0.0, min(conf, 1.0))
    d["conf"] = conf

    # Class id + name
    cls_id = d.get("cls", d.get("class"))
    cls_id = _to_int(cls_id, 0)
    d["cls"] = cls_id
    if class_names is not None:
        if isinstance(class_names, dict):
            d["name"] = class_names.get(cls_id, str(cls_id))
        elif isinstance(class_names, (list, tuple)):
            d["name"] = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)

    # Box coords (if present) – coerce to float
    for k in ("x1", "y1", "x2", "y2", "cx", "cy", "w", "h"):
        if k in d:
            d[k] = _to_float(d[k], 0.0)

    # Track id (optional)
    if "track_id" in d or "id" in d:
        d["track_id"] = _to_int(d.get("track_id", d.get("id")), None)
    if "cluster_id" in d:
        d["cluster_id"] = _to_int(d.get("cluster_id"), None)

    # Geo (optional)
    if "lat" in d: d["lat"] = _to_float(d["lat"], None)
    if "lon" in d: d["lon"] = _to_float(d["lon"], None)
    if "alt" in d: d["alt"] = _to_float(d["alt"], None)
    if "lat_drone" in d: d["lat_drone"] = _to_float(d["lat_drone"], None)
    if "lon_drone" in d: d["lon_drone"] = _to_float(d["lon_drone"], None)
    for k in ("rel_alt", "abs_alt", "focal_len", "dzoom_ratio", "gb_yaw", "gb_pitch", "gb_roll"):
        if k in d:
            d[k] = _to_float(d[k], None)

    # Timestamp passthrough (string is ok), but normalize empty to None
    ts = d.get("utc") or d.get("ts") or d.get("timestamp")
    d["utc"] = ts if (ts is not None and f"{ts}".strip()) else None

    return d
