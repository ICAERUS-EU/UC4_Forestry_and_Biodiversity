# src/map_writer.py
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import folium

try:
    from .geo_utils import conf_to_style, normalize_detection
except ImportError:
    from geo_utils import conf_to_style, normalize_detection  # type: ignore[no-redef]

try:
    import mapbox_vector_tile  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency
    mapbox_vector_tile = None  # type: ignore[assignment]


def _moving_average(points: List[Tuple[float, float]], k: int = 5) -> List[Tuple[float, float]]:
    if len(points) <= 2 or k <= 1:
        return points
    smoothed: List[Tuple[float, float]] = []
    half = k // 2
    for idx in range(len(points)):
        start = max(0, idx - half)
        end = min(len(points), idx + half + 1)
        window = points[start:end]
        count = len(window)
        if count == 0:
            smoothed.append(points[idx])
            continue
        lat_avg = sum(p[0] for p in window) / count
        lon_avg = sum(p[1] for p in window) / count
        smoothed.append((lat_avg, lon_avg))
    return smoothed


def _rdp(points: List[Tuple[float, float]], eps: float = 1e-5) -> List[Tuple[float, float]]:
    if len(points) < 3 or eps <= 0.0:
        return points

    def _perp_distance(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
        (x1, y1), (x2, y2) = a, b
        x0, y0 = p
        if (x1, y1) == (x2, y2):
            return math.hypot(x0 - x1, y0 - y1)
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + (x2 * y1) - (y2 * x1))
        denominator = math.hypot(y2 - y1, x2 - x1)
        return numerator / denominator if denominator else 0.0

    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        dist = _perp_distance(points[0], points[-1], points[i])
        if dist > dmax:
            index = i
            dmax = dist

    if dmax > eps:
        left = _rdp(points[: index + 1], eps)
        right = _rdp(points[index:], eps)
        return left[:-1] + right
    return [points[0], points[-1]]


def _detections_to_features(
    detections: Iterable[Dict[str, Any]],
    class_names: Optional[Union[Dict[int, str], List[str]]] = None,
) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for det in detections:
        norm = normalize_detection(det, class_names=class_names)
        lat = norm.get("lat")
        lon = norm.get("lon")
        if lat is None or lon is None:
            continue
        geom = {"type": "Point", "coordinates": [lon, lat]}
        props = {k: v for k, v in norm.items() if k not in ("lat", "lon")}
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    return features


def _extract_point_props(props: Dict[str, Any], lat: float, lon: float) -> Dict[str, Any]:
    cls_name = (
        props.get("name")
        or props.get("label")
        or props.get("class")
        or str(props.get("cls", ""))
    )
    point = {
        "lat": lat,
        "lon": lon,
        "class": cls_name,
        "conf": props.get("conf", 0.0),
        "image": props.get("image") or props.get("frame") or "",
        "entity_id": props.get("entity_id") or props.get("track_id"),
        "track_id": props.get("track_id"),
        "track_id_original": props.get("track_id_original"),
        "frame": props.get("frame"),
        "utc": props.get("utc"),
        "cluster_id": props.get("cluster_id"),
        "grounded": props.get("grounded"),
        "lat_drone": props.get("lat_drone"),
        "lon_drone": props.get("lon_drone"),
    }
    return point


def _frame_key(value: Any, fallback: int) -> Tuple[int, int]:
    if value is None:
        return (fallback, fallback)
    try:
        iv = int(value)
        return (iv, fallback)
    except Exception:
        return (fallback, fallback)


def _features_to_points(features: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pts: List[Dict[str, Any]] = []
    cluster_latest: Dict[Any, Dict[str, Any]] = {}
    track_latest: Dict[Any, Dict[str, Any]] = {}
    singles: List[Dict[str, Any]] = []

    for fallback_idx, feat in enumerate(features):
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        coords = geom.get("coordinates") or [None, None]
        lon, lat = coords
        if lat is None or lon is None:
            continue
        lat_f = float(lat)
        lon_f = float(lon)

        point = _extract_point_props(props, lat_f, lon_f)
        order = _frame_key(point.get("frame"), fallback_idx)
        cluster_id = point.get("cluster_id")
        track_id = point["track_id"]
        if cluster_id is not None:
            existing = cluster_latest.get(cluster_id)
            if not existing or order > existing["_order"]:
                point["_order"] = order
                cluster_latest[cluster_id] = point
            continue
        if track_id is not None:
            existing = track_latest.get(track_id)
            if not existing or order > existing["_order"]:
                point["_order"] = order
                track_latest[track_id] = point
            continue

        singles.append(point)

    for lookup in (cluster_latest, track_latest):
        for point in lookup.values():
            point.pop("_order", None)
            pts.append(point)

    pts.extend(singles)

    return pts


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> Tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    xtile = min(max(xtile, 0), n - 1)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    ytile = min(max(ytile, 0), n - 1)
    return xtile, ytile


def _tile_bounds(x: int, y: int, zoom: int) -> Tuple[float, float, float, float]:
    n = 2 ** zoom
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return (lon_min, lat_min, lon_max, lat_max)


def _features_bounds(features: Iterable[Dict[str, Any]]) -> Optional[Tuple[float, float, float, float]]:
    lon_min = math.inf
    lat_min = math.inf
    lon_max = -math.inf
    lat_max = -math.inf
    found = False
    for feat in features:
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if not coords:
            continue
        lon, lat = coords
        lon_min = min(lon_min, lon)
        lon_max = max(lon_max, lon)
        lat_min = min(lat_min, lat)
        lat_max = max(lat_max, lat)
        found = True
    if not found:
        return None
    return (lon_min, lat_min, lon_max, lat_max)


def _build_track_paths(features: Iterable[Dict[str, Any]]) -> Dict[Any, List[Tuple[float, float]]]:
    tracks: Dict[Any, List[Tuple[Tuple[int, int], float, float]]] = defaultdict(list)

    for fallback_idx, feat in enumerate(features):
        props = feat.get("properties") or {}
        track_id = props.get("track_id")
        if track_id is None:
            continue
        coords = feat.get("geometry", {}).get("coordinates")
        if not coords:
            continue
        lon, lat = coords
        if lat is None or lon is None:
            continue
        order = _frame_key(props.get("frame"), fallback_idx)
        tracks[track_id].append((order, float(lat), float(lon)))

    ordered: Dict[Any, List[Tuple[float, float]]] = {}
    for track_id, entries in tracks.items():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda x: x[0])
        ordered[track_id] = [(lat, lon) for _, lat, lon in entries]
    return ordered

def write_map(
    points: List[Dict],
    out_html: Path,
    drone_path: Optional[List[Tuple[float, float]]] = None,
    track_paths: Optional[Dict[Any, List[Tuple[float, float]]]] = None,
) -> Optional[Path]:
    """
    points: list of dicts with lat, lon, class, conf, image.
    """
    if not points:
        return None
    lat0, lon0 = points[0]["lat"], points[0]["lon"]
    m = folium.Map(location=[lat0, lon0], zoom_start=16, control_scale=True)

    if drone_path:
        smoothed_path = _moving_average(drone_path, k=5)
        smoothed_path = _rdp(smoothed_path, eps=1e-5)
        folium.PolyLine(
            smoothed_path,
            color="#3498db",
            weight=2.5,
            opacity=0.7,
            tooltip="Drone path",
        ).add_to(m)

    if track_paths:
        track_colors = [
            "#e74c3c",
            "#8e44ad",
            "#16a085",
            "#f39c12",
            "#2c3e50",
            "#d35400",
            "#27ae60",
            "#c0392b",
        ]
        for idx, (track_id, coords) in enumerate(track_paths.items()):
            if len(coords) < 2:
                continue
            smoothed = _moving_average(coords, k=5)
            smoothed = _rdp(smoothed, eps=1e-5)
            if len(smoothed) < 2:
                continue
            color = track_colors[idx % len(track_colors)]
            folium.PolyLine(
                smoothed,
                color=color,
                weight=3,
                opacity=0.85,
                tooltip=f"Bison track {track_id}",
            ).add_to(m)

    for p in points:
        color = conf_to_style(p["conf"])
        popup_lines = [
            f"<b>{p['class']}</b>",
            f"conf: {p['conf']:.2f}",
        ]
        if p.get("track_id") is not None:
            popup_lines.append(f"id: {p['track_id']}")
        if p.get("track_id_original") is not None and p.get("track_id_original") != p.get("track_id"):
            popup_lines.append(f"tracker id: {p['track_id_original']}")
        if p.get("cluster_id") is not None:
            popup_lines.append(f"cluster: {p['cluster_id']}")
        if p.get("frame") is not None:
            popup_lines.append(f"frame: {p['frame']}")
        if p.get("grounded"):
            popup_lines.append("projection: ground")
        elif p.get("lat_drone") is not None and p.get("lon_drone") is not None:
            popup_lines.append("projection: drone (fallback)")
        if p.get("image"):
            popup_lines.append(str(p["image"]))

        popup = folium.Popup("<br/>".join(popup_lines), max_width=280)
        folium.CircleMarker(
            radius=6,
            location=[p["lat"], p["lon"]],
            fill=True,
            color=color,
            fill_opacity=0.8,
            weight=2,
            popup=popup,
        ).add_to(m)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_html))
    return out_html

def save_detections_map(
    detections: Iterable[Dict[str, Any]],
    out_html: Union[Path, str],
    class_names: Optional[Union[Dict[int, str], List[str]]] = None,
    drone_path: Optional[List[Tuple[float, float]]] = None,
    track_paths: Optional[Dict[Any, List[Tuple[float, float]]]] = None,
) -> Optional[Path]:
    """
    Convert raw detection rows into the format expected by write_map and persist an HTML map.
    Silently skips detections without lat/lon information.
    """
    features = _detections_to_features(detections, class_names=class_names)
    if not features:
        return None
    points = _features_to_points(features)
    if not points:
        return None
    if track_paths is None:
        track_paths = _build_track_paths(features)
    return write_map(
        points,
        Path(out_html),
        drone_path=drone_path,
        track_paths=track_paths if track_paths else None,
    )


def save_detections_geojson(
    detections: Iterable[Dict[str, Any]],
    out_geojson: Union[Path, str],
    class_names: Optional[Union[Dict[int, str], List[str]]] = None,
    indent: Optional[int] = 2,
) -> Optional[Path]:
    """
    Persist detections as a GeoJSON FeatureCollection for GIS tools.
    """
    features = _detections_to_features(detections, class_names=class_names)
    if not features:
        return None
    feature_collection = {"type": "FeatureCollection", "features": features}
    out_path = Path(out_geojson)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(feature_collection, fh, indent=indent)
        fh.write("\n")
    return out_path


def save_detections_mbtiles(
    detections: Iterable[Dict[str, Any]],
    out_mbtiles: Union[Path, str],
    class_names: Optional[Union[Dict[int, str], List[str]]] = None,
    zoom: int = 14,
    extent: int = 4096,
    layer_name: str = "detections",
) -> Optional[Path]:
    """
    Encode detections as vector tiles and store them in an MBTiles archive.
    Requires the optional mapbox_vector_tile dependency.
    """
    if mapbox_vector_tile is None:
        raise ImportError("save_detections_mbtiles requires mapbox-vector-tile; pip install mapbox-vector-tile")

    features = _detections_to_features(detections, class_names=class_names)
    if not features:
        return None

    tiles: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for feat in features:
        lon, lat = feat["geometry"]["coordinates"]
        tx, ty = _lonlat_to_tile(lon, lat, zoom)
        tiles[(tx, ty)].append(feat)

    out_path = Path(out_mbtiles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    conn = sqlite3.connect(out_path)
    cur = conn.cursor()
    cur.executescript(
        """
        PRAGMA application_id = 0x4d504258;
        CREATE TABLE IF NOT EXISTS metadata (name TEXT, value TEXT);
        CREATE TABLE IF NOT EXISTS tiles (
            zoom_level INTEGER,
            tile_column INTEGER,
            tile_row INTEGER,
            tile_data BLOB
        );
        CREATE UNIQUE INDEX IF NOT EXISTS tile_index
            ON tiles (zoom_level, tile_column, tile_row);
        DELETE FROM metadata;
        DELETE FROM tiles;
        """
    )

    bounds = _features_bounds(features)
    metadata: List[Tuple[str, str]] = [
        ("name", out_path.stem),
        ("type", "overlay"),
        ("version", "1.1"),
        ("format", "pbf"),
        ("description", "YOLO detections exported from map_writer"),
    ]
    if bounds:
        metadata.append(("bounds", ",".join(f"{v:.6f}" for v in bounds)))
        center_lon = (bounds[0] + bounds[2]) / 2.0
        center_lat = (bounds[1] + bounds[3]) / 2.0
        metadata.append(("center", f"{center_lon:.6f},{center_lat:.6f},{zoom}"))
    cur.executemany("INSERT INTO metadata (name, value) VALUES (?, ?)", metadata)

    max_y = 2 ** zoom - 1
    for (tx, ty), feats in tiles.items():
        tile_bounds = _tile_bounds(tx, ty, zoom)
        layer = {
            "name": layer_name,
            "features": [
                {"geometry": feat["geometry"], "properties": feat["properties"]}
                for feat in feats
            ],
        }
        tile_data = mapbox_vector_tile.encode(layer, quantize_bounds=tile_bounds, extents=extent)
        cur.execute(
            "INSERT INTO tiles (zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)",
            (zoom, tx, max_y - ty, sqlite3.Binary(tile_data)),
        )

    conn.commit()
    conn.close()
    return out_path
