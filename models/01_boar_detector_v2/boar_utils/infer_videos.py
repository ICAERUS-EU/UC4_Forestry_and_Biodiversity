import argparse
import csv
import math
from collections import defaultdict
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .geo_extractor import extract_geo_from_srt  # if available
from .map_writer import save_detections_map     # optional
from .geo_utils import normalize_detection, project_to_ground
from .cluster_tracker import SpatialClusterer


def _enhance_thermal(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)


def _prepare_frame(frame: np.ndarray, modality: str) -> np.ndarray:
    mode = (modality or "rgb").lower()
    if mode == "thermal":
        return _enhance_thermal(frame)
    if mode == "split":
        mid = frame.shape[1] // 2
        left = _enhance_thermal(frame[:, :mid, :])
        right = frame[:, mid:, :]
        return np.concatenate([left, right], axis=1)
    return frame


def _summarize_tracks(detections):
    tracks = defaultdict(list)
    untracked = []
    for det in detections:
        track_id = det.get("track_id")
        if track_id is None:
            untracked.append(det)
        else:
            tracks[track_id].append(det)

    summaries = []
    for track_id, dets in tracks.items():
        sorted_dets = sorted(
            dets,
            key=lambda d: d.get("frame") if d.get("frame") is not None else -1,
        )
        last = sorted_dets[-1]
        first = sorted_dets[0]
        confs = [float(d.get("conf", 0.0)) for d in dets]
        summaries.append(
            {
                "track_id": track_id,
                "cls": last.get("cls"),
                "name": last.get("name"),
                "frames": len(dets),
                "first_frame": first.get("frame"),
                "last_frame": last.get("frame"),
                "mean_conf": sum(confs) / len(confs) if confs else 0.0,
                "max_conf": max(confs) if confs else 0.0,
                "lat": last.get("lat"),
                "lon": last.get("lon"),
                "alt": last.get("alt"),
                "utc": last.get("utc"),
            }
        )
    return summaries, untracked


def _extract_box_track_id(box) -> Optional[int]:
    if not hasattr(box, "id") or box.id is None:
        return None
    try:
        track_tensor = box.id
        if track_tensor.numel():
            return int(track_tensor.item())
    except Exception:
        return None
    return None


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _suppress_overlapping_tracks(
    boxes: List[Any],
    iou_threshold: float = 0.65,
    center_threshold_px: float = 28.0,
) -> List[Any]:
    # Tracker can occasionally output two IDs for the same animal in one frame.
    # Keep the highest-confidence box when detections heavily overlap.
    kept: List[Any] = []
    kept_xyxy: List[np.ndarray] = []
    for box in boxes:
        try:
            xyxy = box.xyxy[0].cpu().numpy()
            cx = 0.5 * (float(xyxy[0]) + float(xyxy[2]))
            cy = 0.5 * (float(xyxy[1]) + float(xyxy[3]))
            cls_id = int(box.cls[0])
        except Exception:
            kept.append(box)
            kept_xyxy.append(np.zeros((4,), dtype=float))
            continue

        suppress = False
        for idx, prev in enumerate(kept):
            try:
                prev_xyxy = kept_xyxy[idx]
                prev_cx = 0.5 * (float(prev_xyxy[0]) + float(prev_xyxy[2]))
                prev_cy = 0.5 * (float(prev_xyxy[1]) + float(prev_xyxy[3]))
                prev_cls_id = int(prev.cls[0])
            except Exception:
                continue

            iou = _bbox_iou(xyxy, prev_xyxy)
            if iou >= iou_threshold:
                suppress = True
                break

            if cls_id == prev_cls_id:
                center_dist = math.hypot(cx - prev_cx, cy - prev_cy)
                if center_dist <= center_threshold_px and iou >= 0.35:
                    suppress = True
                    break

        if not suppress:
            kept.append(box)
            kept_xyxy.append(xyxy)
    return kept


def _apply_track_class_smoothing(
    result,
    track_class_scores: Dict[int, Dict[int, float]],
    track_stable_classes: Dict[int, int],
    class_switch_margin: float,
) -> Dict[int, int]:
    boxes = getattr(result, "boxes", None)
    frame_stable_classes: Dict[int, int] = {}
    if boxes is None:
        return frame_stable_classes
    for box in boxes:
        track_id = _extract_box_track_id(box)
        if track_id is None:
            continue
        try:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
        except Exception:
            continue
        class_scores = track_class_scores.setdefault(track_id, {})
        class_scores[cls_id] = class_scores.get(cls_id, 0.0) + max(conf_score, 0.0)
        stable_cls = track_stable_classes.get(track_id)
        if stable_cls is None:
            stable_cls = max(class_scores.items(), key=lambda item: item[1])[0]
            track_stable_classes[track_id] = stable_cls
        else:
            best_cls = max(class_scores.items(), key=lambda item: item[1])[0]
            stable_score = float(class_scores.get(stable_cls, 0.0))
            best_score = float(class_scores.get(best_cls, 0.0))
            if best_cls != stable_cls and best_score >= (stable_score + class_switch_margin):
                stable_cls = best_cls
                track_stable_classes[track_id] = stable_cls
        frame_stable_classes[track_id] = stable_cls
    return frame_stable_classes


def _overlay_stable_labels(frame: np.ndarray, frame_detections: List[Dict[str, Any]]) -> None:
    for det in frame_detections:
        try:
            x1 = int(float(det.get("x1", 0)))
            y1 = int(float(det.get("y1", 0)))
            x2 = int(float(det.get("x2", 0)))
            y2 = int(float(det.get("y2", 0)))
        except (TypeError, ValueError):
            continue

        label_name = str(det.get("name") or "obj")
        try:
            conf_val = float(det.get("conf", 0.0))
        except (TypeError, ValueError):
            conf_val = 0.0

        entity_id = det.get("entity_id")
        label = f"{label_name} {conf_val:.2f}"
        if entity_id is not None:
            label = f"{label} id:{entity_id}"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        text_x = max(2, x1)
        text_y = max(text_h + 4, y1 - 6)
        cv2.rectangle(
            frame,
            (text_x - 2, text_y - text_h - 4),
            (text_x + text_w + 2, text_y + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _meters_between(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))
    return radius_m * c


def _relative_offset_m(
    obj_lat: float,
    obj_lon: float,
    drone_lat: float,
    drone_lon: float,
) -> Tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians((obj_lat + drone_lat) * 0.5))
    north_m = (obj_lat - drone_lat) * meters_per_deg_lat
    east_m = (obj_lon - drone_lon) * meters_per_deg_lon
    return east_m, north_m


def _resolve_entity_id(
    *,
    frame: int,
    track_id: Optional[int],
    cluster_id: Optional[int],
    lat: Optional[float],
    lon: Optional[float],
    drone_lat: Optional[float],
    drone_lon: Optional[float],
    track_to_entity: Dict[int, int],
    track_last_seen: Dict[int, int],
    entity_states: Dict[int, Dict[str, Any]],
    next_entity_id: int,
    frame_entity_ids_used: set[int],
    reassoc_max_gap_frames: int,
    reassoc_radius_m: float,
    fps: float,
    max_entity_speed_mps: float,
) -> Tuple[int, int]:
    entity_id: Optional[int] = None

    if track_id is not None and track_id in track_to_entity:
        last_seen = track_last_seen.get(track_id, -10_000_000)
        if frame - last_seen <= reassoc_max_gap_frames:
            # Keep stable mapping for an existing tracker id.
            entity_id = track_to_entity[track_id]

    if entity_id is None:
        best_entity_id: Optional[int] = None
        best_distance = float("inf")
        for candidate_id, state in entity_states.items():
            if candidate_id in frame_entity_ids_used:
                continue
            last_frame = int(state.get("last_frame", -10_000_000))
            if frame - last_frame > reassoc_max_gap_frames:
                continue

            state_lat = state.get("lat")
            state_lon = state.get("lon")
            if (
                lat is None
                or lon is None
                or state_lat is None
                or state_lon is None
            ):
                # Without geo position we skip re-association to avoid collapsing nearby animals.
                continue
            frame_gap = max(1, frame - last_frame)
            dynamic_radius = float(reassoc_radius_m)
            if fps > 0.0 and max_entity_speed_mps > 0.0:
                gap_seconds = frame_gap / fps
                speed_radius = gap_seconds * max_entity_speed_mps + 6.0
                dynamic_radius = min(dynamic_radius, max(6.0, speed_radius))
            state_drone_lat = state.get("drone_lat")
            state_drone_lon = state.get("drone_lon")
            if (
                drone_lat is not None
                and drone_lon is not None
                and state_drone_lat is not None
                and state_drone_lon is not None
            ):
                drone_motion_m = _meters_between(
                    float(drone_lat),
                    float(drone_lon),
                    float(state_drone_lat),
                    float(state_drone_lon),
                )
                dynamic_radius = min(float(reassoc_radius_m), dynamic_radius + (drone_motion_m * 0.9))
            distance_m = _meters_between(
                float(lat), float(lon), float(state_lat), float(state_lon)
            )
            rel_distance_m: Optional[float] = None
            if (
                drone_lat is not None
                and drone_lon is not None
                and state_drone_lat is not None
                and state_drone_lon is not None
            ):
                curr_rel_e, curr_rel_n = _relative_offset_m(
                    float(lat),
                    float(lon),
                    float(drone_lat),
                    float(drone_lon),
                )
                state_rel_e, state_rel_n = _relative_offset_m(
                    float(state_lat),
                    float(state_lon),
                    float(state_drone_lat),
                    float(state_drone_lon),
                )
                rel_distance_m = math.hypot(curr_rel_e - state_rel_e, curr_rel_n - state_rel_n)

            match_distance = distance_m
            if rel_distance_m is not None:
                match_distance = min(distance_m, rel_distance_m * 1.15)

            if match_distance > dynamic_radius:
                continue
            # Prefer entities that remained in the same spatial cluster.
            if cluster_id is not None and state.get("cluster_id") == cluster_id:
                match_distance *= 0.85
            if match_distance < best_distance:
                best_distance = match_distance
                best_entity_id = candidate_id

        if best_entity_id is not None:
            entity_id = best_entity_id
        else:
            entity_id = next_entity_id
            next_entity_id += 1

        if track_id is not None:
            track_to_entity[track_id] = entity_id

    state = entity_states.setdefault(entity_id, {})
    state["last_frame"] = frame
    state["cluster_id"] = cluster_id
    if lat is not None and lon is not None:
        state["lat"] = float(lat)
        state["lon"] = float(lon)
    if drone_lat is not None and drone_lon is not None:
        state["drone_lat"] = float(drone_lat)
        state["drone_lon"] = float(drone_lon)
    if track_id is not None:
        known_tracks = state.setdefault("track_ids", set())
        if isinstance(known_tracks, set):
            known_tracks.add(track_id)
        track_to_entity[track_id] = entity_id
        track_last_seen[track_id] = frame

    frame_entity_ids_used.add(entity_id)
    return entity_id, next_entity_id


def _dedupe_frame_detections(
    frame_detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Keep only one detection per stable entity per frame to avoid identity churn
    # when tracker emits overlapping tracks for the same object.
    deduped: List[Dict[str, Any]] = []
    best_index_by_entity: Dict[int, int] = {}
    for det in frame_detections:
        entity_id = det.get("entity_id")
        if entity_id is None:
            deduped.append(det)
            continue
        try:
            entity_id_int = int(entity_id)
        except (TypeError, ValueError):
            deduped.append(det)
            continue

        try:
            curr_conf = float(det.get("conf", 0.0))
        except (TypeError, ValueError):
            curr_conf = 0.0

        existing_idx = best_index_by_entity.get(entity_id_int)
        if existing_idx is None:
            best_index_by_entity[entity_id_int] = len(deduped)
            deduped.append(det)
            continue

        prev = deduped[existing_idx]
        try:
            prev_conf = float(prev.get("conf", 0.0))
        except (TypeError, ValueError):
            prev_conf = 0.0
        if curr_conf > prev_conf:
            deduped[existing_idx] = det

    return deduped


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _merge_entity_fragments(
    detections: List[Dict[str, Any]],
    fps: float,
    max_gap_frames: int,
    base_radius_m: float,
    max_speed_mps: float,
) -> Tuple[int, Dict[int, int]]:
    by_entity: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for det in detections:
        entity_id = det.get("entity_id")
        if entity_id is None:
            continue
        try:
            entity_key = int(entity_id)
        except (TypeError, ValueError):
            continue
        by_entity[entity_key].append(det)

    if len(by_entity) < 2:
        return 0, {}

    stats: Dict[int, Dict[str, Any]] = {}
    for entity_id, rows in by_entity.items():
        rows.sort(key=lambda d: int(d.get("frame", 0)))
        start = rows[0]
        end = rows[-1]
        class_votes: Dict[str, int] = defaultdict(int)
        for row in rows:
            class_votes[str(row.get("name") or "")] += 1
        main_name = max(class_votes.items(), key=lambda kv: kv[1])[0]

        start_rel: Optional[Tuple[float, float]] = None
        end_rel: Optional[Tuple[float, float]] = None
        for row in rows:
            lat = _safe_float(row.get("lat"))
            lon = _safe_float(row.get("lon"))
            dlat = _safe_float(row.get("lat_drone"))
            dlon = _safe_float(row.get("lon_drone"))
            if None in (lat, lon, dlat, dlon):
                continue
            start_rel = _relative_offset_m(lat, lon, dlat, dlon)
            break
        for row in reversed(rows):
            lat = _safe_float(row.get("lat"))
            lon = _safe_float(row.get("lon"))
            dlat = _safe_float(row.get("lat_drone"))
            dlon = _safe_float(row.get("lon_drone"))
            if None in (lat, lon, dlat, dlon):
                continue
            end_rel = _relative_offset_m(lat, lon, dlat, dlon)
            break

        stats[entity_id] = {
            "start_frame": int(start.get("frame", 0)),
            "end_frame": int(end.get("frame", 0)),
            "start_abs": (_safe_float(start.get("lat")), _safe_float(start.get("lon"))),
            "end_abs": (_safe_float(end.get("lat")), _safe_float(end.get("lon"))),
            "start_rel": start_rel,
            "end_rel": end_rel,
            "name": main_name,
        }

    entity_ids = sorted(stats.keys())
    candidates: List[Tuple[float, float, int, int]] = []
    fps_safe = fps if fps and fps > 0 else 1.0
    for src in entity_ids:
        src_stats = stats[src]
        for dst in entity_ids:
            if dst == src:
                continue
            dst_stats = stats[dst]
            gap = int(dst_stats["start_frame"]) - int(src_stats["end_frame"])
            if gap <= 0 or gap > max_gap_frames:
                continue
            if src_stats["name"] and dst_stats["name"] and src_stats["name"] != dst_stats["name"]:
                continue

            distance_m: Optional[float] = None
            if src_stats["end_rel"] is not None and dst_stats["start_rel"] is not None:
                sx, sy = src_stats["end_rel"]
                dx, dy = dst_stats["start_rel"]
                distance_m = math.hypot(dx - sx, dy - sy)
            else:
                src_abs = src_stats["end_abs"]
                dst_abs = dst_stats["start_abs"]
                if None not in (src_abs[0], src_abs[1], dst_abs[0], dst_abs[1]):
                    distance_m = _meters_between(
                        float(src_abs[0]), float(src_abs[1]), float(dst_abs[0]), float(dst_abs[1])
                    )
            if distance_m is None:
                continue

            gap_seconds = gap / fps_safe
            allowed = base_radius_m + max_speed_mps * gap_seconds
            if distance_m > allowed:
                continue
            score = distance_m / max(allowed, 1e-6)
            candidates.append((score, distance_m, src, dst))

    if not candidates:
        return 0, {}

    parent: Dict[int, int] = {entity_id: entity_id for entity_id in entity_ids}
    has_successor: set[int] = set()
    has_predecessor: set[int] = set()

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merges = 0
    for _, _, src, dst in sorted(candidates, key=lambda x: (x[0], x[1])):
        root_src = find(src)
        root_dst = find(dst)
        if root_src == root_dst:
            continue
        if root_src in has_successor or root_dst in has_predecessor:
            continue
        # Keep earlier entity id as canonical id.
        keep = min(root_src, root_dst)
        drop = max(root_src, root_dst)
        parent[drop] = keep
        has_successor.add(keep)
        has_predecessor.add(drop)
        merges += 1

    if merges == 0:
        return 0, {}

    canonical_map: Dict[int, int] = {}
    for entity_id in entity_ids:
        canonical_map[entity_id] = find(entity_id)

    for det in detections:
        entity_id = det.get("entity_id")
        if entity_id is None:
            continue
        try:
            entity_key = int(entity_id)
        except (TypeError, ValueError):
            continue
        if entity_key not in canonical_map:
            continue
        merged = canonical_map[entity_key]
        det["entity_id"] = merged
        det["track_id"] = merged

    return merges, canonical_map


def _entity_class_key(
    entity_id: Optional[int], cluster_id: Optional[int], track_id: Optional[int]
) -> Optional[str]:
    if entity_id is not None:
        return f"entity:{entity_id}"
    if cluster_id is not None:
        return f"cluster:{cluster_id}"
    if track_id is not None:
        return f"track:{track_id}"
    return None


def _apply_entity_class_smoothing(
    norm: Dict[str, Any],
    class_names: Dict[int, str] | List[str],
    entity_class_scores: Dict[str, Dict[int, float]],
    entity_class_hits: Dict[str, Dict[int, int]],
    entity_stable_classes: Dict[str, int],
    class_switch_margin: float,
) -> None:
    try:
        cls_id = int(norm.get("cls", 0))
        conf_score = float(norm.get("conf", 0.0))
    except (TypeError, ValueError):
        return

    entity_id = norm.get("entity_id")
    cluster_id = norm.get("cluster_id")
    track_id = norm.get("track_id_original")
    entity_key = _entity_class_key(entity_id, cluster_id, track_id)
    if entity_key is None:
        return

    class_scores = entity_class_scores.setdefault(entity_key, {})
    class_hits = entity_class_hits.setdefault(entity_key, {})
    class_scores[cls_id] = class_scores.get(cls_id, 0.0) + max(conf_score, 0.0)
    class_hits[cls_id] = class_hits.get(cls_id, 0) + 1

    stable_cls = entity_stable_classes.get(entity_key)
    if stable_cls is None:
        stable_cls = max(class_scores.items(), key=lambda item: item[1])[0]
        entity_stable_classes[entity_key] = stable_cls
    else:
        best_cls = max(class_scores.items(), key=lambda item: item[1])[0]
        stable_score = float(class_scores.get(stable_cls, 0.0))
        best_score = float(class_scores.get(best_cls, 0.0))
        best_hits = int(class_hits.get(best_cls, 0))
        # Require repeated evidence before class switch to avoid frame-level flicker.
        if (
            best_cls != stable_cls
            and best_hits >= 3
            and best_score >= (stable_score + max(0.0, class_switch_margin))
        ):
            stable_cls = best_cls
            entity_stable_classes[entity_key] = stable_cls

    if stable_cls == cls_id:
        return

    norm["cls_original"] = norm.get("cls")
    norm["name_original"] = norm.get("name")
    norm["cls"] = stable_cls
    if isinstance(class_names, dict):
        norm["name"] = class_names.get(stable_cls, str(stable_cls))
    elif isinstance(class_names, (list, tuple)):
        norm["name"] = class_names[stable_cls] if 0 <= stable_cls < len(class_names) else str(stable_cls)
    else:
        norm["name"] = str(stable_cls)


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _derive_geo_calibration(
    geo_data: Optional[list],
    default_cluster_radius: float,
    default_cluster_reassoc: float,
    default_cluster_freeze: int,
) -> Dict[str, Any]:
    calibration: Dict[str, Any] = {
        "pitch_bias": 0.0,
        "pitch_bias_applied": False,
        "median_pitch": None,
        "median_rel_alt": None,
        "cluster_radius_m": default_cluster_radius,
        "cluster_reassoc_m": default_cluster_reassoc,
        "cluster_freeze_after": default_cluster_freeze,
        "max_ground_distance": None,
    }
    if not geo_data:
        return calibration

    pitch_vals = [entry.get("gb_pitch") for entry in geo_data if entry.get("gb_pitch") is not None]
    rel_alt_vals = [entry.get("rel_alt") for entry in geo_data if entry.get("rel_alt") is not None]

    median_pitch = _median(pitch_vals) if pitch_vals else None
    median_rel_alt = _median(rel_alt_vals) if rel_alt_vals else None
    calibration["median_pitch"] = median_pitch
    calibration["median_rel_alt"] = median_rel_alt

    if median_pitch is None or median_rel_alt is None:
        return calibration

    off_nadir = abs(median_pitch + 90.0)
    if off_nadir > 1.0:
        pitch_bias = max(-45.0, min(45.0, median_pitch + 90.0))
        calibration["pitch_bias"] = pitch_bias
        calibration["pitch_bias_applied"] = True

    expected_span = 0.0
    if off_nadir > 0.5:
        expected_span = float(median_rel_alt) * math.tan(math.radians(off_nadir))

    suggested_radius = default_cluster_radius
    suggested_reassoc = default_cluster_reassoc
    suggested_freeze = min(default_cluster_freeze, 12)

    if expected_span:
        # Keep the live clustering radius conservative so nearby animals remain separate.
        suggested_radius = default_cluster_radius
        # Allow re-association to reach further once the camera slews.
        suggested_reassoc = max(
            default_cluster_reassoc,
            min(8.0, expected_span * 0.4),
        )

    calibration["cluster_radius_m"] = suggested_radius
    calibration["cluster_reassoc_m"] = suggested_reassoc
    calibration["cluster_freeze_after"] = suggested_freeze

    if suggested_reassoc != default_cluster_reassoc:
        calibration["cluster_reassoc_auto"] = True
    if suggested_freeze != default_cluster_freeze:
        calibration["cluster_freeze_auto"] = True

    if expected_span:
        calibration["max_ground_distance"] = max(500.0, expected_span * 2.0)

    return calibration


def _estimate_intrinsics(width: int, height: int, focal_len_mm: Optional[float], dzoom: Optional[float]) -> Optional[Tuple[float, float]]:
    if focal_len_mm is None or focal_len_mm <= 0:
        return None
    zoom = dzoom if dzoom and dzoom > 0 else 1.0
    effective_focal = focal_len_mm * zoom
    try:
        hfov = 2.0 * math.atan(36.0 / (2.0 * effective_focal))
        vfov = 2.0 * math.atan(24.0 / (2.0 * effective_focal))
        fx = (width / 2.0) / math.tan(hfov / 2.0)
        fy = (height / 2.0) / math.tan(vfov / 2.0)
        return fx, fy
    except Exception:
        return None


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    # DJI gimbal pitch is -90 when the camera points straight down. Offset so that
    # pitch = -90 -> pitch_eff = 0 (nadir) in our rotation model.
    pitch = math.radians(pitch_deg + 90.0)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    return Rz @ Ry @ Rx


def _project_detection_to_ground(
    bbox: Dict[str, float],
    image_dims: Tuple[int, int],
    geo_meta: Dict[str, Any],
    default_focal: Optional[float] = None,
    geo_calibration: Optional[Dict[str, Any]] = None,
) -> Optional[Tuple[float, float]]:
    lat = geo_meta.get("lat")
    lon = geo_meta.get("lon")
    rel_alt = geo_meta.get("rel_alt")
    if rel_alt is None:
        rel_alt = geo_meta.get("alt") or geo_meta.get("abs_alt")
    yaw = geo_meta.get("gb_yaw")
    pitch = geo_meta.get("gb_pitch")
    roll = geo_meta.get("gb_roll", 0.0)
    focal_len = geo_meta.get("focal_len") if geo_meta.get("focal_len") is not None else default_focal
    dzoom = geo_meta.get("dzoom_ratio")

    pitch_bias = 0.0
    max_ground_distance_override = None
    if geo_calibration:
        pitch_bias = float(geo_calibration.get("pitch_bias", 0.0))
        max_ground_distance_override = geo_calibration.get("max_ground_distance")

    if lat is None or lon is None or rel_alt is None or yaw is None or pitch is None or focal_len is None:
        return None

    width, height = image_dims
    intrinsics = _estimate_intrinsics(width, height, float(focal_len), float(dzoom) if dzoom is not None else 1.0)
    if intrinsics is None:
        return None
    fx, fy = intrinsics
    if fx <= 0 or fy <= 0:
        return None

    pitch = float(pitch) - pitch_bias

    cx = (bbox.get("x1", 0.0) + bbox.get("x2", 0.0)) * 0.5
    cy = (bbox.get("y1", 0.0) + bbox.get("y2", 0.0)) * 0.5
    principal_x = width / 2.0
    principal_y = height / 2.0

    x_cam = (cx - principal_x) / fx
    y_cam = -(cy - principal_y) / fy  # invert to make positive up
    # Camera looks along negative Z after the pitch offset above.
    ray_cam = np.array([x_cam, y_cam, -1.0], dtype=float)
    ray_cam /= np.linalg.norm(ray_cam)

    R = _rotation_matrix(float(yaw), float(pitch), float(roll))
    ray_world = R @ ray_cam
    ray_world /= np.linalg.norm(ray_world)

    dir_z = ray_world[2]
    if dir_z >= -1e-3:
        return None

    camera_height = float(rel_alt)
    t = camera_height / -dir_z
    east_offset = ray_world[0] * t
    north_offset = ray_world[1] * t
    horizontal_offset = math.hypot(east_offset, north_offset)

    # Reject projections that imply implausibly long ground offsets. This typically happens when the
    # optical axis is close to horizontal and would otherwise explode the map coordinates.
    max_ground_distance = max(15.0 * camera_height, 500.0)
    if max_ground_distance_override is not None:
        try:
            max_ground_distance = max(max_ground_distance, float(max_ground_distance_override))
        except (TypeError, ValueError):
            pass
    if not math.isfinite(horizontal_offset) or horizontal_offset > max_ground_distance:
        return None

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(float(lat)))
    if not meters_per_deg_lon:
        return None

    lat_ground = float(lat) + (north_offset / meters_per_deg_lat)
    lon_ground = float(lon) + (east_offset / meters_per_deg_lon if meters_per_deg_lon else 0.0)
    if not (math.isfinite(lat_ground) and math.isfinite(lon_ground)):
        return None
    if not (-90.0 <= lat_ground <= 90.0):
        return None
    # Wrap longitude into [-180, 180] to keep within map bounds.
    lon_ground = ((lon_ground + 180.0) % 360.0) - 180.0
    if not (-180.0 <= lon_ground <= 180.0):
        return None
    return lat_ground, lon_ground


def run_inference(
    weights,
    source,
    srt=None,
    imgsz=960,
    conf=0.25,
    iou=0.45,
    device="0",
    save_video=False,
    save_csv=None,
    project="runs/infer/videos",
    name="exp",
    vid_stride=1,
    tracker=None,
    default_focal_mm=None,
    cluster_radius_m=5.0,
    cluster_idle_frames=120,
    cluster_min_hits=3,
    min_track_frames=3,
    min_track_conf=0.5,
    min_detection_conf=0.35,
    allow_no_ground=False,
    cluster_freeze_after=30,
    cluster_reassoc_m=30.0,
    entity_reassoc_radius_m=25.0,
    entity_reassoc_gap_frames=300,
    entity_max_speed_mps=8.0,
    entity_merge_gap_frames=220,
    entity_merge_base_radius_m=10.0,
    entity_merge_speed_mps=5.0,
    hfov=None,
    vfov=None,
    ground_alt=0.0,
    modality="rgb",
    class_switch_margin=1.5,
):
    # Load YOLOv11 model
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = YOLO(weights)
    print(
        "[INFO] identity matcher: drone-motion-v2 (dynamic reassoc + overlap suppression)",
        flush=True,
    )
    mode = (modality or "rgb").lower()

    # Prepare output folder
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Optional: video writer
    if save_video:
        out_path = str(save_dir / "result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    else:
        out = None

    # Optional: CSV logging
    raw_detections = []
    track_counts: Dict[int, int] = defaultdict(int)
    track_conf_sums: Dict[int, float] = defaultdict(float)
    track_class_scores: Dict[int, Dict[int, float]] = {}
    track_stable_classes: Dict[int, int] = {}
    frame_track_classes: Dict[int, int] = {}
    entity_class_scores: Dict[str, Dict[int, float]] = {}
    entity_class_hits: Dict[str, Dict[int, int]] = {}
    entity_stable_classes: Dict[str, int] = {}
    csv_path = Path(save_csv) if save_csv else None

    # Optional: load GPS data from .SRT
    geo_data = extract_geo_from_srt(srt) if srt and os.path.exists(srt) else None
    geo_lookup = {}
    if geo_data:
        for idx, entry in enumerate(geo_data):
            key = entry.get("frame")
            if isinstance(key, int) and key >= 0:
                geo_lookup.setdefault(key, entry)
            else:
                geo_lookup.setdefault(idx, entry)

    geo_calibration = _derive_geo_calibration(
        geo_data,
        cluster_radius_m,
        cluster_reassoc_m,
        cluster_freeze_after,
    )
    if geo_calibration.get("pitch_bias_applied"):
        pitch_bias_val = geo_calibration.get("pitch_bias", 0.0)
        median_pitch = geo_calibration.get("median_pitch")
        median_pitch_str = f"{median_pitch:.2f}" if median_pitch is not None else "n/a"
        print(
            f"[INFO] Pitch bias correction {pitch_bias_val:+.2f}° applied (median pitch {median_pitch_str}°)"
        )
    if geo_calibration.get("cluster_radius_auto"):
        print(
            "[INFO] Auto-adjusted cluster radius to {:.1f} m based on SRT pitch".format(
                geo_calibration["cluster_radius_m"]
            )
        )
    if geo_calibration.get("cluster_reassoc_auto"):
        print(
            "[INFO] Auto-adjusted cluster reassociation radius to {:.1f} m".format(
                geo_calibration["cluster_reassoc_m"]
            )
        )
    if geo_calibration.get("cluster_freeze_auto"):
        print(
            "[INFO] Auto-adjusted cluster freeze threshold to {} detections".format(
                geo_calibration["cluster_freeze_after"]
            )
        )

    cluster_radius_m = geo_calibration.get("cluster_radius_m", cluster_radius_m)
    cluster_reassoc_m = geo_calibration.get("cluster_reassoc_m", cluster_reassoc_m)
    cluster_freeze_after = geo_calibration.get("cluster_freeze_after", cluster_freeze_after)

    clusterer = SpatialClusterer(
        radius_m=cluster_radius_m,
        max_idle_frames=cluster_idle_frames,
        min_detections=cluster_min_hits,
        freeze_after=cluster_freeze_after if cluster_freeze_after and cluster_freeze_after > 0 else None,
        reassoc_radius_m=cluster_reassoc_m,
    )
    entity_reassoc_max_gap_frames = max(int(entity_reassoc_gap_frames), 30)
    entity_reassoc_radius_m = max(float(entity_reassoc_radius_m), 3.0)
    track_to_entity: Dict[int, int] = {}
    track_last_seen: Dict[int, int] = {}
    entity_states: Dict[int, Dict[str, Any]] = {}
    next_entity_id = 1

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % vid_stride != 0:
            frame_idx += 1
            continue

        infer_frame = _prepare_frame(frame, mode)

        if tracker:
            results = model.track(
                infer_frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
                persist=True,
                tracker=tracker,
            )
        else:
            results = model.predict(
                infer_frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=device,
                verbose=False,
            )

        if tracker:
            frame_track_classes = _apply_track_class_smoothing(
                results[0],
                track_class_scores,
                track_stable_classes,
                max(0.0, float(class_switch_margin)),
            )
        else:
            frame_track_classes = {}

        # Save detection info
        frame_detections: List[Dict[str, Any]] = []
        frame_entity_ids_used: set[int] = set()
        boxes_iter = list(results[0].boxes)
        boxes_iter.sort(
            key=lambda box: float(box.conf[0]) if hasattr(box, "conf") else 0.0,
            reverse=True,
        )
        boxes_iter = _suppress_overlapping_tracks(boxes_iter)
        for box in boxes_iter:
            xyxy = box.xyxy[0].cpu().numpy()
            conf_score = float(box.conf[0])
            raw_cls_id = int(box.cls[0])
            track_id = _extract_box_track_id(box) if tracker else None
            cls_id = frame_track_classes.get(track_id, raw_cls_id) if tracker else raw_cls_id
            label = model.names[cls_id]
            if conf_score < min_detection_conf:
                continue
            frame_info = {
                "frame": frame_idx,
                "label": label,
                "conf": conf_score,
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3],
            }
            # Add GPS if available
            if geo_lookup:
                geo_entry = geo_lookup.get(frame_idx)
                if geo_entry:
                    frame_info.update(geo_entry)
            frame_info["cls"] = cls_id
            frame_info["track_id"] = track_id
            norm = normalize_detection(frame_info, class_names=model.names)
            norm["track_id_original"] = track_id
            drone_lat = norm.get("lat")
            drone_lon = norm.get("lon")
            drone_alt = norm.get("rel_alt") or norm.get("alt") or norm.get("abs_alt")
            yaw = norm.get("gb_yaw")
            pitch = norm.get("gb_pitch")
            roll = norm.get("gb_roll")
            norm["lat_drone"] = drone_lat
            norm["lon_drone"] = drone_lon
            ground_point = None
            hfov_use = hfov
            vfov_use = vfov
            try:
                focal_len = norm.get("focal_len")
                dzoom_ratio = norm.get("dzoom_ratio") or 1.0
                if (hfov_use is None or vfov_use is None) and focal_len:
                    focal_equiv = float(focal_len) * float(dzoom_ratio)
                    if hfov_use is None:
                        hfov_use = math.degrees(2.0 * math.atan(36.0 / (2.0 * focal_equiv)))
                    if vfov_use is None:
                        vfov_use = math.degrees(2.0 * math.atan(24.0 / (2.0 * focal_equiv)))
            except Exception:
                pass
            if (
                drone_lat is not None
                and drone_lon is not None
                and drone_alt is not None
                and yaw is not None
                and pitch is not None
                and roll is not None
            ):
                u = (norm.get("x1", 0.0) + norm.get("x2", 0.0)) * 0.5
                v = (norm.get("y1", 0.0) + norm.get("y2", 0.0)) * 0.5
                try:
                    ground_point = project_to_ground(
                        u,
                        v,
                        width,
                        height,
                        float(drone_lat),
                        float(drone_lon),
                        float(drone_alt),
                        float(yaw),
                        float(pitch),
                        float(roll),
                        hfov=hfov_use,
                        vfov=vfov_use,
                        ground_alt_m=ground_alt,
                    )
                except Exception:
                    ground_point = None
            if ground_point:
                ground_lat, ground_lon = ground_point
                norm["lat"] = ground_lat
                norm["lon"] = ground_lon
                norm["grounded"] = True
            else:
                norm["lat"] = None
                norm["lon"] = None
                norm["grounded"] = False
            cluster_lat = norm.get("lat") if norm.get("grounded") else None
            cluster_lon = norm.get("lon") if norm.get("grounded") else None
            cluster_id = clusterer.update(
                frame=frame_idx,
                lat=cluster_lat,
                lon=cluster_lon,
                conf=norm["conf"],
                track_id=norm.get("track_id_original"),
            )
            if cluster_id is not None:
                norm["cluster_id"] = cluster_id
                norm["cluster_confirmed"] = clusterer.is_confirmed(cluster_id)
                centroid = clusterer.centroid(cluster_id)
                if centroid:
                    norm["cluster_lat"] = centroid[0]
                    norm["cluster_lon"] = centroid[1]
            else:
                norm["cluster_confirmed"] = False

            stable_id, next_entity_id = _resolve_entity_id(
                frame=frame_idx,
                track_id=norm.get("track_id_original"),
                cluster_id=cluster_id,
                lat=norm.get("lat"),
                lon=norm.get("lon"),
                drone_lat=norm.get("lat_drone"),
                drone_lon=norm.get("lon_drone"),
                track_to_entity=track_to_entity,
                track_last_seen=track_last_seen,
                entity_states=entity_states,
                next_entity_id=next_entity_id,
                frame_entity_ids_used=frame_entity_ids_used,
                reassoc_max_gap_frames=entity_reassoc_max_gap_frames,
                reassoc_radius_m=entity_reassoc_radius_m,
                fps=float(fps) if fps and fps > 0 else 0.0,
                max_entity_speed_mps=max(float(entity_max_speed_mps), 0.0),
            )
            norm["entity_id"] = stable_id
            norm["track_id"] = stable_id
            frame_detections.append(norm)

        frame_detections = _dedupe_frame_detections(frame_detections)
        for det in frame_detections:
            track_id = det.get("track_id_original")
            if track_id is not None:
                track_counts[track_id] += 1
                track_conf_sums[track_id] += float(det.get("conf", 0.0))
            _apply_entity_class_smoothing(
                det,
                model.names,
                entity_class_scores,
                entity_class_hits,
                entity_stable_classes,
                class_switch_margin,
            )
            raw_detections.append(det)

        if save_video and out:
            annotated = frame.copy()
            _overlay_stable_labels(annotated, frame_detections)
            out.write(annotated)

        frame_idx += 1

    cap.release()
    if out:
        out.release()

    valid_track_ids = set()
    min_track_frames = max(1, min_track_frames)
    for track_id, count in track_counts.items():
        mean_conf = track_conf_sums[track_id] / count if count else 0.0
        if count >= min_track_frames and mean_conf >= min_track_conf:
            valid_track_ids.add(track_id)

    confirmed_detections = []
    rejected_detections = []
    for det in raw_detections:
        if not det.get("grounded"):
            if allow_no_ground:
                det["status"] = "confirmed"
                confirmed_detections.append(det)
                continue
            det["status"] = "rejected"
            det["reject_reason"] = "no_ground_projection"
            det["cls_original"] = det.get("cls")
            det["name_original"] = det.get("name")
            rejected_detections.append(det)
            continue
        track_id_original = det.get("track_id_original")
        cluster_id = det.get("cluster_id")
        cluster_confirmed = det.get("cluster_confirmed")
        is_confirmed = False
        reject_reason = ""
        if cluster_confirmed and cluster_id is not None:
            is_confirmed = True
        elif track_id_original is not None:
            if track_id_original in valid_track_ids:
                is_confirmed = True
            else:
                reject_reason = "track_short_or_low_conf"
        else:
            reject_reason = "cluster_unconfirmed"

        if is_confirmed:
            det["status"] = "confirmed"
            confirmed_detections.append(det)
        else:
            det["status"] = "rejected"
            det["reject_reason"] = reject_reason or "unconfirmed"
            det["cls_original"] = det.get("cls")
            det["name_original"] = det.get("name")
            rejected_detections.append(det)

    merged_entities, merge_map = _merge_entity_fragments(
        confirmed_detections,
        fps=float(fps) if fps and fps > 0 else 1.0,
        max_gap_frames=max(1, int(entity_merge_gap_frames)),
        base_radius_m=max(1.0, float(entity_merge_base_radius_m)),
        max_speed_mps=max(0.1, float(entity_merge_speed_mps)),
    )
    if merged_entities > 0:
        print(f"[INFO] Merged entity fragments: {merged_entities}", flush=True)
        for det in rejected_detections:
            entity_id = det.get("entity_id")
            if entity_id is None:
                continue
            try:
                entity_key = int(entity_id)
            except (TypeError, ValueError):
                continue
            merged_entity = merge_map.get(entity_key)
            if merged_entity is not None:
                det["entity_id"] = merged_entity
                det["track_id"] = merged_entity

    if csv_path:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as csv_file:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "frame",
                    "cls",
                    "name",
                    "entity_id",
                    "cls_original",
                    "name_original",
                    "conf",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "lat",
                    "lon",
                    "alt",
                    "rel_alt",
                    "abs_alt",
                    "utc",
                    "track_id",
                    "track_id_original",
                    "cluster_id",
                    "lat_drone",
                    "lon_drone",
                    "grounded",
                    "status",
                    "reject_reason",
                ],
            )
            csv_writer.writeheader()
            for det in confirmed_detections + rejected_detections:
                csv_writer.writerow(
                    {
                        "frame": det.get("frame"),
                        "cls": det.get("cls"),
                        "name": det.get("name"),
                        "entity_id": det.get("entity_id"),
                        "cls_original": det.get("cls_original", ""),
                        "name_original": det.get("name_original", ""),
                        "conf": det.get("conf"),
                        "x1": det.get("x1"),
                        "y1": det.get("y1"),
                        "x2": det.get("x2"),
                        "y2": det.get("y2"),
                        "lat": det.get("lat") if det.get("lat") is not None else "",
                        "lon": det.get("lon") if det.get("lon") is not None else "",
                        "alt": det.get("alt"),
                        "rel_alt": det.get("rel_alt"),
                        "abs_alt": det.get("abs_alt"),
                        "utc": det.get("utc"),
                        "track_id": det.get("track_id"),
                        "track_id_original": det.get("track_id_original"),
                        "cluster_id": det.get("cluster_id"),
                        "lat_drone": det.get("lat_drone"),
                        "lon_drone": det.get("lon_drone"),
                        "grounded": det.get("grounded"),
                        "status": det.get("status"),
                        "reject_reason": det.get("reject_reason", ""),
                    }
                )
        print(f"[INFO] Saved detections to {csv_path}")
    print(f"[INFO] Confirmed detections: {len(confirmed_detections)}")
    if rejected_detections:
        print(f"[INFO] Rejected detections: {len(rejected_detections)}")

    track_summaries, untracked = _summarize_tracks(confirmed_detections)
    track_csv_path = None
    if csv_path and track_summaries:
        track_csv_path = csv_path.with_name(f"{csv_path.stem}_tracks.csv")
        with track_csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "track_id",
                    "cls",
                    "name",
                    "frames",
                    "first_frame",
                    "last_frame",
                    "mean_conf",
                    "max_conf",
                    "lat",
                    "lon",
                    "alt",
                    "utc",
                ],
            )
            writer.writeheader()
            for row in track_summaries:
                writer.writerow(row)
        print(f"[INFO] Saved track summaries to {track_csv_path}")
    if track_summaries:
        print(f"[INFO] Active tracks: {len(track_summaries)} (untracked detections: {len(untracked)})")

    cluster_csv_path = None
    cluster_summaries = clusterer.summaries()
    if csv_path and cluster_summaries:
        cluster_csv_path = csv_path.with_name(f"{csv_path.stem}_clusters.csv")
        with cluster_csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "cluster_id",
                    "frames",
                    "first_frame",
                    "last_frame",
                    "mean_conf",
                    "max_conf",
                    "lat",
                    "lon",
                    "track_ids",
                ],
            )
            writer.writeheader()
            for row in cluster_summaries:
                writer.writerow(row)
        print(f"[INFO] Saved cluster summaries to {cluster_csv_path}")
        print(f"[INFO] Unique clusters: {len(cluster_summaries)}")

    entity_points: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for det in confirmed_detections:
        if not det.get("grounded"):
            continue
        eid = det.get("entity_id")
        lat = det.get("lat")
        lon = det.get("lon")
        if eid is None or lat is None or lon is None:
            continue
        try:
            frame_id = int(det.get("frame", 0))
            lat_f = float(lat)
            lon_f = float(lon)
        except (TypeError, ValueError):
            continue
        entity_points[str(eid)].append((frame_id, lat_f, lon_f))

    entity_track_paths: Dict[str, List[Tuple[float, float]]] = {}
    for eid, pts in entity_points.items():
        pts.sort(key=lambda x: x[0])
        coords = [(lat, lon) for _, lat, lon in pts]
        if len(coords) >= 2:
            entity_track_paths[eid] = coords

    # Optional: save a map of detections
    drone_path = []
    if geo_data:
        for entry in geo_data:
            lat = entry.get("lat")
            lon = entry.get("lon")
            if lat is None or lon is None:
                continue
            drone_path.append((float(lat), float(lon)))

    map_path = save_detections_map(
        confirmed_detections,
        save_dir / "map.html",
        class_names=model.names,
        drone_path=drone_path if drone_path else None,
        track_paths=entity_track_paths if entity_track_paths else None,
    )
    if map_path:
        print(f"[INFO] Saved map to {map_path}")

    print(f"[INFO] Inference completed. Results saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to YOLO weights file (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path to input video")
    parser.add_argument("--srt", type=str, default=None, help="Path to video SRT file with GPS data")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="GPU device id")
    parser.add_argument("--project", type=str, default="runs/infer/videos", help="Save directory")
    parser.add_argument("--name", type=str, default="exp", help="Run name")
    parser.add_argument("--save-video", action="store_true", help="Save output video with bounding boxes")
    parser.add_argument("--save-csv", type=str, default=None, help="Save detections as CSV")
    parser.add_argument("--vid-stride", type=int, default=1, help="Frame skip interval")
    parser.add_argument("--tracker", type=str, default=None, help="Tracker config YAML (e.g., 'bytetrack.yaml')")
    parser.add_argument("--default-focal-mm", type=float, default=None, help="Fallback focal length (35mm equivalent) if missing from metadata")
    parser.add_argument("--cluster-radius-m", type=float, default=5.0, help="Clustering radius on ground plane (meters)")
    parser.add_argument("--cluster-idle-frames", type=int, default=120, help="How many frames to keep a cluster alive without observations")
    parser.add_argument("--cluster-min-hits", type=int, default=3, help="Minimum detections required to keep/export a cluster")
    parser.add_argument("--cluster-freeze-after", type=int, default=30, help="Number of detections after which cluster centroid is frozen (set to 0 to disable)")
    parser.add_argument("--cluster-reassoc-m", type=float, default=30.0, help="Fallback spatial radius (meters) to re-associate detections with frozen clusters")
    parser.add_argument("--entity-reassoc-radius-m", type=float, default=25.0, help="Spatial radius (meters) for re-linking the same entity after tracker ID switches")
    parser.add_argument("--entity-reassoc-gap-frames", type=int, default=300, help="Max frame gap to keep entity identity between tracker ID switches")
    parser.add_argument("--entity-max-speed-mps", type=float, default=8.0, help="Maximum expected animal speed (m/s) used for dynamic entity re-association gating")
    parser.add_argument("--entity-merge-gap-frames", type=int, default=220, help="Max frame gap for merging split entity fragments after inference")
    parser.add_argument("--entity-merge-base-radius-m", type=float, default=10.0, help="Base merge distance (meters) between fragment endpoints")
    parser.add_argument("--entity-merge-speed-mps", type=float, default=5.0, help="Additional merge allowance per second of gap")
    parser.add_argument("--min-track-frames", type=int, default=3, help="Minimum frames required for a track to be kept in outputs")
    parser.add_argument("--min-track-conf", type=float, default=0.5, help="Minimum mean confidence required for a track to be kept")
    parser.add_argument("--class-switch-margin", type=float, default=1.5, help="Minimum cumulative confidence lead required to switch class within a track")
    parser.add_argument("--min-detection-conf", type=float, default=0.35, help="Minimum confidence required for raw detections before tracking/clustering")
    parser.add_argument("--allow-no-ground", action="store_true", help="Keep detections even when ground projection is unavailable")
    parser.add_argument("--hfov", type=float, default=None, help="Horizontal field of view in degrees (overrides SRT metadata if provided)")
    parser.add_argument("--vfov", type=float, default=None, help="Vertical field of view in degrees (overrides SRT metadata if provided)")
    parser.add_argument("--ground-alt", type=float, default=0.0, help="Ground altitude (meters) for projection baseline")
    parser.add_argument("--modality", type=str, default="rgb", choices=["rgb", "thermal", "split"], help="Video modality for preprocessing")

    args = parser.parse_args()
    run_inference(**vars(args))
