# src/cluster_tracker.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class ClusterTrack:
    cluster_id: int
    first_frame: int
    last_frame: int
    detections: int = 0
    sum_conf: float = 0.0
    sum_weight: float = 0.0
    sum_x_m: float = 0.0
    sum_y_m: float = 0.0
    max_conf: float = 0.0
    lat: float = 0.0
    lon: float = 0.0
    x_m: float = 0.0
    y_m: float = 0.0
    track_ids: Set[int] = field(default_factory=set)

    def update(
        self,
        frame: int,
        x_m: float,
        y_m: float,
        conf: float,
        track_id: Optional[int],
        origin_lat: float,
        origin_lon: float,
        meters_per_deg_lat: float,
        meters_per_deg_lon: float,
        update_position: bool,
    ) -> None:
        self.last_frame = frame
        self.detections += 1
        self.sum_conf += conf
        self.max_conf = max(self.max_conf, conf)
        weight = max(conf, 1e-3)
        if update_position or self.sum_weight == 0.0:
            self.sum_weight += weight
            self.sum_x_m += x_m * weight
            self.sum_y_m += y_m * weight
            mean_x = self.sum_x_m / self.sum_weight
            mean_y = self.sum_y_m / self.sum_weight
            self.x_m = mean_x
            self.y_m = mean_y
            if meters_per_deg_lat:
                self.lat = origin_lat + (mean_y / meters_per_deg_lat)
            else:
                self.lat = origin_lat
            if meters_per_deg_lon:
                self.lon = origin_lon + (mean_x / meters_per_deg_lon)
            else:
                self.lon = origin_lon
        if track_id is not None:
            self.track_ids.add(track_id)

    @property
    def mean_conf(self) -> float:
        return self.sum_conf / self.detections if self.detections else 0.0

    def as_summary(self) -> Dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "frames": self.detections,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "mean_conf": self.mean_conf,
            "max_conf": self.max_conf,
            "lat": self.lat,
            "lon": self.lon,
            "track_ids": sorted(self.track_ids),
        }


class SpatialClusterer:
    """
    Deduplicate detections by clustering them in a projected metric space.
    """

    def __init__(
        self,
        radius_m: float = 1.5,
        max_idle_frames: int = 45,
        min_detections: int = 1,
        freeze_after: Optional[int] = None,
        reassoc_radius_m: float = 30.0,
    ):
        self.radius_m = radius_m
        self.max_idle_frames = max_idle_frames
        self.min_detections = min_detections
        self.freeze_after = freeze_after
        self.reassoc_radius_m = reassoc_radius_m

        self._clusters: Dict[int, ClusterTrack] = {}
        self._next_cluster_id: int = 1
        self._origin: Optional[Tuple[float, float]] = None
        self._meters_per_deg_lat: Optional[float] = None
        self._meters_per_deg_lon: Optional[float] = None

    def _ensure_origin(self, lat: float, lon: float) -> None:
        if self._origin is None:
            self._origin = (lat, lon)
            self._meters_per_deg_lat = 111_320.0
            self._meters_per_deg_lon = self._meters_per_deg_lat * math.cos(math.radians(lat))

    def _to_meters(self, lat: float, lon: float) -> Tuple[float, float]:
        assert self._origin is not None and self._meters_per_deg_lat is not None and self._meters_per_deg_lon is not None
        lat0, lon0 = self._origin
        dx = (lon - lon0) * self._meters_per_deg_lon
        dy = (lat - lat0) * self._meters_per_deg_lat
        return dx, dy

    def update(
        self,
        frame: int,
        lat: Optional[float],
        lon: Optional[float],
        conf: float,
        track_id: Optional[int],
    ) -> Optional[int]:
        """
        Add a detection to the clustering pool.
        Returns the cluster id the detection was assigned to.
        """
        if lat is None or lon is None:
            return None

        self._ensure_origin(lat, lon)
        x_m, y_m = self._to_meters(lat, lon)

        if track_id is not None:
            track_match_radius = max(self.radius_m * 2.0, 12.0)
            best_track_cluster: Optional[int] = None
            best_track_dist = float("inf")
            for cluster_id, cluster in self._clusters.items():
                if frame - cluster.last_frame > self.max_idle_frames:
                    continue
                if track_id in cluster.track_ids:
                    dist = math.hypot(x_m - cluster.x_m, y_m - cluster.y_m)
                    if dist <= track_match_radius and dist < best_track_dist:
                        best_track_dist = dist
                        best_track_cluster = cluster_id
            if best_track_cluster is not None:
                cluster = self._clusters[best_track_cluster]
                cluster.update(
                    frame,
                    x_m,
                    y_m,
                    conf,
                    track_id,
                    self._origin[0],
                    self._origin[1],
                    self._meters_per_deg_lat or 1.0,
                    self._meters_per_deg_lon or 0.0,
                    self.freeze_after is None or cluster.detections < self.freeze_after,
                )
                return best_track_cluster

        best_cluster_id: Optional[int] = None
        best_dist = float("inf")
        for cluster_id, cluster in self._clusters.items():
            if frame - cluster.last_frame > self.max_idle_frames:
                continue
            dx = x_m - cluster.x_m
            dy = y_m - cluster.y_m
            dist = math.hypot(dx, dy)
            if (
                track_id is not None
                and cluster.track_ids
                and track_id not in cluster.track_ids
                and dist > (self.radius_m * 0.45)
            ):
                # Avoid collapsing nearby but different tracked animals into one cluster.
                continue
            if dist <= self.radius_m and dist < best_dist:
                best_dist = dist
                best_cluster_id = cluster_id

        if best_cluster_id is None:
            fallback_cluster_id: Optional[int] = None
            fallback_dist = float("inf")
            effective_reassoc_radius = min(
                self.reassoc_radius_m,
                max(self.radius_m * 3.0, 25.0),
            )
            for cluster_id, cluster in self._clusters.items():
                if frame - cluster.last_frame > self.max_idle_frames:
                    continue
                if (
                    track_id is not None
                    and cluster.track_ids
                    and track_id not in cluster.track_ids
                ):
                    continue
                if self.freeze_after is not None and cluster.detections >= self.freeze_after:
                    dist = math.hypot(x_m - cluster.x_m, y_m - cluster.y_m)
                    if dist <= effective_reassoc_radius and dist < fallback_dist:
                        fallback_dist = dist
                        fallback_cluster_id = cluster_id
            if fallback_cluster_id is not None:
                cluster = self._clusters[fallback_cluster_id]
                cluster.update(
                    frame,
                    x_m,
                    y_m,
                    conf,
                    track_id,
                    self._origin[0],
                    self._origin[1],
                    self._meters_per_deg_lat or 1.0,
                    self._meters_per_deg_lon or 0.0,
                    update_position=False,
                )
                return fallback_cluster_id

            best_cluster_id = self._next_cluster_id
            self._next_cluster_id += 1
            self._clusters[best_cluster_id] = ClusterTrack(
                cluster_id=best_cluster_id,
                first_frame=frame,
                last_frame=frame,
            )

        cluster = self._clusters[best_cluster_id]
        cluster.update(
            frame,
            x_m,
            y_m,
            conf,
            track_id,
            self._origin[0],
            self._origin[1],
            self._meters_per_deg_lat or 1.0,
            self._meters_per_deg_lon or 0.0,
            self.freeze_after is None or cluster.detections < self.freeze_after,
        )
        return best_cluster_id

    def summaries(self) -> List[Dict[str, object]]:
        summaries: List[Dict[str, object]] = []
        for cluster in self._clusters.values():
            if cluster.detections >= self.min_detections:
                summaries.append(cluster.as_summary())
        return summaries

    def export_points(self) -> Iterable[Dict[str, object]]:
        for cluster in self._clusters.values():
            if cluster.detections >= self.min_detections:
                yield {
                    "cluster_id": cluster.cluster_id,
                    "lat": cluster.lat,
                    "lon": cluster.lon,
                    "frames": cluster.detections,
                    "mean_conf": cluster.mean_conf,
                    "max_conf": cluster.max_conf,
                }

    def is_confirmed(self, cluster_id: Optional[int]) -> bool:
        if cluster_id is None:
            return False
        cluster = self._clusters.get(cluster_id)
        return bool(cluster and cluster.detections >= self.min_detections)

    def centroid(self, cluster_id: Optional[int]) -> Optional[Tuple[float, float]]:
        if cluster_id is None:
            return None
        cluster = self._clusters.get(cluster_id)
        if not cluster:
            return None
        return (cluster.lat, cluster.lon)
