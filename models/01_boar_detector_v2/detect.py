#!/usr/bin/env python3
"""
Unified detection entry point for YOLO Stumbrai.

Scans the provided --source (file, folder, or glob). For each media item it:
  - Identifies modality (RGB, thermal, or split) from Futural naming patterns.
  - Runs Ultralytics YOLOv11 with thermal enhancements where needed.
  - Saves annotated videos under runs/detect/<exp>/videos.
  - For videos, reuses the advanced geo/cluster pipeline from boar_utils.infer_videos,
    emitting CSV summaries and interactive maps when metadata is available.
"""

import argparse
import os
import json
from enum import Enum
from itertools import count
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "disabled")

import csv
import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from boar_utils.infer_videos import run_inference as run_video_pipeline
except ModuleNotFoundError:
    from yolo.boar_utils.infer_videos import run_inference as run_video_pipeline


VIDEO_EXTS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".ts",
    ".webm",
    ".3gp",
}


class MediaKind(str, Enum):
    RGB = "rgb"
    THERMAL = "thermal"
    SPLIT = "split"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO Wildlife media detector")
    parser.add_argument("--source", type=str, required=True, help="File, directory, or glob pattern")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference size (pixels)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="0", help="'0', '0,1', 'cpu', etc.")
    parser.add_argument("--project", type=str, default="detections", help="Output root directory")
    parser.add_argument("--name", type=str, default="data", help="Run subfolder name")
    parser.add_argument("--exist-ok", action="store_true", help="Do not increment run folder if it exists")
    parser.add_argument("--vid-stride", type=int, default=1, help="Evaluate every Nth video frame")
    parser.add_argument("--save-video", action="store_true", help="Write annotated video mp4s")
    parser.add_argument("--tracker", type=str, default=None, help="Tracker config YAML (e.g. 'bytetrack.yaml')")
    parser.add_argument(
        "--modality",
        type=str,
        default="auto",
        choices=["auto", "rgb", "thermal", "split"],
        help="Force modality for all inputs (default: auto from filename)",
    )
    parser.add_argument("--default-focal-mm", type=float, default=None, help="Fallback focal length (mm, 35mm equiv)")
    parser.add_argument("--cluster-radius-m", type=float, default=5.0, help="Clustering radius on ground plane")
    parser.add_argument("--cluster-idle-frames", type=int, default=120, help="Cluster persistence (frames)")
    parser.add_argument("--cluster-min-hits", type=int, default=3, help="Min detections to keep cluster")
    parser.add_argument("--cluster-freeze-after", type=int, default=30, help="Freeze centroid after N hits (0=off)")
    parser.add_argument("--cluster-reassoc-m", type=float, default=30.0, help="Radius to re-associate frozen clusters")
    parser.add_argument("--entity-reassoc-radius-m", type=float, default=25.0, help="Radius to re-link entity identity after tracker ID switch")
    parser.add_argument("--entity-reassoc-gap-frames", type=int, default=300, help="Max frame gap to keep entity identity")
    parser.add_argument("--entity-max-speed-mps", type=float, default=8.0, help="Maximum expected animal speed (m/s) used for dynamic entity re-association gating")
    parser.add_argument("--entity-merge-gap-frames", type=int, default=220, help="Max frame gap for post-inference entity fragment merge")
    parser.add_argument("--entity-merge-base-radius-m", type=float, default=10.0, help="Base distance threshold for fragment merge")
    parser.add_argument("--entity-merge-speed-mps", type=float, default=5.0, help="Extra merge distance allowance per second of gap")
    parser.add_argument("--min-track-frames", type=int, default=3, help="Min frames to keep a tracker trajectory")
    parser.add_argument("--min-track-conf", type=float, default=0.5, help="Min mean conf for tracks to stay")
    parser.add_argument(
        "--class-switch-margin",
        type=float,
        default=1.5,
        help="Min cumulative confidence lead required before switching track class",
    )
    parser.add_argument("--min-detection-conf", type=float, default=0.35, help="Min per-detection confidence after model output")
    parser.add_argument("--allow-no-ground", action="store_true", help="Keep detections without projected ground coordinates")
    parser.add_argument("--hfov", type=float, default=None, help="Horizontal FoV override in degrees")
    parser.add_argument("--vfov", type=float, default=None, help="Vertical FoV override in degrees")
    parser.add_argument("--ground-alt", type=float, default=0.0, help="Ground altitude baseline (meters)")
    parser.add_argument("--srt", type=str, required=True, help="Optional SRT path (single video only)")
    parser.add_argument("--image-target-short", type=int, default=0,
                        help="Downscale images so the shorter edge equals this value (0 disables)")
    parser.add_argument("--line-thickness", type=int, default=2, help="Line width for image annotations")
    parser.add_argument("--images-csv", type=str, default="",
                        help="Optional CSV path for image detections (defaults to run/images/detections.csv)")
    return parser.parse_args()


def _increment_path(base: Path, exist_ok: bool) -> Path:
    if exist_ok or not base.exists():
        return base
    for idx in count(1):
        candidate = Path(f"{base}_{idx}")
        if not candidate.exists():
            return candidate
    return base


def _read_list_file(list_path: Path) -> Sequence[Path]:
    with list_path.open("r", encoding="utf-8") as fh:
        return [
            Path(line.strip()).expanduser()
            for line in fh
            if line.strip() and not line.strip().startswith("#")
        ]


def _gather_sources(src: str) -> List[Path]:
    p = Path(src).expanduser()
    if p.is_file():
        if p.suffix.lower() == ".txt":
            entries = _read_list_file(p)
            return [
                entry
                for entry in entries
                if entry.suffix.lower() in IMAGE_EXTS or entry.suffix.lower() in VIDEO_EXTS
            ]
        if  p.suffix.lower() in VIDEO_EXTS:
            return [p]
        raise SystemExit(f"Unsupported file type: {p}")
    if p.is_dir():
        matches = sorted(
            x
            for x in p.rglob("*")
            if x.is_file() and (x.suffix.lower() in IMAGE_EXTS or x.suffix.lower() in VIDEO_EXTS)
        )
        if matches:
            return matches
        raise SystemExit(f"No media found under directory: {p}")
    matches = sorted(
        x
        for x in Path().glob(src)
        if x.is_file() and (x.suffix.lower() in IMAGE_EXTS or x.suffix.lower() in VIDEO_EXTS)
    )
    if matches:
        return matches
    raise SystemExit(f"No files matched source: {src}")


def _infer_kind(path: Path) -> MediaKind:
    stem = path.stem.lower()
    name = path.name.lower()
    if stem.endswith("_s") or stem.endswith("-s") or "split" in stem:
        return MediaKind.SPLIT
    thermal_tokens = ("_t", "-t", "_ir", "-ir", "thermal", "infrared")
    if any(token in stem for token in thermal_tokens) or any(token in name for token in ("_t.", "_ir.")):
        return MediaKind.THERMAL
    return MediaKind.RGB


def _should_use_half(device: str) -> bool:
    dev = device.lower()
    if dev.startswith("cpu") or dev == "mps":
        return False
    return torch.cuda.is_available()


def _auto_srt_for_video(video_path: Path) -> Optional[str]:
    candidates = [
        video_path.with_suffix(".srt"),
        video_path.with_suffix(".SRT"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _sanitize_name(p: Path) -> str:
    return p.stem.replace(" ", "_")

def _process_videos(
    video_paths: List[Path],
    args: argparse.Namespace,
    videos_root: Path,
) -> None:
    if not video_paths:
        return
    videos_root.mkdir(parents=True, exist_ok=True)

    explicit_srt = Path(args.srt).expanduser() if args.srt else None
    if explicit_srt and not explicit_srt.exists():
        raise SystemExit(f"SRT file not found: {explicit_srt}")

    for idx, video_path in enumerate(video_paths):
        if not video_path.exists():
            print(f"[WARN] Video not found: {video_path}")
            continue

        if args.modality != "auto":
            kind = MediaKind(args.modality)
        else:
            kind = _infer_kind(video_path)
        run_path = _increment_path(videos_root / _sanitize_name(video_path), args.exist_ok)
        run_path.parent.mkdir(parents=True, exist_ok=True)
        run_project = run_path.parent
        run_name = run_path.name

        if explicit_srt:
            srt_path = str(explicit_srt) if len(video_paths) == 1 else _auto_srt_for_video(video_path)
        else:
            srt_path = _auto_srt_for_video(video_path)

        csv_path = run_path / "detections.csv"
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        print(f"[VID] {video_path} → {run_path} (kind={kind.value}, srt={srt_path or 'none'})")
        model = os.path.join(BASE_DIR, "..", "weights", "boars.pt")
        run_video_pipeline(
            weights = model,
            source=str(video_path),
            srt=srt_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            save_video=args.save_video,
            save_csv=False,
            project=str(run_project),
            name=run_name,
            vid_stride=max(1, args.vid_stride),
            tracker=args.tracker,
            default_focal_mm=args.default_focal_mm,
            cluster_radius_m=args.cluster_radius_m,
            cluster_idle_frames=args.cluster_idle_frames,
            cluster_min_hits=args.cluster_min_hits,
            min_track_frames=args.min_track_frames,
            min_track_conf=args.min_track_conf,
            class_switch_margin=args.class_switch_margin,
            min_detection_conf=args.min_detection_conf,
            allow_no_ground=args.allow_no_ground,
            cluster_freeze_after=args.cluster_freeze_after,
            cluster_reassoc_m=args.cluster_reassoc_m,
            entity_reassoc_radius_m=args.entity_reassoc_radius_m,
            entity_reassoc_gap_frames=args.entity_reassoc_gap_frames,
            entity_max_speed_mps=args.entity_max_speed_mps,
            entity_merge_gap_frames=args.entity_merge_gap_frames,
            entity_merge_base_radius_m=args.entity_merge_base_radius_m,
            entity_merge_speed_mps=args.entity_merge_speed_mps,
            hfov=args.hfov,
            vfov=args.vfov,
            ground_alt=args.ground_alt,
            modality=kind.value,
        )
        import json


def main() -> None:
    args = parse_args()
    sources = _gather_sources(args.source)
    if not sources:
        raise SystemExit("No media found.")

    base_dir = Path(args.project) / args.name
    base_dir = _increment_path(base_dir, args.exist_ok)

    video_paths = [p for p in sources if p.suffix.lower() in VIDEO_EXTS]
    video_out = base_dir / "videos"

    _process_videos(video_paths, args, video_out)

    for video_path in video_paths:
        run_path = video_out / _sanitize_name(video_path)
        csv_file = run_path / "detections.csv"
        geojson_file = run_path / "detections.geojson"


    print(f"[DONE] Outputs saved under {base_dir.resolve()}")


if __name__ == "__main__":
    main()
