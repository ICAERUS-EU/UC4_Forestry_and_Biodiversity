"""
Lightweight DJI-style SRT parser used by infer_videos.py.

Exports:
    extract_geo_from_srt(srt_path: str) -> dict[int, dict]
        Returns a dict keyed by *frame index*:
        {
            <frame_idx>: {"lat": float, "lon": float, "alt": float|None, "utc": str|None},
            ...
        }

Notes:
- Handles common DJI SRT formats (Latitude/Longitude fields, raw "lat, lon" pairs,
  frame counters like "FrameCnt: 1234", and timecodes "hh:mm:ss,ms --> hh:mm:ss,ms").
- If a frame index is not present in a block, we assign one sequentially.
- If altitude cannot be parsed, it is set to None.
- UTC is taken from the subtitle timecode start; if not present, None.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List, Optional


# --- Regex patterns (tuned for DJI variants) ---
RE_TIMECODE   = re.compile(r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})")
RE_FRAMECNT   = re.compile(r"(?:FrameCnt|Frame|frame|frame_idx)\s*[:=]\s*(\d+)", re.IGNORECASE)

# Latitude/Longitude in named form: "Latitude: 55.123456", "Longitude = 23.123456"
RE_NAMED_LAT  = re.compile(r"(?:lat|latitude)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_NAMED_LON  = re.compile(r"(?:lon|longitude)\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)

# Raw pair somewhere in the line: "55.123456, 23.123456" or "55.123456 23.123456"
RE_PAIR_LATLON = re.compile(r"([+-]?\d+(?:\.\d+))[,;\s]+([+-]?\d+(?:\.\d+))")

# Altitude variants seen in DJI SRTs
RE_ALT        = re.compile(r"alt\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_REL_ALT    = re.compile(r"rel[_\s]?alt\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_ABS_ALT    = re.compile(r"abs[_\s]?alt\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_FOCAL      = re.compile(r"focal_len\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_DZOOM      = re.compile(r"dzoom_ratio\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_GB_YAW     = re.compile(r"gb_yaw\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_GB_PITCH   = re.compile(r"gb_pitch\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
RE_GB_ROLL    = re.compile(r"gb_roll\s*[:=]\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)

def _parse_srt_blocks(srt_path: Path) -> List[List[str]]:
    """Split an SRT into blocks (lists of non-empty lines)."""
    text = srt_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    blocks: List[List[str]] = []
    cur: List[str] = []
    for ln in lines + [""]:
        if ln.strip():
            cur.append(ln)
        else:
            if cur:
                blocks.append(cur)
                cur = []
    return blocks


def _timecode_to_utc(tc: str) -> str:
    """
    Normalize SRT timecode 'hh:mm:ss,ms' or 'hh:mm:ss.ms' to 'hh:mm:ss.ms' and return it.
    We only store the START timecode for the block as 'utc' string.
    """
    return tc.replace(",", ".")


def _safe_float(x: Optional[str]) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def extract_geo_from_srt(srt_path: str) -> List[Dict[str, Any]]:
    """
    Parse a DJI .SRT file and return:
        [
            {"frame": int, "lat": float, "lon": float, "alt": float|None, "utc": str|None},
            ...
        ]

    Frame index strategy:
    - If a block contains a numeric frame counter (FrameCnt/Frame/etc), we use that (0-based).
    - Otherwise we assign sequential indices (starting at 0) in the order blocks appear.
    """
    p = Path(srt_path)
    if not p.exists():
        raise FileNotFoundError(f"SRT not found: {p}")

    blocks = _parse_srt_blocks(p)
    out: Dict[int, Dict[str, Any]] = {}
    next_seq_frame = 0

    for blk in blocks:
        blob = " ".join(blk)

        # 1) timecode (use start as "utc")
        utc: Optional[str] = None
        m_tc = RE_TIMECODE.search(blob)
        if m_tc:
            utc = _timecode_to_utc(m_tc.group(1))

        # 2) frame index if present
        frame_idx: Optional[int] = None
        m_frame = RE_FRAMECNT.search(blob)
        if m_frame:
            try:
                # DJI FrameCnt is typically 1-based; we normalize to 0-based
                frame_idx = max(0, int(m_frame.group(1)) - 1)
            except Exception:
                frame_idx = None

        # 3) latitude/longitude
        lat: Optional[float] = None
        lon: Optional[float] = None

        # Try named fields first
        m_lat = RE_NAMED_LAT.search(blob)
        m_lon = RE_NAMED_LON.search(blob)
        if m_lat and m_lon:
            lat = _safe_float(m_lat.group(1))
            lon = _safe_float(m_lon.group(1))
        else:
            # Fallback: generic "pair somewhere" pattern
            m_pair = RE_PAIR_LATLON.search(blob)
            if m_pair:
                lat = _safe_float(m_pair.group(1))
                lon = _safe_float(m_pair.group(2))

        # 4) altitude (optional)
        rel_alt: Optional[float] = None
        abs_alt: Optional[float] = None
        m_rel = RE_REL_ALT.search(blob)
        if m_rel:
            rel_alt = _safe_float(m_rel.group(1))
        m_abs = RE_ABS_ALT.search(blob)
        if m_abs:
            abs_alt = _safe_float(m_abs.group(1))
        if rel_alt is None and abs_alt is None:
            m_alt = RE_ALT.search(blob)
            if m_alt:
                rel_alt = _safe_float(m_alt.group(1))

        focal_len = None
        m_focal = RE_FOCAL.search(blob)
        if m_focal:
            focal_len = _safe_float(m_focal.group(1))

        dzoom = None
        m_zoom = RE_DZOOM.search(blob)
        if m_zoom:
            dzoom = _safe_float(m_zoom.group(1))

        gb_yaw = None
        m_yaw = RE_GB_YAW.search(blob)
        if m_yaw:
            gb_yaw = _safe_float(m_yaw.group(1))

        gb_pitch = None
        m_pitch = RE_GB_PITCH.search(blob)
        if m_pitch:
            gb_pitch = _safe_float(m_pitch.group(1))

        gb_roll = None
        m_roll = RE_GB_ROLL.search(blob)
        if m_roll:
            gb_roll = _safe_float(m_roll.group(1))

        # If we got valid coordinates, store the record
        if lat is not None and lon is not None:
            if frame_idx is None:
                frame_idx = next_seq_frame
                next_seq_frame += 1

            out[int(frame_idx)] = {
                "lat": float(lat),
                "lon": float(lon),
                "alt": float(rel_alt) if rel_alt is not None else (float(abs_alt) if abs_alt is not None else None),
                "utc": utc,
                "rel_alt": float(rel_alt) if rel_alt is not None else None,
                "abs_alt": float(abs_alt) if abs_alt is not None else None,
                "focal_len": focal_len,
                "dzoom_ratio": dzoom,
                "gb_yaw": gb_yaw,
                "gb_pitch": gb_pitch,
                "gb_roll": gb_roll,
            }

    return [{"frame": idx, **rec} for idx, rec in sorted(out.items())]
