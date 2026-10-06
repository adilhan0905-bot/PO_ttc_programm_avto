from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Typical real-world heights used only for an educational monocular-distance estimate.
OBJECT_HEIGHTS_M = {
    "person": 1.70,
    "car": 1.50,
    "truck": 3.20,
    "bus": 3.20,
    "motorcycle": 1.20,
    "bicycle": 1.20,
    "dog": 0.60,
}
DEFAULT_OBJECT_HEIGHT_M = 1.50


@dataclass
class TTCResult:
    distance_m: Optional[float]
    ttc_s: Optional[float]
    level: str


def focal_length_px(frame_width: int, hfov_deg: float) -> float:
    """Approximate focal length from horizontal field of view."""
    hfov_rad = math.radians(max(10.0, min(170.0, hfov_deg)))
    return frame_width / (2.0 * math.tan(hfov_rad / 2.0))


def estimate_distance_m(frame_width: int, bbox_height_px: float, object_type: str, hfov_deg: float) -> Optional[float]:
    if bbox_height_px <= 2:
        return None
    object_height = OBJECT_HEIGHTS_M.get(object_type, DEFAULT_OBJECT_HEIGHT_M)
    fx = focal_length_px(frame_width, hfov_deg)
    distance = object_height * fx / bbox_height_px
    return max(0.5, float(distance))


def calculate_ttc(
    distance_m: Optional[float],
    ego_speed_kmh: float,
    previous_distance_m: Optional[float] = None,
    fps: float = 30.0,
) -> Optional[float]:
    """
    Educational TTC estimate.

    Uses the vehicle speed as specified by the project requirements and,
    when history exists, also considers the observed rate of approach.
    """
    if distance_m is None:
        return None
    ego_speed_ms = max(0.0, ego_speed_kmh) / 3.6
    temporal_closing_ms = 0.0
    if previous_distance_m is not None and fps > 0:
        temporal_closing_ms = max(0.0, (previous_distance_m - distance_m) * fps)
    closing_speed_ms = max(ego_speed_ms, temporal_closing_ms)
    if closing_speed_ms <= 0.05:
        return None
    return max(0.05, distance_m / closing_speed_ms)


def danger_level(ttc_s: Optional[float], threshold_s: float) -> str:
    if ttc_s is None:
        return "SAFE"
    if ttc_s < 1.0:
        return "CRITICAL"
    if ttc_s <= threshold_s:
        return "WARNING"
    return "SAFE"
