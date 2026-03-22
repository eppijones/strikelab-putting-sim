"""
Weighted fusion of initial ball speed / direction from Arducam + ZED + RealSense.

RealSense side-view speed is a different geometric projection; we use it with a
lower default weight and mainly for cross-check / inconsistency flags.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpeedDirectionFusion:
    """Result of fusing speed/direction from multiple sensors."""
    speed_m_s: float
    direction_deg: float
    speed_sources_used: List[str]
    sensor_inconsistency: bool
    inconsistency_detail: Optional[str] = None
    # Optional raw values for diagnostics
    arducam_speed_m_s: float = 0.0
    zed_speed_m_s: Optional[float] = None
    realsense_speed_m_s: Optional[float] = None


def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest difference between two angles in degrees."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def fuse_speed_and_direction(
    arducam_speed_m_s: float,
    arducam_direction_deg: float,
    zed_speed_m_s: Optional[float] = None,
    zed_direction_deg: Optional[float] = None,
    realsense_speed_m_s: Optional[float] = None,
    weight_arducam: float = 0.5,
    weight_zed: float = 0.35,
    weight_realsense: float = 0.15,
    speed_inconsistency_threshold_m_s: float = 0.45,
    direction_inconsistency_threshold_deg: float = 12.0,
) -> SpeedDirectionFusion:
    """
    Fuse speed using normalized weights over available sensors.
    Direction: circular mean of Arducam + ZED when both present; else primary.
    """
    sources: List[str] = ["arducam"]
    w_a, w_z, w_r = weight_arducam, weight_zed, weight_realsense

    has_zed = zed_speed_m_s is not None and zed_speed_m_s > 0.05
    has_rs = realsense_speed_m_s is not None and realsense_speed_m_s > 0.05

    inconsistency = False
    detail: Optional[str] = None

    if has_zed and abs(arducam_speed_m_s - zed_speed_m_s) > speed_inconsistency_threshold_m_s:
        inconsistency = True
        detail = (
            f"speed_arducam_vs_zed: {arducam_speed_m_s:.3f} vs {zed_speed_m_s:.3f} m/s"
        )

    if has_zed and zed_direction_deg is not None:
        if _angle_diff_deg(arducam_direction_deg, zed_direction_deg) > direction_inconsistency_threshold_deg:
            inconsistency = True
            extra = (
                f"dir_arducam_vs_zed: {arducam_direction_deg:.1f} vs {zed_direction_deg:.1f} deg"
            )
            detail = f"{detail}; {extra}" if detail else extra

    # Normalize weights
    w_sum = w_a
    if has_zed:
        w_sum += w_z
    if has_rs:
        w_sum += w_r
    if w_sum <= 0:
        w_sum = 1.0

    fused_speed = (w_a / w_sum) * arducam_speed_m_s
    if has_zed:
        fused_speed += (w_z / w_sum) * zed_speed_m_s
        sources.append("zed")
    if has_rs:
        fused_speed += (w_r / w_sum) * realsense_speed_m_s
        sources.append("realsense")

    # Direction: vector sum for available horizontal-plane directions
    if has_zed and zed_direction_deg is not None:
        ar = np.radians(arducam_direction_deg)
        zr = np.radians(zed_direction_deg)
        # Weighted vector average
        wx_a, wy_a = w_a * np.cos(ar), w_a * np.sin(ar)
        wx_z, wy_z = w_z * np.cos(zr), w_z * np.sin(zr)
        sx, sy = wx_a + wx_z, wy_a + wy_z
        fused_dir = float(np.degrees(np.arctan2(sy, sx)))
        if fused_dir > 180:
            fused_dir -= 360
        elif fused_dir < -180:
            fused_dir += 360
    else:
        fused_dir = arducam_direction_deg

    if inconsistency:
        logger.info("Sensor inconsistency flagged: %s", detail)

    return SpeedDirectionFusion(
        speed_m_s=fused_speed,
        direction_deg=fused_dir,
        speed_sources_used=sources,
        sensor_inconsistency=inconsistency,
        inconsistency_detail=detail,
        arducam_speed_m_s=arducam_speed_m_s,
        zed_speed_m_s=zed_speed_m_s if has_zed else None,
        realsense_speed_m_s=realsense_speed_m_s if has_rs else None,
    )


def classify_fusion_quality(
    has_arducam: bool,
    has_zed_ball: bool,
    has_zed_club: bool,
    has_realsense_launch: bool,
) -> Tuple[str, List[str]]:
    """
    Return (quality, missing_streams) where quality is full | partial | minimal.

    "full" = Arducam + ZED contributed (ball and/or club) + RealSense launch data.
    """
    missing: List[str] = []
    zed_ok = has_zed_ball or has_zed_club
    if not zed_ok:
        missing.append("zed")
    if not has_realsense_launch:
        missing.append("realsense")

    if has_arducam and zed_ok and has_realsense_launch:
        return "full", []
    if has_arducam and (zed_ok or has_realsense_launch):
        return "partial", missing
    return "minimal", missing
