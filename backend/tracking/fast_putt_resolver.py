"""
Fast putt resolver - uses ZED and RealSense data to get actual ball speed
when the Arducam loses the ball during a fast putt.

Problem: The Arducam's HSV + circularity detector fails on fast putts because
motion blur breaks the ball's circular shape. The tracker falls back to a
hardcoded 1500 px/s speed estimate, which is inaccurate.

Solution: When a "ball vanished" event is detected by the Arducam tracker,
query the ZED 2i and RealSense for actual ball speed measured at the same
moment. Both cameras are less affected by this issue:
- ZED 2i: Stereo depth matching works differently from HSV thresholding
- RealSense: Side view keeps the ball in frame longer for more measurements
"""

import numpy as np
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ResolvedSpeed:
    """Resolved ball speed from multi-camera data."""
    speed_px_s: float               # Speed in Arducam pixels/s
    speed_m_s: float                # Speed in m/s
    direction_deg: float            # Direction in degrees
    source: str                     # "zed", "realsense", "fused", "estimated"
    confidence: float               # 0-1 how confident we are
    vx_px_s: float = 0.0           # Velocity X in Arducam pixels/s
    vy_px_s: float = 0.0           # Velocity Y in Arducam pixels/s


class FastPuttResolver:
    """
    Resolves actual ball speed for fast putts using multi-camera data.

    When the Arducam reports a fast putt (ball vanished from ARMED state),
    this module checks the ZED and RealSense for measured speed data and
    returns the most accurate estimate available.
    """

    def __init__(self, pixels_per_meter: float = 1150.0, forward_direction_deg: float = 0.0):
        self._ppm = pixels_per_meter
        self._forward_dir = forward_direction_deg

        # Recent speed measurements from depth cameras
        self._zed_speeds: deque[Tuple[float, float, int]] = deque(maxlen=30)
        # (speed_m_s, direction_deg, timestamp_ns)
        self._rs_speeds: deque[Tuple[float, float, int]] = deque(maxlen=30)

    def feed_zed_speed(self, speed_m_s: float, direction_deg: float, timestamp_ns: int) -> None:
        """Feed a speed measurement from ZED 3D ball tracker."""
        if speed_m_s > 0.05:
            self._zed_speeds.append((speed_m_s, direction_deg, timestamp_ns))

    def feed_realsense_speed(self, speed_m_s: float, direction_deg: float, timestamp_ns: int) -> None:
        """Feed a speed measurement from RealSense launch detector."""
        if speed_m_s > 0.05:
            self._rs_speeds.append((speed_m_s, direction_deg, timestamp_ns))

    def resolve(self, event_timestamp_ns: int, tolerance_ns: int = 100_000_000) -> ResolvedSpeed:
        """
        Resolve ball speed at the time of a fast putt event.

        Args:
            event_timestamp_ns: When the Arducam detected ball vanished
            tolerance_ns: Max time difference to accept measurements (default 100ms)

        Returns:
            ResolvedSpeed with the best available estimate
        """
        zed_match = self._find_best_match(self._zed_speeds, event_timestamp_ns, tolerance_ns)
        rs_match = self._find_best_match(self._rs_speeds, event_timestamp_ns, tolerance_ns)

        if zed_match and rs_match:
            return self._fuse_measurements(zed_match, rs_match)
        elif zed_match:
            return self._from_single("zed", zed_match)
        elif rs_match:
            return self._from_single("realsense", rs_match)
        else:
            return self._estimated_fallback()

    def _find_best_match(
        self, buffer: deque, target_ns: int, tolerance_ns: int
    ) -> Optional[Tuple[float, float, int]]:
        """Find the measurement closest to target time within tolerance."""
        best = None
        best_dt = float("inf")
        for entry in buffer:
            dt = abs(entry[2] - target_ns)
            if dt < best_dt and dt <= tolerance_ns:
                best_dt = dt
                best = entry
        return best

    def _fuse_measurements(
        self,
        zed: Tuple[float, float, int],
        rs: Tuple[float, float, int],
    ) -> ResolvedSpeed:
        """Fuse ZED and RealSense speed measurements (weighted average)."""
        # Weight ZED more heavily - it has better 3D accuracy
        zed_weight = 0.65
        rs_weight = 0.35

        speed_m_s = zed[0] * zed_weight + rs[0] * rs_weight
        direction = zed[1] * zed_weight + rs[1] * rs_weight

        speed_px_s = speed_m_s * self._ppm
        dir_rad = np.radians(direction)
        vx = speed_px_s * np.cos(dir_rad)
        vy = speed_px_s * np.sin(dir_rad)

        logger.info(
            f"Fast putt resolved (fused): {speed_m_s:.2f} m/s "
            f"({speed_px_s:.0f} px/s) @ {direction:.1f}° "
            f"[ZED={zed[0]:.2f}, RS={rs[0]:.2f}]"
        )

        return ResolvedSpeed(
            speed_px_s=speed_px_s,
            speed_m_s=speed_m_s,
            direction_deg=direction,
            source="fused",
            confidence=0.9,
            vx_px_s=vx,
            vy_px_s=vy,
        )

    def _from_single(self, source: str, data: Tuple[float, float, int]) -> ResolvedSpeed:
        """Create resolved speed from a single camera source."""
        speed_m_s = data[0]
        direction = data[1]
        speed_px_s = speed_m_s * self._ppm
        dir_rad = np.radians(direction)
        vx = speed_px_s * np.cos(dir_rad)
        vy = speed_px_s * np.sin(dir_rad)

        logger.info(
            f"Fast putt resolved ({source}): {speed_m_s:.2f} m/s "
            f"({speed_px_s:.0f} px/s) @ {direction:.1f}°"
        )

        return ResolvedSpeed(
            speed_px_s=speed_px_s,
            speed_m_s=speed_m_s,
            direction_deg=direction,
            source=source,
            confidence=0.7,
            vx_px_s=vx,
            vy_px_s=vy,
        )

    def _estimated_fallback(self) -> ResolvedSpeed:
        """
        Fallback estimate when no depth camera data is available.
        Uses the forward direction and a heuristic speed estimate based on
        how quickly the ball disappeared.
        """
        speed_m_s = 2.0  # Conservative estimate for a fast putt
        speed_px_s = speed_m_s * self._ppm
        direction = self._forward_dir
        dir_rad = np.radians(direction)
        vx = speed_px_s * np.cos(dir_rad)
        vy = speed_px_s * np.sin(dir_rad)

        logger.warning(
            f"Fast putt estimated (no depth data): {speed_m_s:.2f} m/s "
            f"({speed_px_s:.0f} px/s) @ {direction:.1f}°"
        )

        return ResolvedSpeed(
            speed_px_s=speed_px_s,
            speed_m_s=speed_m_s,
            direction_deg=direction,
            source="estimated",
            confidence=0.3,
            vx_px_s=vx,
            vy_px_s=vy,
        )

    def set_calibration(self, pixels_per_meter: float, forward_direction_deg: float) -> None:
        self._ppm = pixels_per_meter
        self._forward_dir = forward_direction_deg
