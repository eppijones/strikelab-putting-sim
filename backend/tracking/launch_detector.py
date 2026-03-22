"""
Launch angle detector using Intel RealSense D455 at ground level.

The D455 is mounted at ground level looking straight at the ball from the side.
This gives a perfect vantage point for measuring:
- Ball launch angle (vertical angle at which ball leaves the surface)
- Ball speed from the side view
- Whether a shot is a putt (0° launch) or chip (5-60° launch)

Ball moves right to left in the image.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class ShotType(Enum):
    UNKNOWN = "unknown"
    PUTT = "putt"
    CHIP = "chip"


@dataclass
class LaunchData:
    """Launch angle and trajectory data from side-view camera."""
    launch_angle_deg: float = 0.0       # Vertical launch angle
    ball_speed_m_s: float = 0.0         # Ball speed from side view
    shot_type: ShotType = ShotType.UNKNOWN
    ball_positions_px: List[Tuple[int, int]] = field(default_factory=list)
    ball_heights_m: List[float] = field(default_factory=list)
    peak_height_m: float = 0.0
    confidence: float = 0.0
    # Timestamp (ns) of the last sample used to compute this launch (for multi-camera sync)
    timestamp_ns: int = 0

    def to_dict(self) -> dict:
        return {
            "launch_angle_deg": round(self.launch_angle_deg, 1),
            "ball_speed_m_s": round(self.ball_speed_m_s, 2),
            "shot_type": self.shot_type.value,
            "peak_height_m": round(self.peak_height_m, 3),
            "confidence": round(self.confidence, 2),
            "timestamp_ns": int(self.timestamp_ns),
        }


class LaunchDetector:
    """
    Detects ball launch from the RealSense D455 ground-level side view.

    Detection strategy:
    1. Track the ball's vertical position (y pixel) over the first 5-10 frames after impact
    2. If y decreases significantly (ball rises in image), compute launch angle from
       the vertical vs horizontal displacement
    3. Use depth to convert pixel displacements to real-world measurements
    4. Classify as putt (<3°) or chip (>3°)
    """

    BALL_HSV_LOWER = np.array([0, 0, 150])
    BALL_HSV_UPPER = np.array([180, 80, 255])
    MIN_BALL_AREA = 30
    MAX_BALL_AREA = 3000
    MIN_CIRCULARITY = 0.5  # More lenient - side view may not be perfectly round
    PUTT_ANGLE_THRESHOLD = 3.0  # degrees: below = putt, above = chip

    # Ball tracking
    MAX_TRACKING_FRAMES = 30   # Track ball for up to 30 frames after first detection
    MIN_FRAMES_FOR_ANGLE = 5   # Need at least 5 frames to compute angle (was 3, too noisy)
    MOTION_THRESHOLD_PX = 15   # Min horizontal movement to consider ball in motion (was 5)
    LAUNCH_COOLDOWN_FRAMES = 60  # Min frames between launch computations (~1s at 60fps)
    MIN_RESTING_FRAMES = 10     # Ball must rest for this many frames before tracking starts

    def __init__(self):
        self._ball_positions: deque[Tuple[int, int, int, float]] = deque(maxlen=60)
        # (px_x, px_y, frame_id, timestamp_ns)
        self._tracking_active = False
        self._tracking_start_frame = 0
        self._resting_position: Optional[Tuple[int, int]] = None
        self._resting_depth_m: float = 0.0
        self._resting_frames: int = 0
        self._launch_data: Optional[LaunchData] = None
        self._frame_count = 0
        self._depth_at_ball: float = 1.0  # meters from RS to ball
        self._last_launch_frame: int = 0
        self._prev_ball_in_motion: bool = False

    def update(self, color: np.ndarray, depth: np.ndarray,
               timestamp_ns: int, ball_in_motion: bool = False) -> Optional[LaunchData]:
        """
        Process a RealSense frame.

        Args:
            color: BGR color image (aligned to depth)
            depth: float32 depth map in meters
            timestamp_ns: Frame timestamp
            ball_in_motion: Whether the Arducam tracker says ball is moving

        Returns:
            LaunchData when launch angle computation is complete, else None.
        """
        self._frame_count += 1

        # Enforce cooldown between launch computations
        if (self._frame_count - self._last_launch_frame) < self.LAUNCH_COOLDOWN_FRAMES:
            if not self._tracking_active:
                self._prev_ball_in_motion = ball_in_motion
                return None

        ball_2d = self._detect_ball(color)
        if ball_2d is None:
            self._resting_frames = 0
            if self._tracking_active:
                # Lost ball during tracking - finalize
                if len(self._ball_positions) >= self.MIN_FRAMES_FOR_ANGLE:
                    result = self._compute_launch()
                    self._prev_ball_in_motion = ball_in_motion
                    return result
                self._tracking_active = False
            self._prev_ball_in_motion = ball_in_motion
            return None

        px_x, px_y, radius = ball_2d

        # Get depth at ball position
        h, w = depth.shape[:2]
        if 0 <= px_x < w and 0 <= px_y < h:
            d = depth[px_y, px_x]
            if np.isfinite(d) and d > 0.1:
                self._depth_at_ball = float(d)

        if not self._tracking_active:
            if not ball_in_motion:
                self._resting_frames += 1
                self._resting_position = (px_x, px_y)
                self._resting_depth_m = self._depth_at_ball
            elif (ball_in_motion and not self._prev_ball_in_motion and
                  self._resting_position is not None and
                  self._resting_frames >= self.MIN_RESTING_FRAMES):
                # Only start tracking on a fresh ARMED->TRACKING transition
                self._tracking_active = True
                self._tracking_start_frame = self._frame_count
                self._ball_positions.clear()
                self._ball_positions.append((px_x, px_y, self._frame_count, timestamp_ns))
                logger.debug(f"Launch tracking started at frame {self._frame_count}")
        else:
            self._ball_positions.append((px_x, px_y, self._frame_count, timestamp_ns))

            frames_tracked = self._frame_count - self._tracking_start_frame
            if frames_tracked >= self.MAX_TRACKING_FRAMES:
                self._tracking_active = False
                if len(self._ball_positions) >= self.MIN_FRAMES_FOR_ANGLE:
                    result = self._compute_launch()
                    self._prev_ball_in_motion = ball_in_motion
                    return result

        self._prev_ball_in_motion = ball_in_motion
        return None

    def _detect_ball(self, color: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """Detect ball in the side-view color image."""
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.BALL_HSV_LOWER, self.BALL_HSV_UPPER)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_BALL_AREA or area > self.MAX_BALL_AREA:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circ = 4 * np.pi * area / (perimeter ** 2)
            if circ < self.MIN_CIRCULARITY:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            score = circ * area
            if score > best_score:
                best_score = score
                best = (int(cx), int(cy), float(radius))

        return best

    def _compute_launch(self) -> Optional[LaunchData]:
        """Compute launch angle from tracked positions."""
        self._last_launch_frame = self._frame_count
        positions = list(self._ball_positions)
        data = LaunchData()
        data.ball_positions_px = [(p[0], p[1]) for p in positions]

        if len(positions) < self.MIN_FRAMES_FOR_ANGLE:
            data.confidence = 0.0
            data.timestamp_ns = int(positions[-1][3]) if positions else 0
            return None

        # Use first few positions for launch angle (before gravity bends the path)
        n = min(8, len(positions))
        early = positions[:n]

        # Horizontal displacement (ball moves right to left, so dx is negative)
        dx_px = early[-1][0] - early[0][0]
        # Vertical displacement (positive dy_px = ball moving down in image, but up in world)
        dy_px = early[0][1] - early[-1][1]  # Inverted: lower y-pixel = higher in world

        if abs(dx_px) < self.MOTION_THRESHOLD_PX:
            return None

        # Launch angle from pixel displacement
        # Using depth to convert to real-world scale if available
        launch_angle = np.degrees(np.arctan2(dy_px, abs(dx_px)))
        data.launch_angle_deg = max(0.0, launch_angle)

        # Ball speed from displacement and timing
        dt = (positions[-1][3] - positions[0][3]) / 1e9
        if dt > 0:
            # Approximate real-world distance using depth
            scale = self._depth_at_ball / 500.0  # rough px-to-m at this depth
            total_px = np.sqrt(dx_px ** 2 + dy_px ** 2)
            data.ball_speed_m_s = (total_px * scale) / dt

        # Compute heights from y-pixel positions
        if self._resting_position:
            rest_y = self._resting_position[1]
            scale_y = self._depth_at_ball / 500.0
            for p in positions:
                h = max(0.0, (rest_y - p[1]) * scale_y)
                data.ball_heights_m.append(h)
            if data.ball_heights_m:
                data.peak_height_m = max(data.ball_heights_m)

        # Classify shot type
        if data.launch_angle_deg < self.PUTT_ANGLE_THRESHOLD:
            data.shot_type = ShotType.PUTT
        else:
            data.shot_type = ShotType.CHIP

        data.confidence = min(1.0, len(positions) / 10.0)

        # Last tracked frame timestamp for fusion / FastPuttResolver alignment
        data.timestamp_ns = int(positions[-1][3])

        logger.info(
            f"Launch computed: angle={data.launch_angle_deg:.1f}°, "
            f"type={data.shot_type.value}, speed={data.ball_speed_m_s:.2f} m/s, "
            f"peak={data.peak_height_m:.3f}m"
        )

        return data

    def reset(self) -> None:
        """Reset for a new shot."""
        self._ball_positions.clear()
        self._tracking_active = False
        self._launch_data = None
        self._resting_position = None
        self._resting_frames = 0
        self._prev_ball_in_motion = False

    def get_latest(self) -> Optional[LaunchData]:
        return self._launch_data
