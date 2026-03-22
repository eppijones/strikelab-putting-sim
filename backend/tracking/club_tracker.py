"""
Club head tracker using ZED 2i depth data.

Detects and tracks the putter/wedge head through the impact zone using
depth segmentation and motion analysis from the ZED stereo camera.

Provides TrackMan-style metrics:
- Club path angle (direction the club head is traveling at impact)
- Face angle estimate (orientation of the club face at impact)
- Attack angle (vertical angle of approach)
- Stroke tempo (backswing time vs forward swing time)
- Impact speed
- Backswing length and forward swing length

Ball moves right to left. Putter approaches from the right side.
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class StrokePhase(Enum):
    IDLE = "idle"
    BACKSWING = "backswing"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"
    COMPLETE = "complete"


@dataclass
class ClubPosition:
    """Club head position in 3D."""
    x: float          # meters
    y: float          # meters (height)
    z: float          # meters (depth)
    px_x: int         # pixel x
    px_y: int         # pixel y
    timestamp_ns: int
    area_px: float = 0.0


@dataclass
class ClubMetrics:
    """TrackMan-style club metrics computed after a stroke."""
    club_path_deg: float = 0.0          # Horizontal path angle at impact (+ = in-to-out, - = out-to-in)
    face_angle_deg: float = 0.0         # Estimated face angle relative to target line
    attack_angle_deg: float = 0.0       # Vertical attack angle (- = descending)
    club_speed_m_s: float = 0.0         # Club head speed at impact
    impact_point: Optional[Tuple[float, float]] = None  # Relative impact point on face
    stroke_tempo: float = 0.0           # Backswing time / forward swing time ratio
    backswing_length_m: float = 0.0     # Backswing arc length
    forward_swing_length_m: float = 0.0 # Forward swing arc length
    backswing_time_ms: float = 0.0
    forward_swing_time_ms: float = 0.0
    stroke_phase: StrokePhase = StrokePhase.IDLE

    def to_dict(self) -> dict:
        return {
            "club_path_deg": round(self.club_path_deg, 1),
            "face_angle_deg": round(self.face_angle_deg, 1),
            "attack_angle_deg": round(self.attack_angle_deg, 1),
            "club_speed_m_s": round(self.club_speed_m_s, 2),
            "impact_point": self.impact_point,
            "stroke_tempo": round(self.stroke_tempo, 2),
            "backswing_length_m": round(self.backswing_length_m, 3),
            "forward_swing_length_m": round(self.forward_swing_length_m, 3),
            "backswing_time_ms": round(self.backswing_time_ms, 0),
            "forward_swing_time_ms": round(self.forward_swing_time_ms, 0),
            "stroke_phase": self.stroke_phase.value,
        }


@dataclass
class ClubPath:
    """Full club path through the stroke for visualization."""
    positions: List[ClubPosition] = field(default_factory=list)
    path_3d: List[Tuple[float, float, float]] = field(default_factory=list)
    impact_index: int = -1


class ClubTracker:
    """
    Tracks the club head using ZED 2i depth segmentation.

    Strategy:
    1. When ball is stationary (ARMED), identify the "near-ball" region
    2. Use depth to segment objects at club-head height (above ball, below hand)
    3. Track the club head as the moving object closest to the ball
    4. Detect impact as the moment the club reaches the ball position
    5. Compute stroke metrics from the recorded path

    Club head detection uses depth layering:
    - Surface (green): ~0.40-0.50m from camera
    - Ball: ~0.02m above surface
    - Club head: ~0.02-0.05m above surface (putter sole to top)
    - Hands/wrists: ~0.15-0.30m above surface
    """

    CLUB_HEIGHT_MIN_M = 0.005     # Club head is at least 5mm above surface
    CLUB_HEIGHT_MAX_M = 0.08      # Club head at most 8cm above surface
    CLUB_MIN_AREA = 100           # Minimum contour area in pixels
    CLUB_MAX_AREA = 8000          # Maximum contour area
    MOTION_THRESHOLD_M = 0.005    # 5mm movement to consider "moving"
    IMPACT_PROXIMITY_M = 0.04     # Within 4cm of ball = potential impact
    VELOCITY_WINDOW = 4           # Frames for velocity computation

    # Stroke detection
    BACKSWING_THRESHOLD_M_S = 0.15    # Minimum speed to start backswing detection
    IMPACT_SPEED_THRESHOLD_M_S = 0.3  # Must be at least this fast at impact

    def __init__(self, surface_depth_m: float = 0.45):
        self._surface_depth_m = surface_depth_m
        self._positions: deque[ClubPosition] = deque(maxlen=300)  # ~5s at 60fps
        self._stroke_phase = StrokePhase.IDLE
        self._ball_position_2d: Optional[Tuple[int, int]] = None
        self._ball_position_3d: Optional[Tuple[float, float, float]] = None

        # Stroke recording
        self._stroke_path = ClubPath()
        self._backswing_start_ns: int = 0
        self._downswing_start_ns: int = 0
        self._impact_ns: int = 0
        self._backswing_positions: List[ClubPosition] = []
        self._downswing_positions: List[ClubPosition] = []

        self._metrics: Optional[ClubMetrics] = None
        self._prev_frame_gray: Optional[np.ndarray] = None

    def set_ball_position(self, px_x: int, px_y: int,
                          pos_3d: Optional[Tuple[float, float, float]] = None) -> None:
        """Set ball position so we can detect impact."""
        self._ball_position_2d = (px_x, px_y)
        self._ball_position_3d = pos_3d

    def update(self, color: np.ndarray, depth: np.ndarray,
               point_cloud: Optional[np.ndarray], timestamp_ns: int,
               ball_in_motion: bool = False) -> Optional[ClubMetrics]:
        """
        Process a ZED frame to detect/track the club head.
        Returns ClubMetrics when a complete stroke is detected.
        """
        club_pos = self._detect_club_head(color, depth, point_cloud, timestamp_ns)
        if club_pos is None:
            return None

        self._positions.append(club_pos)

        # Update stroke phase state machine
        completed_metrics = self._update_stroke_phase(club_pos, timestamp_ns, ball_in_motion)

        return completed_metrics

    def _detect_club_head(self, color: np.ndarray, depth: np.ndarray,
                          point_cloud: Optional[np.ndarray],
                          timestamp_ns: int) -> Optional[ClubPosition]:
        """Detect the club head using depth segmentation."""
        if depth is None:
            return None

        h, w = depth.shape[:2]

        # Depth layer for club head height
        surface = self._surface_depth_m
        club_min_depth = surface - self.CLUB_HEIGHT_MAX_M
        club_max_depth = surface - self.CLUB_HEIGHT_MIN_M

        # Create mask of pixels in the club height range
        valid = np.isfinite(depth) & (depth > 0.1)
        club_mask = valid & (depth >= club_min_depth) & (depth <= club_max_depth)
        club_mask = club_mask.astype(np.uint8) * 255

        # Focus near the ball if we know where it is
        if self._ball_position_2d is not None:
            bx, by = self._ball_position_2d
            search_radius = 200  # pixels
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(roi_mask, (bx, by), search_radius, 255, -1)
            club_mask = cv2.bitwise_and(club_mask, roi_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        club_mask = cv2.morphologyEx(club_mask, cv2.MORPH_OPEN, kernel)
        club_mask = cv2.morphologyEx(club_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(club_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find the best club head candidate
        best_pos = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.CLUB_MIN_AREA or area > self.CLUB_MAX_AREA:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            if cx < 0 or cx >= w or cy < 0 or cy >= h:
                continue

            d = depth[cy, cx]
            if not np.isfinite(d) or d < 0.1:
                continue

            # Score: prefer contours near the ball and at the right depth
            score = area / self.CLUB_MAX_AREA
            if self._ball_position_2d:
                dist = np.sqrt((cx - self._ball_position_2d[0]) ** 2 +
                               (cy - self._ball_position_2d[1]) ** 2)
                proximity_bonus = max(0, 1.0 - dist / 300.0)
                score *= (1.0 + proximity_bonus)

            if score > best_score:
                best_score = score
                x_m, y_m, z_m = 0.0, 0.0, float(d)
                if point_cloud is not None and cx < point_cloud.shape[1] and cy < point_cloud.shape[0]:
                    pt = point_cloud[cy, cx]
                    if np.isfinite(pt[0]):
                        x_m, y_m, z_m = float(pt[0]), float(pt[1]), float(pt[2])

                best_pos = ClubPosition(
                    x=x_m, y=y_m, z=z_m,
                    px_x=cx, px_y=cy,
                    timestamp_ns=timestamp_ns,
                    area_px=area,
                )

        return best_pos

    def _update_stroke_phase(self, pos: ClubPosition, timestamp_ns: int,
                             ball_in_motion: bool) -> Optional[ClubMetrics]:
        """State machine for stroke phase detection."""
        vel = self._compute_club_velocity()
        if vel is None:
            return None

        speed = np.sqrt(sum(v ** 2 for v in vel))

        if self._stroke_phase == StrokePhase.IDLE:
            if speed > self.BACKSWING_THRESHOLD_M_S and not ball_in_motion:
                # Club is moving - check direction to determine if backswing
                # Backswing moves club away from target (right, for R-to-L ball direction)
                if vel[0] > 0:  # Moving right = backswing
                    self._stroke_phase = StrokePhase.BACKSWING
                    self._backswing_start_ns = timestamp_ns
                    self._backswing_positions = [pos]
                    self._stroke_path = ClubPath()
                    self._stroke_path.positions.append(pos)
                    logger.debug(f"Backswing detected: speed={speed:.2f} m/s")

        elif self._stroke_phase == StrokePhase.BACKSWING:
            self._backswing_positions.append(pos)
            self._stroke_path.positions.append(pos)
            # Transition to downswing when direction reverses (club starts moving left)
            if vel[0] < -self.BACKSWING_THRESHOLD_M_S:
                self._stroke_phase = StrokePhase.DOWNSWING
                self._downswing_start_ns = timestamp_ns
                self._downswing_positions = [pos]
                logger.debug(f"Downswing started after backswing of "
                             f"{(timestamp_ns - self._backswing_start_ns) / 1e6:.0f}ms")

        elif self._stroke_phase == StrokePhase.DOWNSWING:
            self._downswing_positions.append(pos)
            self._stroke_path.positions.append(pos)
            # Check for impact
            if self._ball_position_3d is not None:
                dist_to_ball = np.sqrt(
                    (pos.x - self._ball_position_3d[0]) ** 2 +
                    (pos.z - self._ball_position_3d[2]) ** 2
                )
                if dist_to_ball < self.IMPACT_PROXIMITY_M:
                    self._stroke_phase = StrokePhase.IMPACT
                    self._impact_ns = timestamp_ns
                    self._stroke_path.impact_index = len(self._stroke_path.positions) - 1
                    logger.info(f"Impact detected: speed={speed:.2f} m/s, "
                                f"dist_to_ball={dist_to_ball:.3f}m")

            elif ball_in_motion:
                # Ball started moving = impact happened
                self._stroke_phase = StrokePhase.IMPACT
                self._impact_ns = timestamp_ns
                self._stroke_path.impact_index = len(self._stroke_path.positions) - 1

        elif self._stroke_phase == StrokePhase.IMPACT:
            self._stroke_path.positions.append(pos)
            self._stroke_phase = StrokePhase.FOLLOW_THROUGH

        elif self._stroke_phase == StrokePhase.FOLLOW_THROUGH:
            self._stroke_path.positions.append(pos)
            if speed < self.BACKSWING_THRESHOLD_M_S:
                self._stroke_phase = StrokePhase.COMPLETE
                metrics = self._compute_metrics()
                self._metrics = metrics
                self._stroke_phase = StrokePhase.IDLE
                logger.info(f"Stroke complete: path={metrics.club_path_deg:.1f}°, "
                            f"speed={metrics.club_speed_m_s:.2f} m/s, "
                            f"tempo={metrics.stroke_tempo:.2f}")
                return metrics

        return None

    def _compute_club_velocity(self) -> Optional[Tuple[float, float, float]]:
        """Compute club head velocity from recent positions."""
        if len(self._positions) < 2:
            return None

        window = min(self.VELOCITY_WINDOW, len(self._positions))
        positions = list(self._positions)[-window:]
        p1, p2 = positions[0], positions[-1]

        dt = (p2.timestamp_ns - p1.timestamp_ns) / 1e9
        if dt <= 0:
            return None

        return (
            (p2.x - p1.x) / dt,
            (p2.y - p1.y) / dt,
            (p2.z - p1.z) / dt,
        )

    def _compute_metrics(self) -> ClubMetrics:
        """Compute TrackMan-style metrics from the recorded stroke."""
        metrics = ClubMetrics(stroke_phase=StrokePhase.COMPLETE)

        # Backswing and forward swing timing
        if self._backswing_start_ns and self._downswing_start_ns:
            metrics.backswing_time_ms = (self._downswing_start_ns - self._backswing_start_ns) / 1e6
        if self._downswing_start_ns and self._impact_ns:
            metrics.forward_swing_time_ms = (self._impact_ns - self._downswing_start_ns) / 1e6
        if metrics.forward_swing_time_ms > 0:
            metrics.stroke_tempo = metrics.backswing_time_ms / metrics.forward_swing_time_ms

        # Path lengths
        metrics.backswing_length_m = self._compute_path_length(self._backswing_positions)
        metrics.forward_swing_length_m = self._compute_path_length(self._downswing_positions)

        # Club speed at impact (last few downswing frames)
        if len(self._downswing_positions) >= 2:
            p1 = self._downswing_positions[-min(3, len(self._downswing_positions))]
            p2 = self._downswing_positions[-1]
            dt = (p2.timestamp_ns - p1.timestamp_ns) / 1e9
            if dt > 0:
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                dz = p2.z - p1.z
                metrics.club_speed_m_s = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) / dt

                # Club path angle (horizontal)
                # In our coordinate system, target is to the left (-x)
                # Path angle: 0 = straight at target, + = in-to-out, - = out-to-in
                metrics.club_path_deg = np.degrees(np.arctan2(dz, -dx))

                # Attack angle (vertical)
                horizontal = np.sqrt(dx ** 2 + dz ** 2)
                if horizontal > 0.001:
                    metrics.attack_angle_deg = np.degrees(np.arctan2(-dy, horizontal))

        # Face angle estimate from club head shape orientation (simplified)
        # Full implementation would analyze the contour shape at impact
        # For now, approximate from path + ball direction
        metrics.face_angle_deg = metrics.club_path_deg * 0.7  # Simplified estimate

        return metrics

    def _compute_path_length(self, positions: List[ClubPosition]) -> float:
        """Compute arc length of a path segment."""
        if len(positions) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(positions)):
            p1, p2 = positions[i - 1], positions[i]
            total += np.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2 + (p2.z - p1.z) ** 2)
        return total

    def reset(self) -> None:
        """Reset for a new stroke."""
        self._positions.clear()
        self._stroke_phase = StrokePhase.IDLE
        self._backswing_positions.clear()
        self._downswing_positions.clear()
        self._stroke_path = ClubPath()
        self._metrics = None

    def get_stroke_path(self) -> ClubPath:
        return self._stroke_path

    def get_latest_metrics(self) -> Optional[ClubMetrics]:
        return self._metrics

    @property
    def stroke_phase(self) -> StrokePhase:
        return self._stroke_phase
