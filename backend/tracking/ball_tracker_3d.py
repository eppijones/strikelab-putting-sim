"""
3D ball tracker using ZED 2i depth data.

Detects and tracks the golf ball in 3D space:
- X,Y position on the green surface (confirms Arducam 2D track)
- Z height above the surface (critical for chipping detection)
- Ball liftoff event for chip shots
- 3D velocity and trajectory

Ball moves right to left in the image.
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

GOLF_BALL_DIAMETER_M = 0.04267
GOLF_BALL_RADIUS_M = GOLF_BALL_DIAMETER_M / 2


class BallFlightPhase(Enum):
    GROUNDED = "grounded"
    LAUNCHING = "launching"
    AIRBORNE = "airborne"
    LANDING = "landing"


@dataclass
class BallPosition3D:
    """3D ball position from depth camera."""
    x: float   # meters, right (+) / left (-)
    y: float   # meters, up (+) / down (-)
    z: float   # meters, depth from camera
    px_x: int  # pixel x in depth image
    px_y: int  # pixel y in depth image
    height_above_surface: float = 0.0  # meters above putting surface
    timestamp_ns: int = 0
    confidence: float = 0.0


@dataclass
class BallTrajectory3D:
    """3D trajectory segment."""
    positions: List[BallPosition3D] = field(default_factory=list)
    velocities: List[Tuple[float, float, float]] = field(default_factory=list)
    flight_phase: BallFlightPhase = BallFlightPhase.GROUNDED
    launch_angle_deg: float = 0.0
    peak_height_m: float = 0.0
    total_distance_m: float = 0.0


@dataclass
class Ball3DState:
    """Current 3D ball state for external consumption."""
    position: Optional[BallPosition3D] = None
    velocity_3d: Optional[Tuple[float, float, float]] = None  # m/s (vx, vy, vz)
    speed_m_s: float = 0.0
    flight_phase: BallFlightPhase = BallFlightPhase.GROUNDED
    height_above_surface: float = 0.0
    is_liftoff: bool = False
    launch_angle_deg: float = 0.0

    def to_dict(self) -> dict:
        pos = None
        if self.position:
            pos = {
                "x": round(self.position.x, 4),
                "y": round(self.position.y, 4),
                "z": round(self.position.z, 4),
                "height_m": round(self.position.height_above_surface, 4),
            }
        vel = None
        if self.velocity_3d:
            vel = {
                "vx": round(self.velocity_3d[0], 3),
                "vy": round(self.velocity_3d[1], 3),
                "vz": round(self.velocity_3d[2], 3),
            }
        return {
            "position_3d": pos,
            "velocity_3d": vel,
            "speed_m_s": round(self.speed_m_s, 3),
            "flight_phase": self.flight_phase.value,
            "height_above_surface_m": round(self.height_above_surface, 4),
            "is_liftoff": self.is_liftoff,
            "launch_angle_deg": round(self.launch_angle_deg, 1),
        }


class BallTracker3D:
    """
    Tracks the golf ball in 3D using the ZED 2i depth camera.

    Uses color segmentation on the left image to find the ball's 2D position,
    then samples the depth map / point cloud for the 3D coordinate.

    Detects ball liftoff by monitoring height above the calibrated surface plane.
    """

    LIFTOFF_THRESHOLD_M = 0.008  # 8mm above surface = liftoff
    BALL_HSV_LOWER = np.array([0, 0, 160])
    BALL_HSV_UPPER = np.array([180, 70, 255])
    MIN_BALL_AREA = 50
    MAX_BALL_AREA = 5000
    MIN_CIRCULARITY = 0.6
    VELOCITY_WINDOW = 5

    def __init__(self, surface_height_m: float = 0.45):
        """
        Args:
            surface_height_m: Expected depth to the putting surface from ZED (meters).
                              At 40-50cm mount height, this is ~0.40-0.50m.
        """
        self._surface_height_m = surface_height_m
        self._positions: deque[BallPosition3D] = deque(maxlen=120)
        self._flight_phase = BallFlightPhase.GROUNDED
        self._liftoff_detected = False
        self._peak_height = 0.0
        self._trajectory = BallTrajectory3D()
        self._surface_calibrated = False
        self._surface_depth_samples: deque[float] = deque(maxlen=60)

    def update(self, color: np.ndarray, depth: np.ndarray,
               point_cloud: Optional[np.ndarray], timestamp_ns: int) -> Ball3DState:
        """
        Process a ZED frame to track the ball in 3D.

        Args:
            color: BGR left image from ZED
            depth: float32 depth map (meters)
            point_cloud: Optional XYZ point cloud (H, W, 3)
            timestamp_ns: Frame timestamp
        """
        state = Ball3DState()

        detection_2d = self._detect_ball_2d(color)
        if detection_2d is None:
            return state

        px_x, px_y, radius = detection_2d

        # Get 3D position from depth or point cloud
        pos_3d = self._get_3d_position(px_x, px_y, radius, depth, point_cloud, timestamp_ns)
        if pos_3d is None:
            return state

        self._positions.append(pos_3d)
        self._trajectory.positions.append(pos_3d)

        # Calibrate surface depth from early grounded observations
        if len(self._positions) < 30 and pos_3d.height_above_surface < 0.01:
            self._surface_depth_samples.append(pos_3d.z)
            if len(self._surface_depth_samples) >= 10:
                self._surface_height_m = float(np.median(list(self._surface_depth_samples)))
                self._surface_calibrated = True

        # Detect flight phase
        height = self._compute_height_above_surface(pos_3d.z)
        pos_3d.height_above_surface = height

        prev_phase = self._flight_phase
        self._update_flight_phase(height)

        liftoff = (self._flight_phase == BallFlightPhase.LAUNCHING and
                   prev_phase == BallFlightPhase.GROUNDED)

        if height > self._peak_height:
            self._peak_height = height

        # Compute velocity
        velocity_3d = self._compute_velocity_3d()
        speed = 0.0
        launch_angle = 0.0
        if velocity_3d:
            speed = np.sqrt(sum(v ** 2 for v in velocity_3d))
            if speed > 0.1 and self._flight_phase != BallFlightPhase.GROUNDED:
                horizontal_speed = np.sqrt(velocity_3d[0] ** 2 + velocity_3d[2] ** 2)
                if horizontal_speed > 0.01:
                    launch_angle = np.degrees(np.arctan2(velocity_3d[1], horizontal_speed))

        state.position = pos_3d
        state.velocity_3d = velocity_3d
        state.speed_m_s = speed
        state.flight_phase = self._flight_phase
        state.height_above_surface = height
        state.is_liftoff = liftoff
        state.launch_angle_deg = launch_angle

        return state

    def _detect_ball_2d(self, color: np.ndarray) -> Optional[Tuple[int, int, float]]:
        """Find the ball in the color image. Returns (cx, cy, radius) or None."""
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.BALL_HSV_LOWER, self.BALL_HSV_UPPER)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.MIN_CIRCULARITY:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            score = circularity * (area / self.MAX_BALL_AREA)
            if score > best_score:
                best_score = score
                best = (int(cx), int(cy), float(radius))

        return best

    def _get_3d_position(
        self, px_x: int, px_y: int, radius: float,
        depth: np.ndarray, point_cloud: Optional[np.ndarray],
        timestamp_ns: int,
    ) -> Optional[BallPosition3D]:
        """Sample depth/point cloud around the ball center for robust 3D position."""
        h, w = depth.shape[:2]
        if px_x < 0 or px_x >= w or px_y < 0 or px_y >= h:
            return None

        # Sample a small region around ball center
        r = max(2, int(radius * 0.3))
        y1, y2 = max(0, px_y - r), min(h, px_y + r + 1)
        x1, x2 = max(0, px_x - r), min(w, px_x + r + 1)

        depth_roi = depth[y1:y2, x1:x2]
        valid = depth_roi[(depth_roi > 0.1) & (depth_roi < 2.0) & np.isfinite(depth_roi)]

        if len(valid) < 3:
            return None

        z = float(np.median(valid))

        # Use point cloud for XY if available
        x_m, y_m = 0.0, 0.0
        if point_cloud is not None and point_cloud.shape[:2] == depth.shape[:2]:
            pc_roi = point_cloud[y1:y2, x1:x2]
            valid_mask = np.isfinite(pc_roi[:, :, 0]) & np.isfinite(pc_roi[:, :, 1])
            if np.sum(valid_mask) > 0:
                x_m = float(np.median(pc_roi[:, :, 0][valid_mask]))
                y_m = float(np.median(pc_roi[:, :, 1][valid_mask]))

        height = self._compute_height_above_surface(z)

        return BallPosition3D(
            x=x_m, y=y_m, z=z,
            px_x=px_x, px_y=px_y,
            height_above_surface=height,
            timestamp_ns=timestamp_ns,
            confidence=min(1.0, len(valid) / max(1, (2 * r + 1) ** 2)),
        )

    def _compute_height_above_surface(self, depth_m: float) -> float:
        """
        Height above surface. Surface is farther from camera than the ball top.
        height = surface_depth - ball_depth - ball_radius
        """
        raw = self._surface_height_m - depth_m
        return max(0.0, raw - GOLF_BALL_RADIUS_M)

    def _update_flight_phase(self, height: float) -> None:
        if self._flight_phase == BallFlightPhase.GROUNDED:
            if height > self.LIFTOFF_THRESHOLD_M:
                self._flight_phase = BallFlightPhase.LAUNCHING
                self._liftoff_detected = True
        elif self._flight_phase == BallFlightPhase.LAUNCHING:
            if len(self._positions) >= 3:
                recent = [p.height_above_surface for p in list(self._positions)[-3:]]
                if len(recent) >= 2 and recent[-1] < recent[-2]:
                    self._flight_phase = BallFlightPhase.AIRBORNE
            if height > self._peak_height * 0.8:
                pass  # still rising
        elif self._flight_phase == BallFlightPhase.AIRBORNE:
            if height < self.LIFTOFF_THRESHOLD_M:
                self._flight_phase = BallFlightPhase.LANDING
        elif self._flight_phase == BallFlightPhase.LANDING:
            if height < 0.002:
                self._flight_phase = BallFlightPhase.GROUNDED
                self._liftoff_detected = False
                self._peak_height = 0.0

    def _compute_velocity_3d(self) -> Optional[Tuple[float, float, float]]:
        """Compute 3D velocity from recent positions."""
        if len(self._positions) < 2:
            return None

        window = min(self.VELOCITY_WINDOW, len(self._positions))
        positions = list(self._positions)[-window:]

        p1, p2 = positions[0], positions[-1]
        dt = (p2.timestamp_ns - p1.timestamp_ns) / 1e9
        if dt <= 0:
            return None

        vx = (p2.x - p1.x) / dt
        vy = (p2.y - p1.y) / dt
        vz = (p2.z - p1.z) / dt

        return (vx, vy, vz)

    def reset(self) -> None:
        """Reset tracker for a new shot."""
        self._positions.clear()
        self._flight_phase = BallFlightPhase.GROUNDED
        self._liftoff_detected = False
        self._peak_height = 0.0
        self._trajectory = BallTrajectory3D()

    def get_trajectory(self) -> BallTrajectory3D:
        return self._trajectory
