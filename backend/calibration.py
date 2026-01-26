"""
Calibration module for StrikeLab Putting Sim.
Handles homography computation for world coordinate mapping.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Result of calibration process."""
    success: bool
    homography_matrix: Optional[np.ndarray] = None
    pixels_per_meter: float = 0.0
    origin_px: Tuple[int, int] = (0, 0)
    forward_direction_deg: float = 0.0
    error_message: Optional[str] = None


class Calibrator:
    """
    Handles calibration for world coordinate mapping.
    
    Supports:
    - ArUco marker detection (4 markers at known positions)
    - Manual 4-point click calibration
    
    World coordinate system:
    - Origin: Configurable (default = first marker/corner)
    - +X: Forward toward hole (calibrated direction)
    - +Y: Perpendicular (left/right)
    """
    
    # ArUco dictionary
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    
    def __init__(self):
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(self.ARUCO_DICT)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)
        
        # Calibration state
        self._homography: Optional[np.ndarray] = None
        self._pixels_per_meter: float = 1500.0
        self._origin_px: Tuple[int, int] = (0, 0)
        self._forward_direction_deg: float = 0.0
    
    def calibrate_from_aruco(
        self,
        frame: np.ndarray,
        marker_world_positions: dict,  # {marker_id: (x_m, y_m)}
        marker_size_m: float = 0.05
    ) -> CalibrationResult:
        """
        Calibrate using ArUco markers.
        
        Args:
            frame: Camera frame with visible markers
            marker_world_positions: Dict mapping marker IDs to world positions in meters
            marker_size_m: Physical size of markers in meters
            
        Returns:
            CalibrationResult with homography matrix
        """
        # Detect markers
        corners, ids, rejected = self._detector.detectMarkers(frame)
        
        if ids is None or len(ids) < 4:
            return CalibrationResult(
                success=False,
                error_message=f"Need 4 markers, found {len(ids) if ids is not None else 0}"
            )
        
        # Match detected markers to world positions
        image_points = []
        world_points = []
        
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in marker_world_positions:
                # Use center of marker
                marker_corners = corners[i][0]
                center = marker_corners.mean(axis=0)
                
                image_points.append(center)
                world_points.append(marker_world_positions[marker_id])
        
        if len(image_points) < 4:
            return CalibrationResult(
                success=False,
                error_message=f"Only {len(image_points)} markers matched to known positions"
            )
        
        return self._compute_homography(
            np.array(image_points),
            np.array(world_points)
        )
    
    def calibrate_from_points(
        self,
        image_points: List[Tuple[float, float]],
        world_points: List[Tuple[float, float]]
    ) -> CalibrationResult:
        """
        Calibrate using manual point correspondences.
        
        Args:
            image_points: 4+ points in pixel coordinates
            world_points: Corresponding points in world coordinates (meters)
            
        Returns:
            CalibrationResult with homography matrix
        """
        if len(image_points) < 4 or len(world_points) < 4:
            return CalibrationResult(
                success=False,
                error_message="Need at least 4 point correspondences"
            )
        
        if len(image_points) != len(world_points):
            return CalibrationResult(
                success=False,
                error_message="Image and world point counts must match"
            )
        
        return self._compute_homography(
            np.array(image_points, dtype=np.float32),
            np.array(world_points, dtype=np.float32)
        )
    
    def calibrate_from_rectangle(
        self,
        image_corners: List[Tuple[float, float]],
        width_m: float,
        height_m: float,
        forward_edge: str = "right"  # Which edge points toward hole
    ) -> CalibrationResult:
        """
        Calibrate using 4 corners of a known rectangle.
        
        Args:
            image_corners: 4 corners in pixel coords [top-left, top-right, bottom-right, bottom-left]
            width_m: Rectangle width in meters (left-right)
            height_m: Rectangle height in meters (top-bottom)
            forward_edge: Which edge points toward hole ("right", "left", "top", "bottom")
            
        Returns:
            CalibrationResult with homography matrix
        """
        if len(image_corners) != 4:
            return CalibrationResult(
                success=False,
                error_message="Need exactly 4 corners"
            )
        
        # Define world coordinates based on forward direction
        # Default: ball rolls left->right, so +X is to the right
        if forward_edge == "right":
            world_corners = [
                (0, height_m),      # top-left -> (0, h)
                (width_m, height_m), # top-right -> (w, h)
                (width_m, 0),        # bottom-right -> (w, 0)
                (0, 0)               # bottom-left -> origin
            ]
        elif forward_edge == "left":
            world_corners = [
                (width_m, 0),
                (0, 0),
                (0, height_m),
                (width_m, height_m)
            ]
        elif forward_edge == "top":
            world_corners = [
                (0, 0),
                (width_m, 0),
                (width_m, height_m),
                (0, height_m)
            ]
        else:  # bottom
            world_corners = [
                (width_m, height_m),
                (0, height_m),
                (0, 0),
                (width_m, 0)
            ]
        
        return self.calibrate_from_points(image_corners, world_corners)
    
    def _compute_homography(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray
    ) -> CalibrationResult:
        """Compute homography matrix from point correspondences."""
        try:
            # Compute homography: image -> world
            H, mask = cv2.findHomography(image_points, world_points, cv2.RANSAC, 5.0)
            
            if H is None:
                return CalibrationResult(
                    success=False,
                    error_message="Failed to compute homography"
                )
            
            # Compute pixels per meter (average scale factor)
            # Transform unit vectors to estimate scale
            p0 = np.array([[0, 0]], dtype=np.float32)
            p1 = np.array([[100, 0]], dtype=np.float32)
            
            w0 = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).flatten()
            w1 = cv2.perspectiveTransform(p1.reshape(-1, 1, 2), H).flatten()
            
            world_dist = np.linalg.norm(w1 - w0)
            pixels_per_meter = 100.0 / world_dist if world_dist > 0 else 1500.0
            
            # Origin in pixel space (inverse transform of world origin)
            H_inv = np.linalg.inv(H)
            origin_world = np.array([[0, 0]], dtype=np.float32)
            origin_px = cv2.perspectiveTransform(
                origin_world.reshape(-1, 1, 2), H_inv
            ).flatten()
            
            # Forward direction: angle of +X axis in image coordinates
            x_world = np.array([[1, 0]], dtype=np.float32)
            x_px = cv2.perspectiveTransform(
                x_world.reshape(-1, 1, 2), H_inv
            ).flatten()
            
            forward_vec = x_px - origin_px
            forward_direction_deg = float(np.degrees(np.arctan2(forward_vec[1], forward_vec[0])))
            
            # Store calibration
            self._homography = H
            self._pixels_per_meter = pixels_per_meter
            self._origin_px = (int(origin_px[0]), int(origin_px[1]))
            self._forward_direction_deg = forward_direction_deg
            
            logger.info(f"Calibration successful: {pixels_per_meter:.1f} px/m, "
                       f"forward={forward_direction_deg:.1f}°")
            
            return CalibrationResult(
                success=True,
                homography_matrix=H,
                pixels_per_meter=pixels_per_meter,
                origin_px=self._origin_px,
                forward_direction_deg=forward_direction_deg
            )
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return CalibrationResult(
                success=False,
                error_message=str(e)
            )
    
    def pixel_to_world(self, x_px: float, y_px: float) -> Optional[Tuple[float, float]]:
        """Convert pixel coordinates to world coordinates (meters)."""
        if self._homography is None:
            return None
        
        point = np.array([[x_px, y_px]], dtype=np.float32)
        world = cv2.perspectiveTransform(point.reshape(-1, 1, 2), self._homography)
        return (float(world[0, 0, 0]), float(world[0, 0, 1]))
    
    def world_to_pixel(self, x_m: float, y_m: float) -> Optional[Tuple[float, float]]:
        """Convert world coordinates (meters) to pixel coordinates."""
        if self._homography is None:
            return None
        
        H_inv = np.linalg.inv(self._homography)
        point = np.array([[x_m, y_m]], dtype=np.float32)
        pixel = cv2.perspectiveTransform(point.reshape(-1, 1, 2), H_inv)
        return (float(pixel[0, 0, 0]), float(pixel[0, 0, 1]))
    
    def velocity_to_world(
        self,
        vx_px_s: float,
        vy_px_s: float,
        at_position: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Convert pixel velocity to world velocity (m/s).
        
        Note: Homography is nonlinear, so velocity depends on position.
        For small velocities, we use local linearization.
        """
        if self._homography is None:
            # Fallback: use simple scale
            scale = 1.0 / self._pixels_per_meter
            return (vx_px_s * scale, vy_px_s * scale)
        
        # Transform position and position + velocity
        x, y = at_position
        dt = 0.001  # Small time step for linearization
        
        p1 = self.pixel_to_world(x, y)
        p2 = self.pixel_to_world(x + vx_px_s * dt, y + vy_px_s * dt)
        
        if p1 is None or p2 is None:
            return (0.0, 0.0)
        
        vx_m_s = (p2[0] - p1[0]) / dt
        vy_m_s = (p2[1] - p1[1]) / dt
        
        return (vx_m_s, vy_m_s)
    
    def direction_relative_to_forward(self, direction_deg_image: float) -> float:
        """
        Convert direction in image coordinates to direction relative to calibrated forward.
        
        Args:
            direction_deg_image: Direction in image coordinates (0 = right, 90 = down)
            
        Returns:
            Direction relative to forward (+X axis), positive = right of forward
        """
        return direction_deg_image - self._forward_direction_deg
    
    def load_from_config(self, calibration_data: dict) -> bool:
        """Load calibration from config data."""
        try:
            if "homography_matrix" in calibration_data and calibration_data["homography_matrix"]:
                self._homography = np.array(calibration_data["homography_matrix"], dtype=np.float64)
            
            self._pixels_per_meter = calibration_data.get("pixels_per_meter", 1500.0)
            
            origin = calibration_data.get("origin_px", (0, 0))
            self._origin_px = tuple(origin) if isinstance(origin, (list, tuple)) else (0, 0)
            
            self._forward_direction_deg = calibration_data.get("forward_direction_deg", 0.0)
            
            logger.info("Loaded calibration from config")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    @property
    def is_calibrated(self) -> bool:
        return self._homography is not None
    
    @property
    def pixels_per_meter(self) -> float:
        return self._pixels_per_meter
    
    @property
    def forward_direction_deg(self) -> float:
        return self._forward_direction_deg


# Convenience functions for ArUco marker detection
def detect_aruco_markers(frame: np.ndarray) -> Tuple[list, list]:
    """
    Detect ArUco markers in frame.
    
    Returns:
        (corners, ids) - corners is list of 4x2 arrays, ids is list of marker IDs
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    
    corners, ids, _ = detector.detectMarkers(frame)
    
    if ids is None:
        return [], []
    
    return corners, ids.flatten().tolist()


def draw_aruco_markers(frame: np.ndarray, corners: list, ids: list) -> np.ndarray:
    """Draw detected ArUco markers on frame."""
    frame_copy = frame.copy()
    
    if corners and ids:
        cv2.aruco.drawDetectedMarkers(frame_copy, corners, np.array(ids))
    
    return frame_copy
