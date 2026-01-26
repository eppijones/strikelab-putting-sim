"""
Ball detector for StrikeLab Putting Sim.
Detects white golf ball on green putting mat using color thresholding.
Excludes ArUco markers from detection to prevent false positives.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def get_aruco_mask(frame: np.ndarray, padding: int = 30) -> np.ndarray:
    """
    Detect ArUco markers and return a mask with marker regions blocked out.
    
    Args:
        frame: BGR image
        padding: Extra pixels to mask around each marker
        
    Returns:
        Mask where 255 = valid area, 0 = marker area (to be excluded)
    """
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    
    try:
        # Detect ArUco markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        
        corners, ids, _ = detector.detectMarkers(frame)
        
        if ids is not None:
            for corner in corners:
                # Get bounding box with padding
                pts = corner[0].astype(np.int32)
                x_min = max(0, pts[:, 0].min() - padding)
                x_max = min(frame.shape[1], pts[:, 0].max() + padding)
                y_min = max(0, pts[:, 1].min() - padding)
                y_max = min(frame.shape[0], pts[:, 1].max() + padding)
                
                # Block out marker region
                mask[y_min:y_max, x_min:x_max] = 0
                
    except Exception as e:
        logger.debug(f"ArUco detection failed: {e}")
    
    return mask


@dataclass
class Detection:
    """Ball detection result."""
    cx: float  # Center X in pixels
    cy: float  # Center Y in pixels
    radius: float  # Radius in pixels
    confidence: float  # Detection confidence [0, 1]
    contour: Optional[np.ndarray] = None  # Original contour for debugging
    
    def as_tuple(self) -> Tuple[float, float]:
        """Return center as (x, y) tuple."""
        return (self.cx, self.cy)
    
    def distance_to(self, other: 'Detection') -> float:
        """Euclidean distance to another detection."""
        return np.sqrt((self.cx - other.cx)**2 + (self.cy - other.cy)**2)


class BallDetector:
    """
    Detects white golf ball on green putting mat.
    
    Uses HSV color thresholding optimized for:
    - White ball: Low saturation, high value
    - Green mat: Provides good contrast
    
    Tunable parameters for different lighting conditions.
    """
    
    # Default HSV thresholds for white ball detection
    # White: any hue, low saturation, high value
    DEFAULT_WHITE_LOWER = np.array([0, 0, 180])
    DEFAULT_WHITE_UPPER = np.array([180, 60, 255])
    
    # Ball size constraints (in pixels at typical mounting height)
    DEFAULT_MIN_RADIUS = 10      # Increased to filter out small marker corners
    DEFAULT_MAX_RADIUS = 60
    DEFAULT_MIN_AREA = 300       # Golf ball should be at least this big
    DEFAULT_MAX_AREA = 12000
    
    # Circularity threshold (1.0 = perfect circle, 0.78 = square)
    DEFAULT_MIN_CIRCULARITY = 0.82  # Must be more circular than a square
    
    def __init__(
        self,
        white_lower: Optional[np.ndarray] = None,
        white_upper: Optional[np.ndarray] = None,
        min_radius: int = DEFAULT_MIN_RADIUS,
        max_radius: int = DEFAULT_MAX_RADIUS,
        min_area: int = DEFAULT_MIN_AREA,
        max_area: int = DEFAULT_MAX_AREA,
        min_circularity: float = DEFAULT_MIN_CIRCULARITY,
        use_blur: bool = True,
        blur_kernel: int = 5,
        exclude_aruco: bool = True
    ):
        self.white_lower = white_lower if white_lower is not None else self.DEFAULT_WHITE_LOWER.copy()
        self.white_upper = white_upper if white_upper is not None else self.DEFAULT_WHITE_UPPER.copy()
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_area = min_area
        self.max_area = max_area
        self.min_circularity = min_circularity
        self.use_blur = use_blur
        self.blur_kernel = blur_kernel
        self.exclude_aruco = exclude_aruco
        
        # Morphological kernel for noise removal
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Debug: store intermediate results
        self.debug_mask: Optional[np.ndarray] = None
        self.debug_contours: list = []
        
        # Cache ArUco mask (update rarely - markers don't move)
        self._aruco_mask: Optional[np.ndarray] = None
        self._aruco_mask_frame_count = 0
        self._aruco_mask_update_interval = 240  # Update every 240 frames (~2 sec at 120fps)
    
    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """
        Detect golf ball in frame.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Detection if ball found, None otherwise
        """
        if frame is None or frame.size == 0:
            return None
        
        # Optional blur to reduce noise
        if self.use_blur:
            blurred = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
        else:
            blurred = frame
        
        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Threshold for white
        mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        
        # Exclude ArUco marker regions
        if self.exclude_aruco:
            self._aruco_mask_frame_count += 1
            # Update ArUco mask periodically (not every frame for performance)
            if self._aruco_mask is None or self._aruco_mask_frame_count >= self._aruco_mask_update_interval:
                self._aruco_mask = get_aruco_mask(frame, padding=40)
                self._aruco_mask_frame_count = 0
            
            # Apply ArUco exclusion mask
            if self._aruco_mask is not None:
                mask = cv2.bitwise_and(mask, self._aruco_mask)
        
        # Morphological operations to clean up
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel)
        
        # Store for debugging
        self.debug_mask = mask.copy()
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.debug_contours = contours
        
        if not contours:
            return None
        
        # Find best ball candidate
        best_detection: Optional[Detection] = None
        best_score = 0.0
        
        for contour in contours:
            detection = self._evaluate_contour(contour)
            if detection and detection.confidence > best_score:
                best_score = detection.confidence
                best_detection = detection
        
        return best_detection
    
    def _evaluate_contour(self, contour: np.ndarray) -> Optional[Detection]:
        """Evaluate a contour as potential ball detection."""
        area = cv2.contourArea(contour)
        
        # Area filter
        if area < self.min_area or area > self.max_area:
            return None
        
        # Fit minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # Radius filter
        if radius < self.min_radius or radius > self.max_radius:
            return None
        
        # Circularity check
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity < self.min_circularity:
            return None
        
        # Compute confidence based on circularity and how well area matches circle
        expected_area = np.pi * radius * radius
        area_ratio = min(area / expected_area, expected_area / area) if expected_area > 0 else 0
        
        confidence = (circularity * 0.6 + area_ratio * 0.4)
        
        return Detection(
            cx=float(cx),
            cy=float(cy),
            radius=float(radius),
            confidence=float(confidence),
            contour=contour
        )
    
    def detect_in_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> Optional[Detection]:
        """
        Detect ball within a region of interest.
        
        Args:
            frame: Full BGR image
            roi: (x, y, width, height) of ROI
            
        Returns:
            Detection with coordinates in full frame space
        """
        x, y, w, h = roi
        
        # Clamp ROI to frame bounds
        h_frame, w_frame = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if w <= 0 or h <= 0:
            return None
        
        # Extract ROI
        roi_frame = frame[y:y+h, x:x+w]
        
        # Detect in ROI
        detection = self.detect(roi_frame)
        
        if detection:
            # Transform coordinates back to full frame
            detection.cx += x
            detection.cy += y
        
        return detection
    
    def update_thresholds(
        self,
        white_lower: Optional[np.ndarray] = None,
        white_upper: Optional[np.ndarray] = None
    ):
        """Update detection thresholds (for runtime tuning)."""
        if white_lower is not None:
            self.white_lower = white_lower
        if white_upper is not None:
            self.white_upper = white_upper
        logger.info(f"Updated thresholds: lower={self.white_lower}, upper={self.white_upper}")


class MultiScaleDetector(BallDetector):
    """
    Ball detector with multi-scale detection for varying ball sizes.
    Useful if camera height varies or ball appears at different distances.
    """
    
    def __init__(self, scales: list[float] = [0.5, 1.0, 1.5], **kwargs):
        super().__init__(**kwargs)
        self.scales = scales
    
    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """Detect at multiple scales and return best result."""
        best_detection: Optional[Detection] = None
        best_confidence = 0.0
        
        for scale in self.scales:
            if scale == 1.0:
                scaled_frame = frame
            else:
                scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            
            detection = super().detect(scaled_frame)
            
            if detection and detection.confidence > best_confidence:
                # Scale coordinates back
                if scale != 1.0:
                    detection.cx /= scale
                    detection.cy /= scale
                    detection.radius /= scale
                best_detection = detection
                best_confidence = detection.confidence
        
        return best_detection
