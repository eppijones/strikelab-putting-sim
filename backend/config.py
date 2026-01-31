"""
Configuration management for StrikeLab Putting Sim.
Handles settings persistence and calibration data.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class CameraSettings:
    """Camera configuration."""
    width: int = 1280
    height: int = 800
    fps: int = 120
    device_id: int = 0
    exposure: int = -6
    auto_exposure: bool = False


@dataclass 
class DetectorSettings:
    """Ball detector configuration."""
    white_lower_h: int = 0
    white_lower_s: int = 0
    white_lower_v: int = 180
    white_upper_h: int = 180
    white_upper_s: int = 60
    white_upper_v: int = 255
    min_radius: int = 5
    max_radius: int = 50
    min_area: int = 80
    max_area: int = 8000
    min_circularity: float = 0.6


@dataclass
class TrackerSettings:
    """Tracker configuration."""
    motion_threshold_px: float = 5.0
    motion_confirm_frames: int = 2
    stopped_velocity_threshold: float = 50.0
    stopped_confirm_frames: int = 10
    cooldown_duration_ms: int = 500
    idle_ema_alpha: float = 0.05
    # Motion direction filter - prevents false triggers from putter swing/hand
    valid_motion_angle_deg: float = 45.0  # Accept motion within +/- this angle from forward
    forward_direction_deg: float = 0.0     # Default forward direction (inherited from calibration)
    # Timestamp-based stop detection (replaces frame-count-based)
    # Ball must be below stopped_velocity_threshold for this duration to confirm stop
    stopped_confirm_time_ms: int = 100  # ~12 frames at 120fps


@dataclass
class LensCalibrationData:
    """Lens distortion calibration data."""
    camera_matrix: Optional[list] = None   # 3x3 intrinsic matrix as nested list
    dist_coeffs: Optional[list] = None     # Distortion coefficients [k1, k2, p1, p2, k3]
    image_size: Optional[tuple] = None     # (width, height)
    reprojection_error: float = 0.0        # RMS error in pixels
    calibrated_at: Optional[str] = None
    
    def is_valid(self) -> bool:
        """Check if lens calibration is valid."""
        return (
            self.camera_matrix is not None and
            self.dist_coeffs is not None and
            len(self.camera_matrix) == 3 and
            len(self.dist_coeffs) >= 4
        )


@dataclass
class CalibrationData:
    """Calibration data for world coordinate mapping."""
    version: int = 1
    homography_matrix: Optional[list] = None  # 3x3 matrix as nested list
    pixels_per_meter: float = 1500.0
    origin_px: tuple = (0, 0)
    forward_direction_deg: float = 0.0  # Angle of +X axis in image coordinates
    created_at: Optional[str] = None
    # Distance scale factor for real-world calibration fine-tuning
    # If measured distances are consistently off, adjust this value
    # Example: if real distance is 1.74m but we measure 1.63m, set to 1.74/1.63 = 1.067
    distance_scale_factor: float = 1.0
    # Virtual ball deceleration in m/s² - tune this for your putting surface
    # Higher friction (rough carpet): 1.0-1.2 m/s²
    # Medium friction (putting mat): 0.6-0.8 m/s²
    # Low friction (fast green): 0.4-0.5 m/s²
    virtual_deceleration_m_s2: float = 0.65
    # Manual override for pixels_per_meter (set to 0 to use auto-calibration)
    # If auto-calibration gives wrong values, measure manually and set here
    manual_pixels_per_meter: float = 0.0
    # UI overlay scale - multiply detected radius by this for display ONLY
    # Does NOT affect tracking or detection - purely visual
    # Typical range: 1.14-1.18 (ball appears ~13% smaller in detection than reality)
    overlay_radius_scale: float = 1.15
    
    def is_valid(self) -> bool:
        """Check if calibration data is valid and complete."""
        return (
            self.homography_matrix is not None and
            len(self.homography_matrix) == 3 and
            all(len(row) == 3 for row in self.homography_matrix)
        )


@dataclass
class PredictionSettings:
    """Ball prediction settings."""
    friction_coefficient: float = 0.15  # Typical putting green
    min_velocity_threshold: float = 10.0  # px/s - below this, stop prediction
    max_prediction_time_s: float = 10.0  # Maximum prediction duration


@dataclass
class Config:
    """Main configuration container."""
    camera: CameraSettings = field(default_factory=CameraSettings)
    detector: DetectorSettings = field(default_factory=DetectorSettings)
    tracker: TrackerSettings = field(default_factory=TrackerSettings)
    calibration: CalibrationData = field(default_factory=CalibrationData)
    lens_calibration: LensCalibrationData = field(default_factory=LensCalibrationData)
    prediction: PredictionSettings = field(default_factory=PredictionSettings)
    
    # Server settings
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    websocket_path: str = "/ws"


class ConfigManager:
    """
    Manages configuration loading and saving.
    Handles config.json persistence with validation.
    """
    
    DEFAULT_CONFIG_PATH = Path("config.json")
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = Config()
    
    def load(self) -> Config:
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.info(f"No config file found at {self.config_path}, using defaults")
            return self.config
        
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            self._apply_dict_to_config(data)
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        
        return self.config
    
    def save(self) -> bool:
        """Save current configuration to file."""
        try:
            data = self._config_to_dict()
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved configuration to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def _config_to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        return {
            "camera": asdict(self.config.camera),
            "detector": asdict(self.config.detector),
            "tracker": asdict(self.config.tracker),
            "calibration": asdict(self.config.calibration),
            "lens_calibration": asdict(self.config.lens_calibration),
            "prediction": asdict(self.config.prediction),
            "server_host": self.config.server_host,
            "server_port": self.config.server_port,
            "websocket_path": self.config.websocket_path
        }
    
    def _apply_dict_to_config(self, data: dict):
        """Apply dictionary data to config object."""
        if "camera" in data:
            self._update_dataclass(self.config.camera, data["camera"])
        if "detector" in data:
            self._update_dataclass(self.config.detector, data["detector"])
        if "tracker" in data:
            self._update_dataclass(self.config.tracker, data["tracker"])
        if "calibration" in data:
            self._update_dataclass(self.config.calibration, data["calibration"])
        if "lens_calibration" in data:
            self._update_dataclass(self.config.lens_calibration, data["lens_calibration"])
        if "prediction" in data:
            self._update_dataclass(self.config.prediction, data["prediction"])
        
        if "server_host" in data:
            self.config.server_host = data["server_host"]
        if "server_port" in data:
            self.config.server_port = data["server_port"]
        if "websocket_path" in data:
            self.config.websocket_path = data["websocket_path"]
    
    def _update_dataclass(self, obj: Any, data: dict):
        """Update dataclass fields from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def update_calibration(
        self,
        homography_matrix: list,
        pixels_per_meter: float,
        origin_px: tuple,
        forward_direction_deg: float
    ) -> bool:
        """Update calibration data and save."""
        self.config.calibration = CalibrationData(
            version=1,
            homography_matrix=homography_matrix,
            pixels_per_meter=pixels_per_meter,
            origin_px=origin_px,
            forward_direction_deg=forward_direction_deg,
            created_at=datetime.utcnow().isoformat() + "Z"
        )
        return self.save()
    
    def get_calibration(self) -> Optional[CalibrationData]:
        """Get calibration data if valid."""
        if self.config.calibration.is_valid():
            return self.config.calibration
        return None


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load()
    return _config_manager


def get_config() -> Config:
    """Get current configuration."""
    return get_config_manager().config
