"""
Multi-camera system for StrikeLab Putting Sim.
Supports Arducam (2D), ZED 2i (stereo depth), and Intel RealSense D455 (depth).
"""

from .base import CameraSource, CameraFrame, CameraType, CameraStatus
from .camera_manager import CameraManager

__all__ = [
    "CameraSource",
    "CameraFrame",
    "CameraType",
    "CameraStatus",
    "CameraManager",
]
