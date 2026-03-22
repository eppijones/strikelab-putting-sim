"""
Abstract camera interface for multi-camera system.
All camera sources (Arducam, ZED, RealSense) implement this interface.
"""

import numpy as np
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Generator, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class CameraType(Enum):
    ARDUCAM = "arducam"
    ZED = "zed"
    REALSENSE = "realsense"


@dataclass
class CameraFrame:
    """Unified frame container for all camera types."""
    color: np.ndarray                           # BGR color image
    timestamp_ns: int                           # Nanoseconds since epoch
    acquisition_monotonic_ns: int               # Monotonic timestamp at acquisition
    frame_id: int                               # Sequential frame number
    camera_type: CameraType                     # Which camera produced this
    depth: Optional[np.ndarray] = None          # Depth map (meters, float32) - ZED/RS only
    point_cloud: Optional[np.ndarray] = None    # XYZ point cloud (N,3) - ZED only
    confidence: Optional[np.ndarray] = None     # Depth confidence map - ZED only

    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1_000_000

    @property
    def has_depth(self) -> bool:
        return self.depth is not None


@dataclass
class CameraStatus:
    """Health/status for a camera."""
    camera_type: CameraType
    connected: bool = False
    running: bool = False
    fps: float = 0.0
    resolution: Tuple[int, int] = (0, 0)
    error: Optional[str] = None
    frame_count: int = 0
    last_frame_age_ms: Optional[float] = None  # time since last delivered frame

    def to_dict(self) -> dict:
        d = {
            "type": self.camera_type.value,
            "connected": self.connected,
            "running": self.running,
            "fps": round(self.fps, 1),
            "resolution": list(self.resolution),
            "error": self.error,
            "frame_count": self.frame_count,
        }
        if self.last_frame_age_ms is not None:
            d["last_frame_age_ms"] = round(self.last_frame_age_ms, 2)
        return d


class FPSCounter:
    """Lightweight FPS tracker using a sliding window."""

    def __init__(self, window: int = 30):
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.time()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / dt if dt > 0 else 0.0

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / dt if dt > 0 else 0.0


class CameraSource(ABC):
    """Abstract base for all camera sources."""

    def __init__(self, camera_type: CameraType):
        self.camera_type = camera_type
        self._running = False
        self._frame_id = 0
        self._fps_counter = FPSCounter()
        self._error: Optional[str] = None
        self._last_frame_monotonic_ns: Optional[int] = None

    @abstractmethod
    def start(self) -> bool:
        """Initialize hardware and begin capture. Returns True on success."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Release hardware resources."""
        ...

    @abstractmethod
    def read_frame(self) -> Optional[CameraFrame]:
        """Read one frame. Returns None when unavailable."""
        ...

    @abstractmethod
    def resolution(self) -> Tuple[int, int]:
        """Current (width, height)."""
        ...

    def frames(self) -> Generator[CameraFrame, None, None]:
        """Yield frames until stopped."""
        while self._running:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def fps(self) -> float:
        return self._fps_counter.fps

    @property
    def frame_count(self) -> int:
        return self._frame_id

    def mark_frame_received(self) -> None:
        """Call after a successful read_frame() for health / last_frame_age_ms."""
        self._last_frame_monotonic_ns = time.monotonic_ns()

    def status(self) -> CameraStatus:
        age_ms: Optional[float] = None
        if self._last_frame_monotonic_ns is not None:
            age_ms = (time.monotonic_ns() - self._last_frame_monotonic_ns) / 1e6
        return CameraStatus(
            camera_type=self.camera_type,
            connected=self._running,
            running=self._running,
            fps=self.fps,
            resolution=self.resolution(),
            error=self._error,
            frame_count=self._frame_id,
            last_frame_age_ms=age_ms,
        )
