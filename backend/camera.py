"""
Camera abstraction for StrikeLab Putting Sim.
Supports Arducam OV9281 UVC capture at 120fps and video file replay.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Generator
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class CameraMode(Enum):
    ARDUCAM = "arducam"
    REPLAY = "replay"
    WEBCAM = "webcam"  # Fallback for testing


@dataclass
class FrameData:
    """Container for frame data with timing information."""
    frame: np.ndarray
    timestamp_ns: int  # Nanoseconds since epoch
    frame_id: int
    
    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1_000_000


@dataclass
class FPSStats:
    """Frame timing statistics."""
    fps: float = 0.0
    dt_mean_ms: float = 0.0
    dt_std_ms: float = 0.0
    dt_min_ms: float = 0.0
    dt_max_ms: float = 0.0
    sample_count: int = 0


class FPSTracker:
    """Track actual FPS using timestamp deltas."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps: deque[int] = deque(maxlen=window_size)
        self._last_fps = 0.0
        
    def update(self, timestamp_ns: int) -> float:
        """Update with new timestamp, return current FPS estimate."""
        self.timestamps.append(timestamp_ns)
        
        if len(self.timestamps) < 2:
            return 0.0
            
        dt_ns = self.timestamps[-1] - self.timestamps[0]
        if dt_ns <= 0:
            return self._last_fps
            
        fps = (len(self.timestamps) - 1) / (dt_ns / 1e9)
        self._last_fps = fps
        return fps
    
    def get_stats(self) -> FPSStats:
        """Get detailed timing statistics."""
        if len(self.timestamps) < 2:
            return FPSStats()
        
        # Calculate dt between consecutive frames
        timestamps = list(self.timestamps)
        dts_ms = []
        for i in range(1, len(timestamps)):
            dt_ms = (timestamps[i] - timestamps[i-1]) / 1e6
            if dt_ms > 0:
                dts_ms.append(dt_ms)
        
        if not dts_ms:
            return FPSStats()
        
        dts = np.array(dts_ms)
        fps = 1000.0 / np.mean(dts) if np.mean(dts) > 0 else 0
        
        return FPSStats(
            fps=fps,
            dt_mean_ms=float(np.mean(dts)),
            dt_std_ms=float(np.std(dts)),
            dt_min_ms=float(np.min(dts)),
            dt_max_ms=float(np.max(dts)),
            sample_count=len(dts)
        )
    
    @property
    def current_fps(self) -> float:
        """Get current FPS estimate."""
        return self._last_fps


class Camera:
    """
    Camera abstraction supporting multiple capture modes.
    
    Usage:
        cam = Camera(mode=CameraMode.ARDUCAM)
        cam.start()
        for frame_data in cam.frames():
            process(frame_data)
        cam.stop()
    """
    
    # Arducam OV9281 configuration
    ARDUCAM_WIDTH = 1280
    ARDUCAM_HEIGHT = 800
    ARDUCAM_FPS = 120
    
    def __init__(
        self,
        mode: CameraMode = CameraMode.ARDUCAM,
        replay_path: Optional[str] = None,
        device_id: int = 0
    ):
        self.mode = mode
        self.replay_path = replay_path
        self.device_id = device_id
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_id = 0
        self._running = False
        self._fps_tracker = FPSTracker()
        
        # For replay mode: track video FPS for proper timing
        self._replay_fps = 120.0
        self._replay_frame_duration_ns = int(1e9 / 120)
        
        # Actual camera FPS (as reported by driver)
        self._reported_fps: float = 120.0
        
        # Log FPS stats periodically
        self._last_fps_log_time = 0.0
        self._fps_log_interval = 10.0  # Log every 10 seconds
        
    def start(self) -> bool:
        """Initialize and start camera capture."""
        if self.mode == CameraMode.ARDUCAM:
            return self._start_arducam()
        elif self.mode == CameraMode.REPLAY:
            return self._start_replay()
        elif self.mode == CameraMode.WEBCAM:
            return self._start_webcam()
        return False
    
    def _start_arducam(self) -> bool:
        """Start Arducam OV9281 capture at 120fps."""
        logger.info(f"Starting Arducam OV9281 on device {self.device_id}")
        
        # Try V4L2 backend first (Linux), then default
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for backend in backends:
            self._cap = cv2.VideoCapture(self.device_id, backend)
            if self._cap.isOpened():
                break
        
        if not self._cap or not self._cap.isOpened():
            logger.error("Failed to open camera")
            return False
        
        # Configure for 120fps capture
        # MJPG format typically required for high FPS
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ARDUCAM_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.ARDUCAM_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, self.ARDUCAM_FPS)
        
        # Disable auto exposure for consistent lighting
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode
        self._cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Fast exposure for 120fps
        
        # Verify settings
        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        
        self._reported_fps = actual_fps if actual_fps > 0 else 120.0
        
        logger.info(f"Camera configured: {actual_w}x{actual_h} @ {actual_fps}fps")
        
        if actual_fps < 100:
            logger.warning(f"Camera FPS ({actual_fps}) lower than expected 120fps - "
                          f"velocity calculations will use actual timestamps")
        
        self._running = True
        self._frame_id = 0
        return True
    
    def _start_replay(self) -> bool:
        """Start video file replay."""
        if not self.replay_path:
            logger.error("Replay mode requires replay_path")
            return False
            
        logger.info(f"Starting replay from: {self.replay_path}")
        
        self._cap = cv2.VideoCapture(self.replay_path)
        if not self._cap.isOpened():
            logger.error(f"Failed to open video file: {self.replay_path}")
            return False
        
        # Get video properties
        self._replay_fps = self._cap.get(cv2.CAP_PROP_FPS) or 120.0
        self._replay_frame_duration_ns = int(1e9 / self._replay_fps)
        self._reported_fps = self._replay_fps
        
        width = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_count = self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        logger.info(f"Video: {width}x{height} @ {self._replay_fps}fps, {frame_count} frames")
        
        self._running = True
        self._frame_id = 0
        return True
    
    def _start_webcam(self) -> bool:
        """Start standard webcam (fallback for testing)."""
        logger.info(f"Starting webcam on device {self.device_id}")
        
        self._cap = cv2.VideoCapture(self.device_id)
        if not self._cap.isOpened():
            logger.error("Failed to open webcam")
            return False
        
        # Set reasonable resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self._reported_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        self._running = True
        self._frame_id = 0
        return True
    
    def stop(self):
        """Stop camera capture and release resources."""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped")
    
    def read_frame(self) -> Optional[FrameData]:
        """
        Read a single frame with timestamp.
        Returns None if no frame available or camera stopped.
        """
        if not self._running or not self._cap:
            return None
        
        ret, frame = self._cap.read()
        if not ret or frame is None:
            if self.mode == CameraMode.REPLAY:
                logger.info("Replay finished")
                self._running = False
            return None
        
        # Generate timestamp
        if self.mode == CameraMode.REPLAY:
            # For replay, use synthetic timestamps based on frame number
            timestamp_ns = self._frame_id * self._replay_frame_duration_ns
        else:
            # For live capture, use actual system time
            timestamp_ns = time.time_ns()
        
        self._frame_id += 1
        
        # Update FPS tracker
        self._fps_tracker.update(timestamp_ns)
        
        # Log FPS stats periodically
        now = time.time()
        if now - self._last_fps_log_time > self._fps_log_interval:
            self._last_fps_log_time = now
            stats = self._fps_tracker.get_stats()
            if stats.sample_count > 0:
                logger.debug(f"Camera timing: effective_fps={stats.fps:.1f}, "
                           f"dt={stats.dt_mean_ms:.2f}±{stats.dt_std_ms:.2f}ms, "
                           f"range=[{stats.dt_min_ms:.2f}, {stats.dt_max_ms:.2f}]ms")
        
        return FrameData(
            frame=frame,
            timestamp_ns=timestamp_ns,
            frame_id=self._frame_id
        )
    
    def frames(self) -> Generator[FrameData, None, None]:
        """Generator that yields frames until stopped or end of video."""
        while self._running:
            frame_data = self.read_frame()
            if frame_data is None:
                break
            yield frame_data
    
    @property
    def fps(self) -> float:
        """Get current measured FPS (from actual timestamps)."""
        return self._fps_tracker.current_fps
    
    @property
    def reported_fps(self) -> float:
        """Get FPS as reported by camera driver."""
        return self._reported_fps
    
    @property
    def fps_stats(self) -> FPSStats:
        """Get detailed FPS statistics."""
        return self._fps_tracker.get_stats()
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def frame_count(self) -> int:
        return self._frame_id
    
    @property
    def resolution(self) -> Tuple[int, int]:
        """Get current resolution (width, height)."""
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)
