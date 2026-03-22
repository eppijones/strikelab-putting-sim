"""
Camera abstraction for StrikeLab Putting Sim.
Supports Arducam OV9281 UVC capture at 120fps and video file replay.
"""

import cv2
import numpy as np
import sys
import time
import logging
from typing import Optional, Tuple, Generator, List, Dict, Any
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
    acquisition_monotonic_ns: int  # Monotonic timestamp at acquisition
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


ARDUCAM_MIN_SUSTAINED_FPS = 25.0
ARDUCAM_PROFILE_PROBE_TIMEOUT_S = 3.0


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

        # Post-selection sustained FPS measured during warmup
        self._sustained_startup_fps: float = 0.0
        
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
        """
        Start Arducam OV9281 capture at 120fps.

        On Windows the Arducam is typically opened via DirectShow (DSHOW), but
        the DSHOW UVC driver for this camera ignores programmatic exposure
        control via OpenCV, leaving auto-exposure at ~93 ms and capping the
        frame rate to ~10 fps.  The Windows Camera app uses Media Foundation
        (MSMF) which handles exposure correctly.

        Strategy:
          1. DSHOW — open once, measure FPS, try exposure overrides.
          2. If DSHOW FPS < 30, close it and scan MSMF indices 0-5 for a
             1280x800 non-stereo stream (MSMF enumerates devices in a
             different order than DSHOW).
          3. Use whichever backend gave the best FPS.
        """
        logger.info(f"Starting Arducam OV9281 on device {self.device_id}")

        if sys.platform == "win32":
            primary_backend = cv2.CAP_DSHOW
        elif sys.platform == "darwin":
            primary_backend = cv2.CAP_AVFOUNDATION
        else:
            primary_backend = cv2.CAP_V4L2

        # --- Phase 1: Try primary backend (DSHOW on Windows) ---
        cap, best_fps, fourcc_text = self._try_open_and_measure(
            primary_backend, self.device_id, "DSHOW",
        )

        if cap is not None and best_fps >= 30:
            return self._finalise_arducam(cap, best_fps, fourcc_text)

        # --- Phase 2 (Windows only): MSMF fallback scan ---
        if sys.platform == "win32" and (cap is None or best_fps < 30):
            dshow_fps = best_fps
            if cap is not None:
                cap.release()
                cap = None
                time.sleep(0.4)  # let driver fully release before MSMF opens
            logger.info(
                "DSHOW FPS too low (%.1f). Scanning MSMF backend for Arducam...",
                dshow_fps,
            )
            for msmf_idx in range(6):
                m_cap, m_fps, m_fourcc = self._try_open_and_measure(
                    cv2.CAP_MSMF, msmf_idx, f"MSMF[{msmf_idx}]",
                )
                if m_cap is None:
                    continue
                if m_fps > dshow_fps:
                    logger.info(
                        "MSMF device %d better: %.1f fps (vs DSHOW %.1f)",
                        msmf_idx, m_fps, dshow_fps,
                    )
                    return self._finalise_arducam(m_cap, m_fps, m_fourcc)
                m_cap.release()

            # MSMF didn't help — reopen DSHOW so we at least have a handle.
            logger.info("MSMF scan did not improve FPS. Reopening DSHOW.")
            cap, best_fps, fourcc_text = self._try_open_and_measure(
                primary_backend, self.device_id, "DSHOW(reopen)",
            )

        # --- Phase 3: non-Windows or CAP_ANY fallback ---
        if cap is None:
            cap, best_fps, fourcc_text = self._try_open_and_measure(
                cv2.CAP_ANY, self.device_id, "ANY",
            )

        if cap is None:
            logger.error("Failed to open Arducam on any backend")
            return False

        return self._finalise_arducam(cap, best_fps, fourcc_text)

    # ------------------------------------------------------------------
    def _try_open_and_measure(
        self, backend: int, device_id: int, label: str,
    ) -> Tuple[Optional[cv2.VideoCapture], float, str]:
        """
        Open *device_id* with *backend*, verify 1280x800, measure FPS,
        and try exposure overrides.  Returns (cap, best_fps, fourcc_text)
        or (None, 0, "") on failure.
        """
        cap = cv2.VideoCapture(device_id, backend)
        if not cap.isOpened():
            return (None, 0.0, "")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ARDUCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.ARDUCAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.ARDUCAM_FPS)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if self._is_stereo_sbs(w, h) or not self._is_expected_resolution(w, h):
            cap.release()
            return (None, 0.0, "")

        fourcc_text = self._fourcc_to_text(int(cap.get(cv2.CAP_PROP_FOURCC)))
        reported = cap.get(cv2.CAP_PROP_FPS)

        fps = self._measure_capture_fps(cap, duration_s=0.8)
        logger.info(
            "%s device %d: fourcc=%s, res=%.0fx%.0f, reported=%.0ffps, measured=%.1ffps",
            label, device_id, fourcc_text, w, h, reported, fps,
        )

        if fps < 30:
            exposure_overrides = [
                (0.25, -7, "ae=0.25,ev=-7"),
                (0.25, -9, "ae=0.25,ev=-9"),
                (1, -7, "ae=1,ev=-7"),
                (3, -7, "ae=3,ev=-7"),
            ]
            for ae, ev, desc in exposure_overrides:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ae)
                cap.set(cv2.CAP_PROP_EXPOSURE, ev)
                time.sleep(0.1)
                t = self._measure_capture_fps(cap, duration_s=0.35)
                logger.info("  %s exposure(%s): %.1f fps", label, desc, t)
                if t > fps:
                    fps = t
                if fps >= 30:
                    break

        return (cap, fps, fourcc_text)

    # ------------------------------------------------------------------
    def _finalise_arducam(
        self, cap: cv2.VideoCapture, fps: float, fourcc_text: str,
    ) -> bool:
        self._cap = cap
        reported = cap.get(cv2.CAP_PROP_FPS)
        self._reported_fps = reported if reported > 0 else fps
        self._sustained_startup_fps = fps
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        logger.info(
            "Camera selected: fourcc=%s, measured_fps=%.1f, reported_fps=%.1f, res=%.0fx%.0f",
            fourcc_text, fps, self._reported_fps, w, h,
        )
        if fps < ARDUCAM_MIN_SUSTAINED_FPS:
            logger.warning(
                "Arducam FPS (%.1f) below target (%.1f). Tracking will work but at reduced temporal resolution.",
                fps, ARDUCAM_MIN_SUSTAINED_FPS,
            )

        # DSHOW ignores CAP_PROP_EXPOSURE on this camera, but CAP_PROP_GAIN
        # and CAP_PROP_BRIGHTNESS may work.  Try boosting gain to brighten
        # the image without affecting shutter speed / fps.
        if fps >= 100:
            self._try_boost_brightness(cap, fps)

        self._running = True
        self._frame_id = 0
        return True

    def _try_boost_brightness(
        self, cap: cv2.VideoCapture, baseline_fps: float,
    ) -> None:
        """Best-effort brightness increase via gain/brightness props."""
        MIN_SAFE_FPS = 110.0
        props_to_try = [
            (cv2.CAP_PROP_GAIN, [32, 48, 64], "gain"),
            (cv2.CAP_PROP_BRIGHTNESS, [140, 160, 180], "brightness"),
        ]
        for prop, values, label in props_to_try:
            orig = cap.get(prop)
            applied = False
            for val in values:
                cap.set(prop, val)
                time.sleep(0.1)
                actual = cap.get(prop)
                if abs(actual - val) > 1:
                    continue
                measured = self._measure_capture_fps(cap, duration_s=0.4)
                logger.info("  Brightness probe (%s=%d): %.1f fps", label, val, measured)
                if measured >= MIN_SAFE_FPS:
                    applied = True
                else:
                    cap.set(prop, orig)
                    break
            if applied:
                logger.info("Brightness boost applied via %s", label)
                return
            else:
                cap.set(prop, orig)
        logger.info("DSHOW brightness: driver did not accept gain/brightness adjustments")

    @staticmethod
    def _measure_capture_fps(cap: cv2.VideoCapture, duration_s: float = 0.7) -> float:
        # Let driver settle exposure/frame pacing before timing.
        warmup_end = time.perf_counter() + 0.25
        while time.perf_counter() < warmup_end:
            cap.read()
        start = time.perf_counter()
        end = start + max(0.3, duration_s)
        frames = 0
        while time.perf_counter() < end:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frames += 1
        elapsed = max(1e-6, time.perf_counter() - start)
        return frames / elapsed

    @staticmethod
    def _is_expected_resolution(width: float, height: float) -> bool:
        return abs(width - 1280.0) <= 64.0 and abs(height - 800.0) <= 64.0

    @staticmethod
    def _is_stereo_sbs(width: float, height: float) -> bool:
        """Detect side-by-side stereo (ZED via UVC) — aspect ratio >= 2.5."""
        if height <= 0:
            return False
        return (width / height) >= 2.5

    @staticmethod
    def _fourcc_to_text(fourcc: int) -> str:
        try:
            return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        except Exception:
            return "????"
    
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
        if len(frame.shape) == 2:
            # Monochrome formats (e.g. Y800) should still flow through BGR detector pipeline.
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Generate timestamp
        if self.mode == CameraMode.REPLAY:
            # For replay, use synthetic timestamps based on frame number
            timestamp_ns = self._frame_id * self._replay_frame_duration_ns
        else:
            # For live capture, use actual system time
            timestamp_ns = time.time_ns()
        acquisition_monotonic_ns = time.monotonic_ns()
        
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
            acquisition_monotonic_ns=acquisition_monotonic_ns,
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
