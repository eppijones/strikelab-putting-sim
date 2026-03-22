"""
Multi-camera orchestrator.
Manages Arducam, ZED 2i, and RealSense D455 cameras with independent capture
threads, time synchronisation, and graceful degradation.
"""

import threading
import time
import logging
from typing import Optional, Dict, List, Callable, Any
from collections import deque
from dataclasses import dataclass

from .base import CameraSource, CameraFrame, CameraType, CameraStatus
from .arducam_source import ArducamSource
from .zed_source import ZedSource, ZED_AVAILABLE
from .realsense_source import RealSenseSource, REALSENSE_AVAILABLE

logger = logging.getLogger(__name__)

SYNC_TOLERANCE_NS = 20_000_000  # 20ms tolerance for frame sync


@dataclass
class SyncedFrameSet:
    """Time-synchronized frame set from all active cameras."""
    arducam: Optional[CameraFrame] = None
    zed: Optional[CameraFrame] = None
    realsense: Optional[CameraFrame] = None
    timestamp_ns: int = 0

    @property
    def has_depth(self) -> bool:
        return self.zed is not None or self.realsense is not None


class CameraManager:
    """
    Orchestrates all cameras, each running in its own capture thread.

    Supports graceful degradation:
    - Arducam only        -> existing behaviour
    - Arducam + ZED       -> adds club analytics, 3D ball
    - Arducam + RealSense -> adds launch angle
    - All three           -> full TrackMan-style analytics
    """

    def __init__(
        self,
        enable_arducam: bool = True,
        enable_zed: bool = True,
        enable_realsense: bool = True,
        arducam_device_id: int = 0,
        arducam_replay_path: Optional[str] = None,
        zed_serial: int = 0,
        realsense_serial: Optional[str] = None,
        zed_settings: Optional[Dict[str, Any]] = None,
        realsense_settings: Optional[Dict[str, Any]] = None,
    ):
        self._sources: Dict[CameraType, CameraSource] = {}
        self._threads: Dict[CameraType, threading.Thread] = {}
        self._running = False

        # Latest frame from each camera (thread-safe via locks)
        self._latest: Dict[CameraType, Optional[CameraFrame]] = {}
        self._locks: Dict[CameraType, threading.Lock] = {}

        # Frame callback for each camera type
        self._callbacks: Dict[CameraType, List[Callable[[CameraFrame], None]]] = {}

        # Ring buffers for time-sync lookups
        self._buffers: Dict[CameraType, deque] = {}
        self._buffer_size = 30  # ~0.5s at 60fps

        if enable_arducam:
            src = ArducamSource(
                device_id=arducam_device_id,
                replay_path=arducam_replay_path,
            )
            self._register(src)

        if enable_zed and ZED_AVAILABLE:
            zed_kwargs = dict(zed_settings or {})
            zed_kwargs.setdefault("serial_number", zed_serial)
            src = ZedSource(**zed_kwargs)
            self._register(src)
        elif enable_zed and not ZED_AVAILABLE:
            logger.warning("ZED requested but pyzed not installed – skipping")

        if enable_realsense and REALSENSE_AVAILABLE:
            rs_kwargs = dict(realsense_settings or {})
            rs_kwargs.setdefault("serial_number", realsense_serial)
            src = RealSenseSource(**rs_kwargs)
            self._register(src)
        elif enable_realsense and not REALSENSE_AVAILABLE:
            logger.warning("RealSense requested but pyrealsense2 not installed – skipping")

    def _register(self, source: CameraSource) -> None:
        ct = source.camera_type
        self._sources[ct] = source
        self._latest[ct] = None
        self._locks[ct] = threading.Lock()
        self._callbacks[ct] = []
        self._buffers[ct] = deque(maxlen=self._buffer_size)

    def on_frame(self, camera_type: CameraType, callback: Callable[[CameraFrame], None]) -> None:
        """Register a per-frame callback for a specific camera."""
        if camera_type in self._callbacks:
            self._callbacks[camera_type].append(callback)

    def start(self) -> Dict[CameraType, bool]:
        """Start all registered cameras. Returns {type: success}."""
        return self.start_selected(list(self._sources.keys()))

    def start_selected(self, camera_types: List[CameraType]) -> Dict[CameraType, bool]:
        """Start selected cameras only. Returns {type: success}."""
        results: Dict[CameraType, bool] = {}
        if not camera_types:
            return results
        self._running = True

        for ct in camera_types:
            source = self._sources.get(ct)
            if source is None:
                continue
            if source.is_running:
                results[ct] = True
                continue
            ok = source.start()
            results[ct] = ok
            if ok:
                t = threading.Thread(
                    target=self._capture_loop,
                    args=(ct,),
                    daemon=True,
                    name=f"cam-{ct.value}",
                )
                self._threads[ct] = t
                t.start()
                logger.info(f"{ct.value} capture thread started")
            else:
                logger.error(f"Failed to start {ct.value}: {source._error}")

        return results

    def stop(self) -> None:
        """Stop all cameras and join threads."""
        self._running = False
        for ct, source in self._sources.items():
            source.stop()
        for ct, t in self._threads.items():
            t.join(timeout=3.0)
        self._threads.clear()
        logger.info("CameraManager stopped")

    def _capture_loop(self, camera_type: CameraType) -> None:
        """Per-camera capture loop running in its own thread."""
        source = self._sources[camera_type]
        while self._running and source.is_running:
            frame = source.read_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            source.mark_frame_received()

            with self._locks[camera_type]:
                self._latest[camera_type] = frame

            self._buffers[camera_type].append(frame)

            for cb in self._callbacks[camera_type]:
                try:
                    cb(frame)
                except Exception as e:
                    logger.error(f"Callback error for {camera_type.value}: {e}")

    # --- Accessors ---

    def get_latest(self, camera_type: CameraType) -> Optional[CameraFrame]:
        """Get the most recent frame from a camera (thread-safe)."""
        lock = self._locks.get(camera_type)
        if lock is None:
            return None
        with lock:
            return self._latest.get(camera_type)

    def get_synced_frames(self, reference_ns: int) -> SyncedFrameSet:
        """
        Find frames closest to *reference_ns* from each camera.
        Uses the ring buffers and SYNC_TOLERANCE_NS.
        """
        result = SyncedFrameSet(timestamp_ns=reference_ns)

        for ct, buf in self._buffers.items():
            if not buf:
                continue
            best: Optional[CameraFrame] = None
            best_dt = float("inf")
            for frame in buf:
                dt = abs(frame.timestamp_ns - reference_ns)
                if dt < best_dt:
                    best_dt = dt
                    best = frame
            if best is not None and best_dt <= SYNC_TOLERANCE_NS:
                if ct == CameraType.ARDUCAM:
                    result.arducam = best
                elif ct == CameraType.ZED:
                    result.zed = best
                elif ct == CameraType.REALSENSE:
                    result.realsense = best

        return result

    def get_source(self, camera_type: CameraType) -> Optional[CameraSource]:
        return self._sources.get(camera_type)

    def statuses(self) -> Dict[CameraType, CameraStatus]:
        return {ct: src.status() for ct, src in self._sources.items()}

    @property
    def has_arducam(self) -> bool:
        s = self._sources.get(CameraType.ARDUCAM)
        return s is not None and s.is_running

    @property
    def has_zed(self) -> bool:
        s = self._sources.get(CameraType.ZED)
        return s is not None and s.is_running

    @property
    def has_realsense(self) -> bool:
        s = self._sources.get(CameraType.REALSENSE)
        return s is not None and s.is_running

    @property
    def active_cameras(self) -> List[CameraType]:
        return [ct for ct, src in self._sources.items() if src.is_running]
