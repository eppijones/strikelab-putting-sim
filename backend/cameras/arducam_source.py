"""
Arducam OV9281 camera source - wraps existing camera.py Camera class.
Includes auto-detection to find the correct device index when multiple
cameras (ZED 2i, RealSense) are connected and may steal lower indices.
"""

import cv2
import logging
from typing import Optional, Tuple, List

from .base import CameraSource, CameraFrame, CameraType
from ..camera import Camera, CameraMode

logger = logging.getLogger(__name__)

ARDUCAM_EXPECTED_WIDTH = 1280
ARDUCAM_EXPECTED_HEIGHT = 800
MAX_PROBE_DEVICES = 8


def _probe_device(device_id: int, backend: int = cv2.CAP_DSHOW) -> Optional[Tuple[int, int]]:
    """
    Open a device, request 1280x800, read a frame, and return its
    actual (width, height).  Returns None if the device can't be opened.
    """
    cap = cv2.VideoCapture(device_id, backend)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ARDUCAM_EXPECTED_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ARDUCAM_EXPECTED_HEIGHT)
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            return (w, h)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)
    finally:
        cap.release()


def _is_arducam_resolution(width: int, height: int) -> bool:
    """Check if resolution is consistent with Arducam OV9281 (1280x800)."""
    return (abs(width - ARDUCAM_EXPECTED_WIDTH) <= 64 and
            abs(height - ARDUCAM_EXPECTED_HEIGHT) <= 64)


def _is_stereo_sbs(width: int, height: int) -> bool:
    """Detect side-by-side stereo (ZED via UVC) — width ≥ 2.5x height."""
    if height == 0:
        return False
    aspect = width / height
    return aspect >= 2.5


def find_arducam_device(preferred_id: int = 0, verbose: bool = False) -> int:
    """
    Auto-detect the Arducam device index by scanning all devices and
    picking the one that matches OV9281 resolution (1280x800) while
    rejecting stereo side-by-side cameras (ZED 2i).

    Scoring: exact 1280x800 > close match.  Tie-break favours preferred_id,
    then lowest index for deterministic behaviour across restarts.
    Returns preferred_id only if no candidate is found.
    """
    import sys
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    _log = (lambda msg: print(msg)) if verbose else (lambda msg: logger.info(msg))

    _log(f"Scanning camera devices 0-{MAX_PROBE_DEVICES - 1} ...")
    candidates: List[Tuple[int, int, int, int]] = []  # (score, idx, w, h)

    for idx in range(MAX_PROBE_DEVICES):
        res = _probe_device(idx, backend)
        if res is None:
            continue
        w, h = res
        tag = ""
        if _is_stereo_sbs(w, h):
            tag = " [stereo SBS — skipped]"
        elif _is_arducam_resolution(w, h):
            exact = (w == ARDUCAM_EXPECTED_WIDTH and h == ARDUCAM_EXPECTED_HEIGHT)
            score = 100 if exact else 50
            if idx == preferred_id:
                score += 10
            tag = f" [Arducam match, score={score}]"
            candidates.append((score, idx, w, h))
        else:
            tag = " [non-Arducam]"
        _log(f"  Device {idx}: {w}x{h}{tag}")

    if candidates:
        candidates.sort(key=lambda t: (-t[0], t[1]))
        chosen_score, chosen_idx, chosen_w, chosen_h = candidates[0]
        _log(f"Arducam auto-detected at device {chosen_idx} ({chosen_w}x{chosen_h}, score={chosen_score})")
        return chosen_idx

    _log("WARNING: Could not auto-detect Arducam by resolution.")
    _log(f"Falling back to preferred device {preferred_id}")
    return preferred_id


class ArducamSource(CameraSource):
    """
    Wraps the existing Camera class to conform to CameraSource interface.
    The Arducam remains the primary 2D tracker at 120fps.

    On start(), auto-detects the correct device index so that stereo cameras
    (ZED 2i) that steal lower USB indices don't hijack the Arducam feed.
    """

    def __init__(self, device_id: int = 0, replay_path: Optional[str] = None):
        super().__init__(CameraType.ARDUCAM)
        self._device_id = device_id
        self._replay_path = replay_path
        self._camera: Optional[Camera] = None

        if replay_path:
            self._mode = CameraMode.REPLAY
        else:
            self._mode = CameraMode.ARDUCAM

    def start(self) -> bool:
        if self._mode != CameraMode.REPLAY:
            self._device_id = find_arducam_device(self._device_id)

        self._camera = Camera(
            mode=self._mode,
            replay_path=self._replay_path,
            device_id=self._device_id,
        )
        success = self._camera.start()
        if success:
            w, h = self._camera.resolution
            if not _is_arducam_resolution(w, h) and self._mode != CameraMode.REPLAY:
                logger.warning(
                    f"Opened device {self._device_id} but resolution {w}x{h} "
                    f"doesn't match Arducam — camera feeds may be swapped"
                )
            self._running = True
            logger.info(f"ArducamSource started on device {self._device_id}")
        else:
            self._error = "Failed to open Arducam"
            logger.error(self._error)
        return success

    def stop(self) -> None:
        self._running = False
        if self._camera:
            self._camera.stop()
            self._camera = None
        logger.info("ArducamSource stopped")

    def read_frame(self) -> Optional[CameraFrame]:
        if not self._running or not self._camera:
            return None

        frame_data = self._camera.read_frame()
        if frame_data is None:
            return None

        self._frame_id += 1
        self._fps_counter.tick()

        return CameraFrame(
            color=frame_data.frame,
            timestamp_ns=frame_data.timestamp_ns,
            acquisition_monotonic_ns=frame_data.acquisition_monotonic_ns,
            frame_id=frame_data.frame_id,
            camera_type=CameraType.ARDUCAM,
        )

    def resolution(self) -> Tuple[int, int]:
        if self._camera:
            return self._camera.resolution
        return (1280, 800)

    @property
    def inner_camera(self) -> Optional[Camera]:
        """Access the underlying Camera for legacy code that needs it."""
        return self._camera
