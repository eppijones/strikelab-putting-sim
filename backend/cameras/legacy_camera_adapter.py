from __future__ import annotations

from typing import Optional, Tuple

from ..camera import Camera, CameraMode
from .frame_adapter import frame_data_to_camera_frame
from .interfaces import PrimaryCameraInterface


class LegacyCameraAdapter(PrimaryCameraInterface):
    """Adapter that exposes the protected Camera through the unified camera surface."""

    def __init__(self, camera: Optional[Camera] = None, *, mode: CameraMode = CameraMode.ARDUCAM, replay_path: Optional[str] = None, device_id: int = 0):
        self._camera = camera or Camera(mode=mode, replay_path=replay_path, device_id=device_id)

    @property
    def inner_camera(self) -> Camera:
        return self._camera

    @property
    def is_running(self) -> bool:
        return self._camera.is_running

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._camera.resolution

    @property
    def fps(self) -> float:
        return self._camera.fps

    @property
    def reported_fps(self) -> float:
        return self._camera.reported_fps

    def start(self) -> bool:
        return self._camera.start()

    def stop(self) -> None:
        self._camera.stop()

    def read_frame(self):
        frame_data = self._camera.read_frame()
        if frame_data is None:
            return None
        return frame_data_to_camera_frame(frame_data)
