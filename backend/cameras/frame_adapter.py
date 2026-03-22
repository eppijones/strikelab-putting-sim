from __future__ import annotations

from ..camera import FrameData
from .base import CameraFrame, CameraType


def frame_data_to_camera_frame(frame_data: FrameData) -> CameraFrame:
    return CameraFrame(
        color=frame_data.frame,
        timestamp_ns=frame_data.timestamp_ns,
        acquisition_monotonic_ns=frame_data.acquisition_monotonic_ns,
        frame_id=frame_data.frame_id,
        camera_type=CameraType.ARDUCAM,
    )
