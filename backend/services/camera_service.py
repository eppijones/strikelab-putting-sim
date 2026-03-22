from __future__ import annotations

import time
from typing import Any, Optional

from ..cameras.base import CameraFrame, CameraType
from ..cameras.legacy_camera_adapter import LegacyCameraAdapter
from .runtime_service import RuntimeService


class CameraService:
    """Expose a single orchestration-facing camera surface."""

    def __init__(self, runtime: RuntimeService):
        self._runtime = runtime

    @property
    def runtime(self):
        return self._runtime.get_app()

    def get_primary_status(self) -> dict[str, Any]:
        sim = self.runtime
        adapter = LegacyCameraAdapter(sim.camera) if sim.camera else None
        age_ms: Optional[float] = None
        if getattr(sim, "_last_arducam_frame_monotonic_ns", None) is not None:
            age_ms = (time.monotonic_ns() - sim._last_arducam_frame_monotonic_ns) / 1e6

        status = {
            "type": "arducam",
            "connected": adapter is not None,
            "running": bool(adapter and adapter.is_running),
            "fps": round(adapter.fps, 1) if adapter else 0.0,
            "resolution": list(adapter.resolution) if adapter else [0, 0],
            "frame_count": getattr(sim, "_frame_id", 0),
        }
        if age_ms is not None:
            status["last_frame_age_ms"] = round(age_ms, 2)
        if adapter:
            status["driver_reported_fps"] = round(adapter.reported_fps, 1)
        return status

    def get_latest_depth_frame(self, camera_type: CameraType) -> Optional[CameraFrame]:
        return self.runtime.camera_manager.get_latest(camera_type)
