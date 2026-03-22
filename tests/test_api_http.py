from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend import legacy_main
from backend.app_factory import create_app
from backend.config import get_config
from backend.services.runtime_service import RuntimeService


class FakeCamera:
    is_running = True
    resolution = (1280, 800)
    fps = 118.5
    reported_fps = 120.0

    def stop(self) -> None:
        self.is_running = False


class FakeRuntime:
    def __init__(self):
        self._running = True
        self.camera_mode = SimpleNamespace(value="replay")
        self.calibrator = SimpleNamespace(is_calibrated=False)
        self.camera = FakeCamera()
        self._startup_phase = "running"
        self._startup_fail_reason = None
        self._startup_sustained_fps = 117.2
        self._startup_arducam_profile_info = "1280x800"
        self._cap_fps = SimpleNamespace(timestamps=[0.0, 1 / 118.5])
        self._current_state = None
        self._ws_clients = set()
        self._ws_lock = asyncio.Lock()
        self.reset_calls = 0

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def get_state_message(self) -> dict:
        return {
            "timestamp_ms": 1.0,
            "state": "ARMED",
            "lane": "IDLE",
            "ball": None,
            "ball_visible": False,
            "velocity": None,
            "prediction": None,
            "virtual_ball": None,
            "shot": None,
            "metrics": {
                "cap_fps": 118.5,
                "proc_fps": 116.0,
                "disp_fps": 60.0,
                "proc_latency_ms": 4.2,
                "idle_stddev": 0.2,
            },
            "calibrated": False,
            "auto_calibrated": False,
            "lens_calibrated": False,
            "pixels_per_meter": 1150.0,
            "overlay_radius_scale": 1.15,
            "resolution": [1280, 800],
            "ready_status": "ready",
            "game": None,
            "session": None,
            "drill": None,
            "multi_camera": None,
        }

    async def add_client(self, websocket) -> None:
        self._ws_clients.add(websocket)

    async def remove_client(self, websocket) -> None:
        self._ws_clients.discard(websocket)

    def reset_all(self) -> None:
        self.reset_calls += 1


def _client() -> tuple[TestClient, FakeRuntime]:
    runtime = FakeRuntime()
    runtime_service = RuntimeService(runtime)
    legacy_main.app_instance = runtime
    app = create_app(runtime_service)
    return TestClient(app), runtime


def test_health_route_reports_ready():
    client, _runtime = _client()
    response = client.get("/api/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["camera_ready"] is True
    assert body["arducam"]["resolution"] == [1280, 800]


def test_status_route_keeps_payload_shape():
    client, _runtime = _client()
    response = client.get("/api/status")
    body = response.json()
    assert response.status_code == 200
    assert body["running"] is True
    assert body["state"]["metrics"]["cap_fps"] == 118.5


def test_config_route_preserves_existing_shape():
    client, _runtime = _client()
    response = client.get("/api/config")
    body = response.json()
    assert response.status_code == 200
    assert "camera" in body
    assert "calibration" in body


def test_tracker_reset_route_calls_runtime():
    client, runtime = _client()
    response = client.post("/api/tracker/reset")
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert runtime.reset_calls == 1


def test_health_config_restores_ws_defaults():
    config = get_config()
    config.ws_v2_enabled = False
    config.ws_protocol_version = 1
    config.ws_v2_delta_enabled = False
