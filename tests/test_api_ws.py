from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend import legacy_main
from backend.app_factory import create_app
from backend.config import get_config
from backend.services.runtime_service import RuntimeService
from backend.ws.delta_builder import build_delta_message


class FakeRuntime:
    def __init__(self):
        self._running = True
        self.camera_mode = SimpleNamespace(value="replay")
        self.calibrator = SimpleNamespace(is_calibrated=False)
        self.camera = SimpleNamespace(is_running=True, resolution=(1280, 800), fps=120.0, reported_fps=120.0)
        self._startup_phase = "running"
        self._startup_fail_reason = None
        self._startup_sustained_fps = 120.0
        self._startup_arducam_profile_info = "1280x800"
        self._cap_fps = SimpleNamespace(timestamps=[0.0, 1 / 120.0])
        self._current_state = None
        self._ws_clients = set()
        self._ws_lock = asyncio.Lock()
        self._state_counter = 0
        self.reset_calls = 0

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def get_state_message(self) -> dict:
        self._state_counter += 1
        return {
            "timestamp_ms": float(self._state_counter),
            "state": "ARMED",
            "lane": "IDLE",
            "ball": None,
            "ball_visible": False,
            "velocity": None,
            "prediction": None,
            "virtual_ball": None,
            "shot": None,
            "metrics": {
                "cap_fps": 120.0,
                "proc_fps": 119.0,
                "disp_fps": 60.0,
                "proc_latency_ms": 3.8,
                "idle_stddev": 0.1,
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


def test_v1_websocket_ping_and_reset():
    config = get_config()
    config.ws_v2_enabled = False
    config.ws_protocol_version = 1
    config.ws_v2_delta_enabled = False

    client, runtime = _client()
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "ping"}))
        pong = json.loads(ws.receive_text())
        assert pong == {"type": "pong"}

        ws.send_text(json.dumps({"type": "reset"}))
        assert runtime.reset_calls == 1


def test_v2_websocket_sends_snapshot_on_connect():
    config = get_config()
    config.ws_v2_enabled = True
    config.ws_protocol_version = 2
    config.ws_v2_delta_enabled = True

    client, _runtime = _client()
    with client.websocket_connect("/ws") as ws:
        first = json.loads(ws.receive_text())
        assert first["v"] == 2
        assert first["t"] == "snapshot"
        assert first["payload"]["state"] == "ARMED"


def test_delta_builder_only_emits_changes():
    message = build_delta_message(
        {"state": "ARMED", "metrics": {"cap_fps": 120.0, "proc_fps": 119.0}},
        {"state": "TRACKING", "metrics": {"cap_fps": 120.0, "proc_fps": 117.5}},
        seq=2,
    )
    assert message["t"] == "delta"
    assert message["payload"] == {"state": "TRACKING", "metrics": {"proc_fps": 117.5}}
