from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.ws.broadcaster import BroadcastState, build_broadcast_payload


class BenchmarkRuntime:
    def __init__(self):
        self._frame = 0

    def get_state_message(self) -> dict:
        self._frame += 1
        return {
            "frame_id": self._frame,
            "timestamp_ms": time.time() * 1000,
            "state": "TRACKING" if self._frame % 2 else "ARMED",
            "lane": "CENTER",
            "ball": {"x_px": 100 + self._frame, "y_px": 400, "radius_px": 18, "confidence": 0.98},
            "ball_visible": True,
            "velocity": {"vx_px_s": 1000.0, "vy_px_s": 0.0, "speed_px_s": 1000.0},
            "prediction": None,
            "virtual_ball": None,
            "shot": None,
            "metrics": {
                "cap_fps": 120.0,
                "proc_fps": 118.0,
                "disp_fps": 60.0,
                "proc_latency_ms": 4.1,
                "idle_stddev": 0.1,
            },
            "calibrated": True,
            "auto_calibrated": True,
            "lens_calibrated": True,
            "pixels_per_meter": 1150.0,
            "overlay_radius_scale": 1.15,
            "resolution": [1280, 800],
            "ready_status": "ready",
            "game": {"hole": {"distance_m": 3.0, "position_x_m": 3.0, "position_y_m": 0.0, "radius_m": 0.054}},
            "session": None,
            "drill": None,
            "multi_camera": None,
        }


def main(iterations: int = 500) -> None:
    runtime = BenchmarkRuntime()
    state = BroadcastState()
    encode_times_ms: list[float] = []
    sizes: list[int] = []

    for _ in range(iterations):
        started = time.perf_counter()
        payload = build_broadcast_payload(runtime, state)
        encode_times_ms.append((time.perf_counter() - started) * 1000)
        sizes.append(len(payload.encode("utf-8")))
        json.loads(payload)

    print("WS payload benchmark")
    print(f"iterations: {iterations}")
    print(f"avg_encode_ms: {statistics.mean(encode_times_ms):.3f}")
    print(f"p95_encode_ms: {statistics.quantiles(encode_times_ms, n=20)[18]:.3f}")
    print(f"avg_payload_bytes: {statistics.mean(sizes):.1f}")
    print(f"max_payload_bytes: {max(sizes)}")


if __name__ == "__main__":
    main()
