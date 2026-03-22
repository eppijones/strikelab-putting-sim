from __future__ import annotations

from typing import Any


def start_depth_cameras(sim: Any) -> None:
    sim._start_depth_cameras()


def start_depth_reconnect_worker(sim: Any) -> None:
    sim._start_depth_reconnect_worker()
