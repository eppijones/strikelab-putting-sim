from __future__ import annotations

from typing import Any


def run_capture_loop(sim: Any) -> None:
    sim._capture_loop()


def run_process_loop(sim: Any) -> None:
    sim._process_loop()
