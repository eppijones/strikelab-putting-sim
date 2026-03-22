from __future__ import annotations

from typing import Optional, Protocol, Tuple

from .base import CameraFrame


class PrimaryCameraInterface(Protocol):
    @property
    def is_running(self) -> bool:
        ...

    @property
    def resolution(self) -> Tuple[int, int]:
        ...

    @property
    def fps(self) -> float:
        ...

    @property
    def reported_fps(self) -> float:
        ...

    def start(self) -> bool:
        ...

    def stop(self) -> None:
        ...

    def read_frame(self) -> Optional[CameraFrame]:
        ...
