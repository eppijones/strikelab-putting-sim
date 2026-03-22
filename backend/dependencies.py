from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from .services.camera_service import CameraService
from .services.drill_service import DrillService
from .services.game_service import GameService
from .services.runtime_service import RuntimeService
from .services.session_service import SessionService
from .services.stats_service import StatsService


@dataclass
class AppServices:
    runtime: RuntimeService
    game: GameService
    session: SessionService
    drill: DrillService
    stats: StatsService
    camera: CameraService


def build_services(runtime: RuntimeService) -> AppServices:
    return AppServices(
        runtime=runtime,
        game=GameService(runtime),
        session=SessionService(runtime),
        drill=DrillService(runtime),
        stats=StatsService(runtime),
        camera=CameraService(runtime),
    )


def get_services(request: Request) -> AppServices:
    return request.app.state.services
