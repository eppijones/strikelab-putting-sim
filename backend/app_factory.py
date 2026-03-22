from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import legacy_main
from .dependencies import build_services
from .routers.calibration import router as calibration_router
from .routers.config import router as config_router
from .routers.diagnostics import router as diagnostics_router
from .routers.drills import router as drills_router
from .routers.game import router as game_router
from .routers.health import router as health_router
from .routers.session import router as session_router
from .routers.stats import router as stats_router
from .routers.validation import router as validation_router
from .routers.video import router as video_router
from .services.runtime_service import RuntimeService
from .ws.broadcaster import BroadcastState, broadcast_state
from .ws.endpoint import create_ws_router


def _resolve_frontend_path() -> Path:
    frontend_path = Path(__file__).parent.parent / "frontend" / "dist"
    if frontend_path.exists():
        return frontend_path

    fallback = Path(__file__).parent.parent / "frontend_legacy"
    if fallback.exists():
        return fallback

    return Path(__file__).parent.parent / "frontend"


def create_app(runtime_service: Optional[RuntimeService] = None) -> FastAPI:
    runtime_service = runtime_service or RuntimeService()
    services = build_services(runtime_service)
    broadcast = BroadcastState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        legacy_main.app_instance = runtime_service.app_instance
        if runtime_service.has_app():
            runtime_service.start()

            async def broadcast_loop():
                while runtime_service.has_app() and runtime_service.get_app()._running:
                    await broadcast_state(runtime_service.get_app(), broadcast)
                    await asyncio.sleep(1 / 60)

            app.state.broadcast_task = asyncio.create_task(broadcast_loop())

        yield

        task = getattr(app.state, "broadcast_task", None)
        if task:
            task.cancel()
        if runtime_service.has_app():
            runtime_service.stop()

    app = FastAPI(title="StrikeLab Putting Sim", version="1.0.0", lifespan=lifespan)
    app.state.services = services
    app.state.broadcast_state = broadcast

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    frontend_path = _resolve_frontend_path()
    if frontend_path.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    for router in (
        health_router,
        video_router,
        config_router,
        game_router,
        session_router,
        drills_router,
        stats_router,
        calibration_router,
        diagnostics_router,
        validation_router,
        create_ws_router(services, broadcast),
    ):
        app.include_router(router)

    return app
