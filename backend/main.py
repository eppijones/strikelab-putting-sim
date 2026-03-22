"""
Thin backend composition root for StrikeLab Putting Sim.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from . import legacy_main
from .app_factory import create_app
from .runtime.app import PuttingSimApp
from .services.runtime_service import RuntimeService

logger = logging.getLogger(__name__)

runtime_service = RuntimeService()
app = create_app(runtime_service)


def main() -> None:
    parser = argparse.ArgumentParser(description="StrikeLab Putting Sim")
    parser.add_argument("--arducam", action="store_true", help="Use Arducam OV9281")
    parser.add_argument("--webcam", action="store_true", help="Use standard webcam")
    parser.add_argument("--replay", type=str, help="Replay from video file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.replay:
        camera_mode = legacy_main.CameraMode.REPLAY
        replay_path = args.replay
    elif args.webcam:
        camera_mode = legacy_main.CameraMode.WEBCAM
        replay_path = None
    else:
        camera_mode = legacy_main.CameraMode.ARDUCAM
        replay_path = None

    runtime = PuttingSimApp(camera_mode=camera_mode, replay_path=replay_path)
    runtime_service.set_app(runtime)
    legacy_main.app_instance = runtime

    logger.info("Starting server on %s:%s", args.host, args.port)
    logger.info("Camera mode: %s", camera_mode.value)
    if replay_path:
        logger.info("Replay file: %s", replay_path)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()
