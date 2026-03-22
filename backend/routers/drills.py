from __future__ import annotations

from fastapi import APIRouter

from .. import legacy_main

router = APIRouter()

router.add_api_route("/api/drill", legacy_main.get_drill_state, methods=["GET"])
router.add_api_route("/api/drill/start", legacy_main.start_drill, methods=["POST"])
router.add_api_route("/api/drill/stop", legacy_main.stop_drill, methods=["POST"])
