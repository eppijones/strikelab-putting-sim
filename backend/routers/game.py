from __future__ import annotations

from fastapi import APIRouter

from .. import legacy_main

router = APIRouter()

router.add_api_route("/api/tracker/reset", legacy_main.reset_tracker, methods=["POST"])
router.add_api_route("/api/game/hole", legacy_main.get_hole_config, methods=["GET"])
router.add_api_route("/api/game/hole", legacy_main.set_hole_distance, methods=["POST"])
router.add_api_route("/api/game/last-shot", legacy_main.get_last_shot_result, methods=["GET"])
