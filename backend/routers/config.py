from __future__ import annotations

from fastapi import APIRouter

from .. import legacy_main

router = APIRouter()

router.add_api_route("/api/config", legacy_main.get_config_endpoint, methods=["GET"])
router.add_api_route("/api/green-speed", legacy_main.get_green_speed, methods=["GET"])
router.add_api_route("/api/green-speed", legacy_main.set_green_speed, methods=["POST"])
