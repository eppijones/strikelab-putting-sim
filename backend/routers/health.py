from __future__ import annotations

from fastapi import APIRouter

from .. import legacy_main

router = APIRouter()

router.add_api_route("/", legacy_main.root, methods=["GET"])
router.add_api_route("/api/status", legacy_main.get_status, methods=["GET"])
router.add_api_route("/api/health", legacy_main.health_check, methods=["GET"])
