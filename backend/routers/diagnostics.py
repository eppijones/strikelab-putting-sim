from __future__ import annotations

from fastapi import APIRouter

from .. import legacy_main

router = APIRouter()

router.add_api_route("/api/diagnostics", legacy_main.get_diagnostics, methods=["GET"])
