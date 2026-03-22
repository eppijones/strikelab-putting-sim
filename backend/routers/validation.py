from __future__ import annotations

from fastapi import APIRouter

from .. import legacy_main

router = APIRouter()

router.add_api_route("/api/validation/report-shot", legacy_main.report_validation_shot, methods=["POST"])
router.add_api_route("/api/validation/10-shot-report", legacy_main.ten_shot_validation_report, methods=["POST"])
