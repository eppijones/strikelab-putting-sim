from __future__ import annotations

from fastapi import APIRouter

from .. import legacy_main

router = APIRouter()

router.add_api_route("/api/calibrate/rectangle", legacy_main.calibrate_rectangle, methods=["POST"])
router.add_api_route("/api/measure/distance", legacy_main.measure_distance, methods=["POST"])
router.add_api_route("/api/measure/last-shot", legacy_main.get_last_shot_distance, methods=["GET"])
router.add_api_route("/api/calibrate/correct-scale", legacy_main.correct_calibration_scale, methods=["POST"])
router.add_api_route("/api/calibrate/lens-status", legacy_main.get_lens_calibration_status, methods=["GET"])
router.add_api_route("/api/calibrate/verify", legacy_main.verify_calibration, methods=["POST"])
router.add_api_route("/api/calibration/static-ball-test", legacy_main.static_ball_test, methods=["GET"])
router.add_api_route("/api/calibration/9-position-overlay-test", legacy_main.nine_position_overlay_test, methods=["GET"])
router.add_api_route("/api/calibrate/detect-aruco", legacy_main.detect_aruco, methods=["GET"])
router.add_api_route("/api/calibrate/aruco", legacy_main.calibrate_aruco, methods=["POST"])
