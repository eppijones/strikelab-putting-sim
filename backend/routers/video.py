from __future__ import annotations

from fastapi import APIRouter

from .. import legacy_main

router = APIRouter()

router.add_api_route("/api/video", legacy_main.video_feed, methods=["GET"])
router.add_api_route("/api/video/zed", legacy_main.zed_video_feed, methods=["GET"])
router.add_api_route("/api/video/realsense", legacy_main.realsense_video_feed, methods=["GET"])
router.add_api_route("/api/cameras/status", legacy_main.get_cameras_status, methods=["GET"])
router.add_api_route("/api/v1/shot/latest", legacy_main.get_latest_shot_report, methods=["GET"])
