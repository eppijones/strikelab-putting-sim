from __future__ import annotations

from fastapi import APIRouter

from .. import legacy_main

router = APIRouter()

router.add_api_route("/api/shots/{shot_id}", legacy_main.delete_shot, methods=["DELETE"])
router.add_api_route("/api/stats/all-time", legacy_main.get_all_time_stats, methods=["GET"])
router.add_api_route("/api/users/{user_id}/stats", legacy_main.get_user_stats, methods=["GET"])
router.add_api_route("/api/stats/recent", legacy_main.get_recent_stats, methods=["GET"])
router.add_api_route("/api/stats/trend", legacy_main.get_trend_data, methods=["GET"])
router.add_api_route("/api/stats/by-distance", legacy_main.get_stats_by_distance, methods=["GET"])
router.add_api_route("/api/stats/consistency", legacy_main.get_consistency_stats, methods=["GET"])
router.add_api_route("/api/stats/export", legacy_main.export_stats, methods=["POST"])
