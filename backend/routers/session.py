from __future__ import annotations

from fastapi import APIRouter

from .. import legacy_main

router = APIRouter()

router.add_api_route("/api/users", legacy_main.get_users, methods=["GET"])
router.add_api_route("/api/users", legacy_main.create_user, methods=["POST"])
router.add_api_route("/api/users/{user_id}", legacy_main.delete_user, methods=["DELETE"])
router.add_api_route("/api/users/{user_id}/reset", legacy_main.reset_user_data, methods=["POST"])
router.add_api_route("/api/session", legacy_main.get_session, methods=["GET"])
router.add_api_route("/api/session/user", legacy_main.set_session_user, methods=["POST"])
router.add_api_route("/api/session/reset", legacy_main.reset_session, methods=["POST"])
router.add_api_route("/api/session/history", legacy_main.get_session_history, methods=["GET"])
router.add_api_route("/api/users/{user_id}/history", legacy_main.get_user_shot_history, methods=["GET"])
