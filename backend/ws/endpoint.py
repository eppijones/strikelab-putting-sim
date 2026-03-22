from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..dependencies import AppServices
from .broadcaster import BroadcastState, is_ws_v2_enabled, next_seq
from .event_builder import build_event_message
from .snapshot_builder import build_snapshot_message

logger = logging.getLogger(__name__)


def create_ws_router(services: AppServices, broadcast_state: BroadcastState) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()

        sim = services.runtime.get_app()
        await sim.add_client(websocket)

        try:
            if is_ws_v2_enabled():
                snapshot = build_snapshot_message(sim, next_seq(broadcast_state))
                await websocket.send_text(json.dumps(snapshot))

            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=60)
                    try:
                        cmd = json.loads(data)
                    except json.JSONDecodeError:
                        cmd = {}

                    if cmd.get("type") == "reset":
                        sim.reset_all()
                        if is_ws_v2_enabled():
                            ack = build_event_message("cmd_ack", next_seq(broadcast_state), command="reset", ok=True)
                            await websocket.send_text(json.dumps(ack))
                    elif cmd.get("type") == "ping":
                        if is_ws_v2_enabled():
                            pong = build_event_message("pong", next_seq(broadcast_state))
                            await websocket.send_text(json.dumps(pong))
                        else:
                            await websocket.send_text(json.dumps({"type": "pong"}))
                except asyncio.TimeoutError:
                    try:
                        if is_ws_v2_enabled():
                            ping = build_event_message("ping", next_seq(broadcast_state))
                            await websocket.send_text(json.dumps(ping))
                        else:
                            await websocket.send_text(json.dumps({"type": "ping"}))
                    except Exception:
                        break
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.error("WebSocket error: %s", exc)
        finally:
            try:
                await sim.remove_client(websocket)
            except Exception:
                pass

    return router
