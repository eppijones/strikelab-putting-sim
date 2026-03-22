from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

from ..config import get_config
from .delta_builder import build_delta_message
from .messages_v1 import build_state_message
from .snapshot_builder import build_snapshot_message


@dataclass
class BroadcastState:
    seq: int = 0
    last_snapshot: Optional[dict[str, Any]] = None


def is_ws_v2_enabled() -> bool:
    config = get_config()
    return bool(getattr(config, "ws_v2_enabled", False)) or int(getattr(config, "ws_protocol_version", 1)) >= 2


def is_ws_v2_delta_enabled() -> bool:
    return bool(getattr(get_config(), "ws_v2_delta_enabled", False))


def next_seq(state: BroadcastState) -> int:
    state.seq += 1
    return state.seq


def build_broadcast_payload(sim: Any, state: BroadcastState) -> str:
    seq = next_seq(state)
    if not is_ws_v2_enabled():
        return json.dumps(build_state_message(sim))

    current_snapshot = build_state_message(sim)
    if state.last_snapshot is None or not is_ws_v2_delta_enabled():
        payload = build_snapshot_message(sim, seq)
    else:
        payload = build_delta_message(state.last_snapshot, current_snapshot, seq)
    state.last_snapshot = current_snapshot
    return json.dumps(payload)


async def broadcast_state(sim: Any, state: BroadcastState) -> None:
    if not sim._ws_clients:
        return

    message_json = build_broadcast_payload(sim, state)
    async with sim._ws_lock:
        disconnected = set()
        for ws in sim._ws_clients:
            try:
                await ws.send_text(message_json)
            except Exception:
                disconnected.add(ws)
        sim._ws_clients -= disconnected
