from __future__ import annotations

import time
from typing import Any

from ..protocol.ws_common import WsSnapshotMessage
from .messages_v1 import build_state_message


def build_snapshot_message(sim: Any, seq: int) -> dict:
    return WsSnapshotMessage(
        v=2,
        t="snapshot",
        seq=seq,
        ts_ms=time.time() * 1000,
        payload=build_state_message(sim),
    ).model_dump()
