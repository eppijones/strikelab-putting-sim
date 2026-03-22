from __future__ import annotations

import time
from typing import Any

from ..protocol.ws_common import WsEventMessage


def build_event_message(event_type: str, seq: int, **payload: Any) -> dict:
    return WsEventMessage(
        v=2,
        t="event",
        seq=seq,
        ts_ms=time.time() * 1000,
        payload={"event": event_type, **payload},
    ).model_dump()
