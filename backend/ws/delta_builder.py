from __future__ import annotations

import time
from typing import Any

from ..protocol.ws_common import WsDeltaMessage


def _diff_dict(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, value in current.items():
        prev_value = previous.get(key)
        if isinstance(value, dict) and isinstance(prev_value, dict):
            nested = _diff_dict(prev_value, value)
            if nested:
                delta[key] = nested
        elif value != prev_value:
            delta[key] = value
    for key in previous.keys() - current.keys():
        delta[key] = None
    return delta


def build_delta_message(previous: dict[str, Any], current: dict[str, Any], seq: int) -> dict:
    return WsDeltaMessage(
        v=2,
        t="delta",
        seq=seq,
        ts_ms=time.time() * 1000,
        base_seq=max(seq - 1, 0),
        payload=_diff_dict(previous, current),
    ).model_dump()
