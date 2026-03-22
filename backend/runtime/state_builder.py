from __future__ import annotations

from typing import Any

from ..ws.messages_v1 import build_state_message


def build_runtime_state(sim: Any) -> dict:
    return build_state_message(sim)
