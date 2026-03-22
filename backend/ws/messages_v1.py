from __future__ import annotations

from typing import Any


def build_state_message(sim: Any) -> dict:
    return sim.get_state_message()
