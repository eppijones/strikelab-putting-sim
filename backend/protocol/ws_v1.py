from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class WsV1Message(BaseModel):
    payload: dict[str, Any]
