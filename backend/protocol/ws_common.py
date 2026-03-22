from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class WsEnvelope(BaseModel):
    v: int = Field(..., ge=1)
    t: str
    seq: int = Field(..., ge=0)
    ts_ms: float
    payload: dict[str, Any]


class WsEventMessage(WsEnvelope):
    t: Literal["event"]


class WsSnapshotMessage(WsEnvelope):
    t: Literal["snapshot"]


class WsDeltaMessage(WsEnvelope):
    t: Literal["delta"]
    base_seq: Optional[int] = Field(default=None, ge=0)
