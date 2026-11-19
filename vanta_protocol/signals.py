"""Miner -> validator: trading signal submission and position queries."""
import typing
from typing import List, Optional

from pydantic import BaseModel, Field

from vanta_protocol.base import Ack


class SendSignalRequest(BaseModel):
    signal: typing.Dict = Field(default_factory=dict, max_length=4096)
    repo_version: str = "N/A"
    miner_order_uuid: str = ""
    subaccount_id: Optional[int] = None


class SendSignalResponse(Ack):
    should_retry: bool = True
    validator_hotkey: str = ""
    order_json: str = ""


class GetPositionsRequest(BaseModel):
    version: int = 0


class GetPositionsResponse(Ack):
    positions: List[typing.Dict] = Field(default_factory=list)
