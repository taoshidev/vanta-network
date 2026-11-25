"""Validator <-> validator account-state broadcasts (see validator_broadcast_base.py)."""
import typing

from pydantic import BaseModel, Field

from vanta_protocol.base import Ack


class BroadcastRequest(BaseModel):
    """Generic envelope shared by every broadcast type in MESSAGE_TYPES: an opaque
    payload dict and a bare Ack response. The message_name (not the Python type)
    identifies which broadcast this is (no legacy v10 peer support here)."""
    payload: typing.Dict = Field(default_factory=dict, max_length=4096)


class BroadcastResponse(Ack):
    pass
