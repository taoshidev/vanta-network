"""Validator <-> validator: checkpoint sync and account-state broadcasts.

Each message type here is broadcast from the mothership (or another authoritative
validator) to the rest of the validator set, fanned out via shared_objects/http_client's
HttpAuthClient (see vali_objects/validator_broadcast_base.py, shared_objects/
subtensor_ops/subtensor_ops.py's do_broadcast(), vali_objects/data_sync/p2p_syncer.py).
"""
import typing

from pydantic import BaseModel, Field

from vanta_protocol.base import Ack


class ValidatorCheckpointRequest(BaseModel):
    checkpoint: str = ""


class ValidatorCheckpointResponse(Ack):
    validator_receive_hotkey: str = ""


class BroadcastRequest(BaseModel):
    """Generic envelope for the validator<->validator account-state broadcasts
    (CollateralRecord, AssetSelection, SubaccountRegistration, EntityEndpointUpdate,
    StandardMinerOpening, StandardMinerRecord, AccountSizeLocks in MESSAGE_TYPES) — these
    all carry a single opaque payload dict and a bare Ack response, so one shape covers
    all of them. Which message type a given request/response pair belongs to is carried
    out-of-band (the HTTP path / RPC's explicit message_name), not by the Python type,
    so this assumes every validator is on a build that uses this shape — no legacy v10
    peer support for these message types (see shared_objects/http_auth/legacy_v10.py).

    ValidatorCheckpoint keeps its own shape above: its request field is a str, not a
    dict, and its response carries an extra field.
    """
    payload: typing.Dict = Field(default_factory=dict, max_length=4096)


class BroadcastResponse(Ack):
    pass
