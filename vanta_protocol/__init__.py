"""vanta_protocol — wire message types for miner<->validator and validator<->validator
traffic, replacing the old template/protocol.py (bittensor-scaffolding naming, vestigial
now that bt.Synapse/Axon/Dendrite are gone — see shared_objects/http_server,
shared_objects/http_client, shared_objects/http_auth for the transport they ride on).
"""
from vanta_protocol.base import Ack
from vanta_protocol.broadcast import (
    BroadcastRequest,
    BroadcastResponse,
    ValidatorCheckpointRequest,
    ValidatorCheckpointResponse,
)
from vanta_protocol.signals import (
    GetPositionsRequest,
    GetPositionsResponse,
    SendSignalRequest,
    SendSignalResponse,
)

# name -> (request model, response model), used by shared_objects/http_server and
# shared_objects/http_client to generically wire up routes, and to generically dispatch
# the validator<->validator broadcast types (vali_objects/validator_broadcast_base.py,
# shared_objects/subtensor_ops/subtensor_ops.py's do_broadcast()).
MESSAGE_TYPES: dict = {
    "SendSignal": (SendSignalRequest, SendSignalResponse),
    "GetPositions": (GetPositionsRequest, GetPositionsResponse),
    "ValidatorCheckpoint": (ValidatorCheckpointRequest, ValidatorCheckpointResponse),
    "CollateralRecord": (BroadcastRequest, BroadcastResponse),
    "AssetSelection": (BroadcastRequest, BroadcastResponse),
    "SubaccountRegistration": (BroadcastRequest, BroadcastResponse),
    "EntityEndpointUpdate": (BroadcastRequest, BroadcastResponse),
    "StandardMinerOpening": (BroadcastRequest, BroadcastResponse),
    "StandardMinerRecord": (BroadcastRequest, BroadcastResponse),
    "AccountSizeLocks": (BroadcastRequest, BroadcastResponse),
}

# name -> the fields a legacy bittensor v10 Dendrite hashed into computed_body_hash for
# that message type (SHA3-256 over the concatenation of SHA3-256(str(field_value)) for
# each field, in this order) — a historical fact about the OLD wire format, independent
# of the Request/Response models above. Used only by shared_objects/http_auth/legacy_v10.py
# during the backward-compatibility transition window.
#
# The validator<->validator broadcast types (CollateralRecord, AssetSelection,
# SubaccountRegistration, EntityEndpointUpdate, StandardMinerOpening, StandardMinerRecord,
# AccountSizeLocks) are deliberately absent: they now share BroadcastRequest's generic
# `payload` field, which no longer matches the old per-type field name a v10 peer hashed,
# so legacy verification can't work for them regardless. This is fine only because every
# validator is assumed to upgrade together — see MIGRATION-GUIDE.md.
LEGACY_HASH_FIELDS: dict = {
    "SendSignal": ("signal",),
    "GetPositions": ("positions",),
    "ValidatorCheckpoint": ("checkpoint",),
}

__all__ = [
    "Ack",
    "MESSAGE_TYPES",
    "LEGACY_HASH_FIELDS",
    "SendSignalRequest", "SendSignalResponse",
    "GetPositionsRequest", "GetPositionsResponse",
    "ValidatorCheckpointRequest", "ValidatorCheckpointResponse",
    "BroadcastRequest", "BroadcastResponse",
]
