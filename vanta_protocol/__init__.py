"""vanta_protocol — wire message types for miner<->validator and validator<->validator
traffic (transport lives in shared_objects/http_server, http_client, http_auth)."""
from vanta_protocol.base import Ack
from vanta_protocol.broadcast import (
    BroadcastRequest,
    BroadcastResponse,
)
from vanta_protocol.signals import (
    GetPositionsRequest,
    GetPositionsResponse,
    SendSignalRequest,
    SendSignalResponse,
)

# name -> (request model, response model), used to generically wire up routes and
# dispatch broadcasts (vali_objects/validator_broadcast_base.py).
MESSAGE_TYPES: dict = {
    "SendSignal": (SendSignalRequest, SendSignalResponse),
    "GetPositions": (GetPositionsRequest, GetPositionsResponse),
    "CollateralRecord": (BroadcastRequest, BroadcastResponse),
    "AssetSelection": (BroadcastRequest, BroadcastResponse),
    "SubaccountRegistration": (BroadcastRequest, BroadcastResponse),
    "EntityEndpointUpdate": (BroadcastRequest, BroadcastResponse),
    "StandardMinerOpening": (BroadcastRequest, BroadcastResponse),
    "StandardMinerRecord": (BroadcastRequest, BroadcastResponse),
    "AccountSizeLocks": (BroadcastRequest, BroadcastResponse),
}

# name -> fields a legacy bittensor v10 Dendrite hashed into computed_body_hash, used only
# by shared_objects/http_auth/legacy_v10.py. Broadcast types are absent: they share
# BroadcastRequest's generic `payload` field, which a v10 peer never hashed.
LEGACY_HASH_FIELDS: dict = {
    "SendSignal": ("signal",),
    "GetPositions": ("positions",),
}

__all__ = [
    "Ack",
    "MESSAGE_TYPES",
    "LEGACY_HASH_FIELDS",
    "SendSignalRequest", "SendSignalResponse",
    "GetPositionsRequest", "GetPositionsResponse",
    "BroadcastRequest", "BroadcastResponse",
]
