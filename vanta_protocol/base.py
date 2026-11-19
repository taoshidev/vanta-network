"""Shared base for every wire message type.

Replaces the old template/protocol.py (bittensor-scaffolding naming, vestigial now that
bt.Synapse is gone). Every message type is exactly two pydantic models — `XRequest` (what
a caller may send) and `XResponse` (what comes back) — never a third "internal" class
mutated in place and threaded through RPC. Handlers at every layer (FastAPI route ->
*_client.py -> RPC -> *_server.py -> business logic) take `(request: XRequest,
caller_hotkey: str)` and return `XResponse`, explicitly, with `caller_hotkey` passed as a
plain parameter (set by the signed-request auth layer, shared_objects/http_auth/) rather
than smuggled onto a mutable field the way bittensor's Synapse.dendrite.hotkey was.
"""
from pydantic import BaseModel


class Ack(BaseModel):
    """Common response fields every message type's *Response extends."""
    successfully_processed: bool = False
    error_message: str = ""
