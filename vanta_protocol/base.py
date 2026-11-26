"""Shared base for every wire message type: each is exactly two pydantic models,
`XRequest` and `XResponse`. Handlers take `(request: XRequest, caller_hotkey: str)` and
return `XResponse`; `caller_hotkey` is set by shared_objects/http_auth/."""
from pydantic import BaseModel


class Ack(BaseModel):
    """Common response fields every message type's *Response extends."""
    successfully_processed: bool = False
    error_message: str = ""
