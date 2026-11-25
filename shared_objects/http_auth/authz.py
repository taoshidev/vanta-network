"""Caller-authorization policy — separate from signature verification.

Authentication (AuthVerifier) answers "who signed this." Authorization answers "is that
hotkey welcome here." Per CLAUDE.md, this subnet's policy ignores validator_permit/trust
(under Null consensus one UID holds the permit) and instead checks registration via
stake weight through the existing MetagraphClient.
"""
from shared_objects.http_auth.exceptions import AuthError


class Forbidden(AuthError):
    """Verified caller is not authorized (distinct from a failed signature check)."""


def require_registered_hotkey(hotkey_ss58: str, metagraph_client) -> None:
    if not metagraph_client.has_hotkey(hotkey_ss58):
        raise Forbidden(f"hotkey {hotkey_ss58} is not registered on this subnet's metagraph")
