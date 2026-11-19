"""Dual-scheme signed-request verifier: bittensor v11's btauth/1 plus a legacy v10 bridge.

Tries btauth/1 (bittensor.http_auth.verify, the real SDK implementation — see
SIGNED-REQUESTS.md) first, keyed off the presence of the X-Bittensor-Version header. If
that header is absent and legacy acceptance is enabled, falls back to verifying the old
bittensor v10 Dendrite header scheme (shared_objects/http_auth/legacy_v10.py).

A request that presents itself as btauth/1 but fails verification is a hard failure —
it never falls back to the legacy path. Mixing schemes that way would let an attacker
retry a forged new-scheme request under the weaker legacy algorithm.
"""
from dataclasses import dataclass
from typing import Mapping, Optional

import bittensor as bt

from shared_objects.http_auth import legacy_v10
from shared_objects.http_auth.exceptions import AuthError, NoRecognizedScheme
from shared_objects.log import logger

AuthErrorTypes = (bt.http_auth.AuthError, AuthError)


@dataclass
class VerifiedCaller:
    hotkey_ss58: str
    scheme: str  # "btauth/1" or "legacy_v10"


class AuthVerifier:
    """
    bittensor.http_auth.verify requires every sender hotkey to be pre-registered via
    configure_verifier(self_hotkey, trusted_sender_schemes={hotkey: crypto_type}) —
    "A missing configuration or unknown sender raises BadSignature, rather than guessing
    a scheme" (SIGNED-REQUESTS.md). This subnet only uses ordinary sr25519 hotkeys, so
    refresh_trusted_hotkeys() below just (re)trusts every currently-registered hotkey as
    CRYPTO_SR25519, called once at construction and again on every metagraph refresh
    (shared_objects/subtensor_ops/subtensor_ops.py's update_metagraph()) so a newly
    registered hotkey can be verified without a process restart, and a deregistered one
    eventually stops being trusted once the mapping is replaced wholesale.
    """

    def __init__(self, self_hotkey_ss58: str, legacy_enabled: bool = True, initial_hotkeys: Optional[list] = None):
        self.self_hotkey_ss58 = self_hotkey_ss58
        self.legacy_enabled = legacy_enabled
        self.refresh_trusted_hotkeys(initial_hotkeys or [])

    def refresh_trusted_hotkeys(self, hotkeys: list) -> None:
        bt.http_auth.configure_verifier(
            self.self_hotkey_ss58,
            trusted_sender_schemes={hk: bt.CRYPTO_SR25519 for hk in hotkeys},
        )

    def verify(
        self,
        headers: Mapping[str, str],
        body: bytes,
        *,
        method: str,
        path: str,
        message_name: str,
        legacy_hash_fields: Optional[tuple] = None,
        timeout_s: float = 12.0,
    ) -> VerifiedCaller:
        lowered = {k.lower(): v for k, v in headers.items()}

        if bt.http_auth.HEADER_VERSION.lower() in lowered:
            caller = bt.http_auth.verify(
                headers, body,
                method=method, path=path,
                self_hotkey_ss58=self.self_hotkey_ss58,
            )
            return VerifiedCaller(hotkey_ss58=caller.hotkey_ss58, scheme="btauth/1")

        if not self.legacy_enabled:
            raise NoRecognizedScheme(
                f"no {bt.http_auth.HEADER_VERSION} header and legacy auth is disabled"
            )

        if legacy_hash_fields is None:
            raise NoRecognizedScheme(f"no legacy hash fields registered for {message_name!r}")

        legacy_caller = legacy_v10.verify_v10(
            headers, body,
            self_hotkey_ss58=self.self_hotkey_ss58,
            hash_fields=legacy_hash_fields,
            timeout_s=timeout_s,
        )
        logger.info(f"[http_auth] accepted legacy v10 request from {legacy_caller.hotkey_ss58} "
                    f"for {message_name} — peer has not upgraded off bittensor v10")
        return VerifiedCaller(hotkey_ss58=legacy_caller.hotkey_ss58, scheme="legacy_v10")
