"""
Order envelope for traders who trade through a permissionless entity miner (PEM). Spec: docs/pem_envelope.md.

The trader's device builds a payload (the signal plus who placed it, through which PEM, on which netuid, when, and
the order uuid) and signs its exact UTF-8 bytes with the trader's hotkey. The PEM forwards the payload string and
signature unchanged in SendSignal.origin_payload / origin_signature, signing the synapse itself with its own
dendrite. The validator checks the trader's signature over the bytes it received, then parses them.

This module only builds, signs and checks the envelope's own contents. Whether the relay is a registered PEM, the
timestamp is fresh and the uuid unused are checked by the validator, against live state.
"""
import json
import uuid as uuid_lib
from dataclasses import dataclass

from bittensor_wallet import Keypair

from time_util.time_util import TimeUtil

PAYLOAD_VERSION = 1
MAX_PAYLOAD_LENGTH = 8192        # SendSignal.origin_payload max_length
MAX_SIGNATURE_LENGTH = 256       # SendSignal.origin_signature max_length
MAX_ORDER_UUID_LENGTH = 256      # SendSignal.miner_order_uuid max_length
PAYLOAD_FIELDS = frozenset({"v", "netuid", "origin_hotkey", "relay_hotkey", "miner_order_uuid", "timestamp_ms", "signal"})


class EnvelopeError(ValueError):
    """An envelope that is malformed or not signed by its origin hotkey."""


@dataclass(frozen=True)
class EnvelopePayload:
    v: int
    netuid: int
    origin_hotkey: str
    relay_hotkey: str
    miner_order_uuid: str
    timestamp_ms: int
    signal: dict


def build_payload(origin_hotkey: str, relay_hotkey: str, signal: dict, netuid: int,
                  miner_order_uuid: str = None, timestamp_ms: int = None) -> str:
    """The payload string the trader signs: sorted keys, no whitespace. The validator verifies whatever bytes
    were signed, so this form is only needed to reproduce the test vectors."""
    payload = {
        "v": PAYLOAD_VERSION,
        "netuid": netuid,
        "origin_hotkey": origin_hotkey,
        "relay_hotkey": relay_hotkey,
        "miner_order_uuid": miner_order_uuid or str(uuid_lib.uuid4()),
        "timestamp_ms": timestamp_ms if timestamp_ms is not None else TimeUtil.now_in_millis(),
        "signal": signal,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sign_payload(payload: str, keypair: Keypair) -> str:
    """Hex sr25519 signature of the payload's UTF-8 bytes by `keypair` (the trader's hotkey). sr25519 signatures
    are randomized, so signing the same payload twice gives two different, equally valid signatures."""
    return keypair.sign(payload.encode("utf-8")).hex()


def parse_payload(payload: str) -> EnvelopePayload:
    """Parse and check the payload's structure. Does not check the signature (see verify_envelope)."""
    if not isinstance(payload, str) or not payload:
        raise EnvelopeError("origin_payload must be a non-empty string")
    if len(payload) > MAX_PAYLOAD_LENGTH:
        raise EnvelopeError(f"origin_payload is longer than {MAX_PAYLOAD_LENGTH} characters")

    def no_duplicate_keys(pairs):
        keys = [key for key, _ in pairs]
        if len(keys) != len(set(keys)):
            raise EnvelopeError("origin_payload repeats a key")
        return dict(pairs)

    try:
        fields = json.loads(payload, object_pairs_hook=no_duplicate_keys)
    except EnvelopeError:
        raise
    except ValueError as e:
        raise EnvelopeError(f"origin_payload is not valid JSON: {e}")
    if not isinstance(fields, dict):
        raise EnvelopeError("origin_payload must be a JSON object")

    missing = PAYLOAD_FIELDS - fields.keys()
    if missing:
        raise EnvelopeError(f"origin_payload is missing {sorted(missing)}")
    unknown = fields.keys() - PAYLOAD_FIELDS
    if unknown:
        raise EnvelopeError(f"origin_payload has unknown fields {sorted(unknown)}")

    if not isinstance(fields["v"], int) or isinstance(fields["v"], bool) or fields["v"] != PAYLOAD_VERSION:
        raise EnvelopeError(f"Unsupported envelope version {fields['v']}, expected {PAYLOAD_VERSION}")
    for name in ("netuid", "timestamp_ms"):
        if not isinstance(fields[name], int) or isinstance(fields[name], bool) or fields[name] < 0:
            raise EnvelopeError(f"{name} must be a non-negative integer")
    for name in ("origin_hotkey", "relay_hotkey"):
        try:
            Keypair(ss58_address=fields[name])
        except Exception:
            raise EnvelopeError(f"{name} must be an SS58 address")
    if fields["origin_hotkey"] == fields["relay_hotkey"]:
        raise EnvelopeError("origin_hotkey and relay_hotkey must differ")
    order_uuid = fields["miner_order_uuid"]
    if not isinstance(order_uuid, str) or not order_uuid or len(order_uuid) > MAX_ORDER_UUID_LENGTH:
        raise EnvelopeError(f"miner_order_uuid must be a non-empty string of at most {MAX_ORDER_UUID_LENGTH} characters")
    if not isinstance(fields["signal"], dict) or not fields["signal"]:
        raise EnvelopeError("signal must be a non-empty JSON object")

    return EnvelopePayload(**fields)


def verify_envelope(payload: str, signature: str) -> EnvelopePayload:
    """Check that `signature` is origin_hotkey's signature over the payload's exact bytes, then return the parsed
    payload. A signature over the bytes wrapped as <Bytes>...</Bytes>, as browser wallet extensions sign, is
    accepted too. Raises EnvelopeError otherwise."""
    parsed = parse_payload(payload)
    if not isinstance(signature, str) or not signature or len(signature) > MAX_SIGNATURE_LENGTH:
        raise EnvelopeError("origin_signature must be a hex string")
    signature_hex = signature[2:] if signature.startswith("0x") else signature
    try:
        signature_bytes = bytes.fromhex(signature_hex)
    except ValueError:
        raise EnvelopeError("origin_signature must be a hex string")
    try:
        is_valid = Keypair(ss58_address=parsed.origin_hotkey).verify(payload.encode("utf-8"), signature_bytes)
    except ValueError:  # a signature of the wrong length
        is_valid = False
    if not is_valid:
        raise EnvelopeError("origin_signature is not origin_hotkey's signature over origin_payload")
    return parsed
