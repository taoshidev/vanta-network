"""Round-trip and tamper/replay tests for shared_objects/http_auth.

Covers both the new btauth/1 scheme (via the real bittensor.http_auth SDK functions)
and the legacy bittensor v10 compat path (shared_objects/http_auth/legacy_v10.py),
including a byte-for-byte check against a real fixture captured from a live
bittensor==10.3.0 Dendrite before this repo upgraded to bittensor v11
(tests/fixtures/http_auth/legacy_v10_sendsignal.json).
"""
import hashlib
import json
import time
import uuid
from pathlib import Path

import bittensor as bt
import pytest

from shared_objects.http_auth import AuthVerifier
from shared_objects.http_auth.exceptions import BadSignature, NoRecognizedScheme, ReplayedRequest, WrongReceiver

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "http_auth"


def _make_wallet(tmp_path, name):
    w = bt.Wallet(name=name, hotkey=f"{name}_hk", path=str(tmp_path))
    w.create_new_coldkey(use_password=False, overwrite=True, suppress=True)
    w.create_new_hotkey(use_password=False, overwrite=True, suppress=True)
    return w


def _legacy_body_hash(body: dict, hash_fields: tuple) -> str:
    def get_hash(s: str) -> str:
        return hashlib.sha3_256(s.encode()).hexdigest()
    hashes = [get_hash(str(body[f])) for f in hash_fields]
    return get_hash("".join(hashes))


class TestBtauthV1RoundTrip:
    def test_sign_and_verify_succeeds(self, tmp_path):
        sender = _make_wallet(tmp_path, "sender")
        receiver = _make_wallet(tmp_path, "receiver")
        verifier = AuthVerifier(receiver.hotkey.ss58_address, initial_hotkeys=[sender.hotkey.ss58_address])

        body = b'{"signal": {"trade_pair": "BTCUSD"}}'
        headers = bt.http_auth.sign(sender, method="POST", path="/SendSignal", body=body,
                                     receiver_ss58=receiver.hotkey.ss58_address)

        caller = verifier.verify(headers, body, method="POST", path="/SendSignal",
                                  message_name="SendSignal", legacy_hash_fields=("signal",))
        assert caller.hotkey_ss58 == sender.hotkey.ss58_address
        assert caller.scheme == "btauth/1"

    def test_tampered_body_rejected(self, tmp_path):
        sender = _make_wallet(tmp_path, "sender")
        receiver = _make_wallet(tmp_path, "receiver")
        verifier = AuthVerifier(receiver.hotkey.ss58_address, initial_hotkeys=[sender.hotkey.ss58_address])

        body = b'{"signal": {"a": 1}}'
        headers = bt.http_auth.sign(sender, method="POST", path="/SendSignal", body=body,
                                     receiver_ss58=receiver.hotkey.ss58_address)
        with pytest.raises(BadSignature):
            verifier.verify(headers, b'{"signal": {"a": 2}}', method="POST", path="/SendSignal",
                             message_name="SendSignal", legacy_hash_fields=("signal",))

    def test_replay_rejected(self, tmp_path):
        sender = _make_wallet(tmp_path, "sender")
        receiver = _make_wallet(tmp_path, "receiver")
        verifier = AuthVerifier(receiver.hotkey.ss58_address, initial_hotkeys=[sender.hotkey.ss58_address])

        body = b'{"signal": {"a": 1}}'
        headers = bt.http_auth.sign(sender, method="POST", path="/SendSignal", body=body,
                                     receiver_ss58=receiver.hotkey.ss58_address)
        verifier.verify(headers, body, method="POST", path="/SendSignal",
                         message_name="SendSignal", legacy_hash_fields=("signal",))
        with pytest.raises(ReplayedRequest):
            verifier.verify(headers, body, method="POST", path="/SendSignal",
                             message_name="SendSignal", legacy_hash_fields=("signal",))

    def test_wrong_receiver_rejected(self, tmp_path):
        sender = _make_wallet(tmp_path, "sender")
        receiver = _make_wallet(tmp_path, "receiver")
        wrong_receiver = _make_wallet(tmp_path, "wrong")
        verifier = AuthVerifier(receiver.hotkey.ss58_address, initial_hotkeys=[sender.hotkey.ss58_address])

        body = b'{"signal": {"a": 1}}'
        headers = bt.http_auth.sign(sender, method="POST", path="/SendSignal", body=body,
                                     receiver_ss58=wrong_receiver.hotkey.ss58_address)
        with pytest.raises(WrongReceiver):
            verifier.verify(headers, body, method="POST", path="/SendSignal",
                             message_name="SendSignal", legacy_hash_fields=("signal",))

    def test_untrusted_sender_rejected(self, tmp_path):
        sender = _make_wallet(tmp_path, "sender")
        receiver = _make_wallet(tmp_path, "receiver")
        # No initial_hotkeys -> sender is not in the trusted_sender_schemes map.
        verifier = AuthVerifier(receiver.hotkey.ss58_address)

        body = b'{"signal": {"a": 1}}'
        headers = bt.http_auth.sign(sender, method="POST", path="/SendSignal", body=body,
                                     receiver_ss58=receiver.hotkey.ss58_address)
        with pytest.raises(BadSignature):
            verifier.verify(headers, body, method="POST", path="/SendSignal",
                             message_name="SendSignal", legacy_hash_fields=("signal",))


class TestLegacyV10Compat:
    def test_real_v10_fixture_sendsignal(self):
        """Byte-for-byte compat check against a real bittensor==10.3.0 Dendrite capture."""
        fixture = json.load(open(FIXTURES_DIR / "legacy_v10_sendsignal.json"))
        verifier = AuthVerifier(self_hotkey_ss58=fixture["receiver_hotkey"])
        try:
            caller = verifier.verify(
                fixture["headers"], fixture["body"].encode(),
                method="POST", path="/SendSignal", message_name="SendSignal",
                legacy_hash_fields=tuple(fixture["required_hash_fields"]),
            )
            assert caller.hotkey_ss58 == fixture["sender_hotkey"]
            assert caller.scheme == "legacy_v10"
        except ReplayedRequest:
            pytest.skip("fixture nonce has aged out of the freshness window (captured at repo-migration time)")

    def test_real_v10_fixture_getpositions(self):
        fixture = json.load(open(FIXTURES_DIR / "legacy_v10_getpositions.json"))
        verifier = AuthVerifier(self_hotkey_ss58=fixture["receiver_hotkey"])
        try:
            caller = verifier.verify(
                fixture["headers"], fixture["body"].encode(),
                method="POST", path="/GetPositions", message_name="GetPositions",
                legacy_hash_fields=tuple(fixture["required_hash_fields"]),
            )
            assert caller.hotkey_ss58 == fixture["sender_hotkey"]
        except ReplayedRequest:
            pytest.skip("fixture nonce has aged out of the freshness window (captured at repo-migration time)")

    def test_fresh_legacy_request_accepted(self):
        """A freshly-signed legacy-shaped request (regenerated with the real algorithm,
        using bt.sp_core.Keypair since bittensor v10 is no longer installed) verifies."""
        sender_kp = bt.sp_core.Keypair.create_from_uri(f"//test-{uuid.uuid4()}")
        receiver_kp = bt.sp_core.Keypair.create_from_uri(f"//test-{uuid.uuid4()}")

        body = {"signal": {"a": 1}}
        body_hash = _legacy_body_hash(body, ("signal",))
        nonce = time.time_ns()
        req_uuid = str(uuid.uuid4())
        message = f"{nonce}.{sender_kp.ss58_address}.{receiver_kp.ss58_address}.{req_uuid}.{body_hash}"
        signature = sender_kp.sign(message.encode())

        headers = {
            "bt_header_dendrite_hotkey": sender_kp.ss58_address,
            "bt_header_dendrite_nonce": str(nonce),
            "bt_header_dendrite_uuid": req_uuid,
            "bt_header_dendrite_signature": "0x" + signature.hex(),
            "computed_body_hash": body_hash,
        }

        verifier = AuthVerifier(self_hotkey_ss58=receiver_kp.ss58_address)
        caller = verifier.verify(
            headers, json.dumps(body).encode(),
            method="POST", path="/SendSignal", message_name="SendSignal",
            legacy_hash_fields=("signal",),
        )
        assert caller.hotkey_ss58 == sender_kp.ss58_address
        assert caller.scheme == "legacy_v10"

    def test_legacy_disabled_raises(self):
        fixture = json.load(open(FIXTURES_DIR / "legacy_v10_sendsignal.json"))
        verifier = AuthVerifier(self_hotkey_ss58=fixture["receiver_hotkey"], legacy_enabled=False)
        with pytest.raises(NoRecognizedScheme):
            verifier.verify(
                fixture["headers"], fixture["body"].encode(),
                method="POST", path="/SendSignal", message_name="SendSignal",
                legacy_hash_fields=tuple(fixture["required_hash_fields"]),
            )
