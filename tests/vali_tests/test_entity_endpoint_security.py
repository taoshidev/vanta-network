# developer: Taoshi Inc
# Copyright (c) 2026 Taoshi Inc
"""
Security regression tests for the entity REST endpoints (responsible-disclosure findings):

  F1  POST /entity/subaccount/eliminate — an entity API key may only eliminate its OWN
      subaccounts (tier-500 admin keys retain cross-entity capability).
  F2  POST /admin/rebuild-account/<hotkey> — rebuilding an arbitrary miner's persisted
      account state requires tier 500 (admin), not the self-service entity tier 200.
  F3  POST /entity/create-subaccount — the signed payload carries a one-time nonce +
      timestamp; a captured request cannot be replayed (each replay used to re-trigger
      the Theta registration-fee slashing flow).
  F4  collateral_exempt is rejected outright at the network boundary — any entity could
      previously self-sign the fee-waiver flag.

Uses the lightweight endpoint-test pattern from test_hl_trader_endpoint.py: a bare
ValidatorRestServer via object.__new__ (no heavy constructor), a minimal Flask app, and
mocked RPC clients — the real handler methods run unmodified.
"""
import json
import time
import unittest
import uuid
from unittest.mock import MagicMock

from flask import Flask
from bittensor_wallet import Keypair

from vanta_api.nonce_manager import NonceManager


ENTITY_A = "5EntityAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
ENTITY_B = "5EntityBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
KEY_A = "api-key-entity-a"
KEY_B = "api-key-entity-b"
KEY_ADMIN = "api-key-admin"
KEY_TIERS = {KEY_A: 200, KEY_B: 200, KEY_ADMIN: 500}


def _bare_server():
    """ValidatorRestServer without its heavy constructor, with auth mocked to a fixed key set."""
    from vanta_api.validator_rest_server import ValidatorRestServer

    server = object.__new__(ValidatorRestServer)
    server.api_key_to_alias = {KEY_A: ENTITY_A, KEY_B: ENTITY_B, KEY_ADMIN: "ops-admin"}
    server.is_valid_api_key = lambda k: k in KEY_TIERS
    server.can_access_tier = lambda k, tier: KEY_TIERS.get(k, 0) >= tier
    server._entity_client = MagicMock()
    server.nonce_manager = NonceManager()
    return server


class TestEliminateSubaccountOwnership(unittest.TestCase):
    """F1: cross-entity elimination must be refused."""

    def setUp(self):
        self.server = _bare_server()
        self.server._entity_client.eliminate_subaccount.return_value = (True, "eliminated")
        app = Flask(__name__)
        app.route("/entity/subaccount/eliminate", methods=["POST"])(self.server.eliminate_subaccount)
        self.client = app.test_client()

    def _post(self, api_key, entity_hotkey):
        return self.client.post(
            "/entity/subaccount/eliminate",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"entity_hotkey": entity_hotkey, "subaccount_id": 0},
        )

    def test_cross_entity_eliminate_rejected(self):
        resp = self._post(KEY_B, ENTITY_A)
        self.assertEqual(resp.status_code, 403)
        self.server._entity_client.eliminate_subaccount.assert_not_called()

    def test_own_entity_eliminate_allowed(self):
        resp = self._post(KEY_A, ENTITY_A)
        self.assertEqual(resp.status_code, 200)
        self.server._entity_client.eliminate_subaccount.assert_called_once()

    def test_admin_cross_entity_eliminate_allowed(self):
        resp = self._post(KEY_ADMIN, ENTITY_A)
        self.assertEqual(resp.status_code, 200)

    def test_non_entity_tier200_key_rejected(self):
        # A manually-provisioned tier-200 key whose alias is not an entity hotkey
        # must not be able to eliminate anyone.
        KEY_TIERS["ops-reader"] = 200
        self.server.api_key_to_alias["ops-reader"] = "some-user-id"
        try:
            resp = self._post("ops-reader", ENTITY_A)
            self.assertEqual(resp.status_code, 403)
        finally:
            del KEY_TIERS["ops-reader"]

    def test_key_missing_from_alias_map_rejected(self):
        # A valid tier-200 key absent from api_key_to_alias (stale/out-of-sync
        # mapping) must be rejected, never treated as an owner.
        KEY_TIERS["unmapped-key"] = 200
        try:
            resp = self._post("unmapped-key", ENTITY_A)
            self.assertEqual(resp.status_code, 403)
            self.server._entity_client.eliminate_subaccount.assert_not_called()
        finally:
            del KEY_TIERS["unmapped-key"]

    def test_null_entity_hotkey_rejected(self):
        # JSON null passes the field-presence check; it must never match a
        # missing alias (None == None) and read as ownership.
        KEY_TIERS["unmapped-key"] = 200
        try:
            resp = self.client.post(
                "/entity/subaccount/eliminate",
                headers={"Authorization": "Bearer unmapped-key"},
                json={"entity_hotkey": None, "subaccount_id": 0},
            )
            self.assertEqual(resp.status_code, 400)
            self.server._entity_client.eliminate_subaccount.assert_not_called()
        finally:
            del KEY_TIERS["unmapped-key"]


class TestRebuildAccountTierGate(unittest.TestCase):
    """F2: account rebuild is admin-only (tier 500), including preview mode."""

    def setUp(self):
        self.server = _bare_server()
        self.server._miner_account_client = MagicMock()
        app = Flask(__name__)
        app.route("/admin/rebuild-account/<hotkey>", methods=["POST"])(self.server.rebuild_miner_account)
        self.client = app.test_client()

    def _post(self, api_key, body=None):
        return self.client.post(
            "/admin/rebuild-account/5SomeMinerHotkey",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body or {},
        )

    def test_tier200_rejected(self):
        resp = self._post(KEY_A)
        self.assertEqual(resp.status_code, 403)
        self.server._miner_account_client.get_account.assert_not_called()

    def test_tier200_preview_rejected_too(self):
        # Preview discloses arbitrary miners' account internals — same gate.
        resp = self._post(KEY_B, {"preview": True})
        self.assertEqual(resp.status_code, 403)

    def test_admin_passes_gate(self):
        # 404 (no such account from the mocked client) proves the tier gate admitted the
        # request and the handler proceeded to account lookup.
        self.server._miner_account_client.get_account.return_value = None
        resp = self._post(KEY_ADMIN)
        self.assertEqual(resp.status_code, 404)


class TestCreateSubaccountReplayProtection(unittest.TestCase):
    """F3 + F4: signed nonce/timestamp single-use; collateral_exempt refused outright."""

    def setUp(self):
        self.server = _bare_server()
        self.server._entity_client.create_subaccount.return_value = (
            True, {"synthetic_hotkey": "e_0", "subaccount_id": 0}, "created")
        self.server._verify_coldkey_owns_hotkey = lambda ck, hk: True
        app = Flask(__name__)
        app.route("/entity/create-subaccount", methods=["POST"])(self.server.create_subaccount)
        self.client = app.test_client()
        self.coldkey = Keypair.create_from_uri("//Alice")
        self.hotkey_addr = Keypair.create_from_uri("//Bob").ss58_address

    def _signed_body(self, nonce=None, timestamp=None, extra=None, tamper_nonce=None):
        nonce = nonce or uuid.uuid4().hex
        timestamp = timestamp if timestamp is not None else int(time.time() * 1000)
        sig_dict = {
            "account_size": 25000.0,
            "asset_class": "crypto",
            "entity_coldkey": self.coldkey.ss58_address,
            "entity_hotkey": self.hotkey_addr,
            "nonce": str(nonce),
            "timestamp": timestamp,
        }
        signature = self.coldkey.sign(json.dumps(sig_dict, sort_keys=True).encode()).hex()
        body = {
            "entity_coldkey": self.coldkey.ss58_address,
            "entity_hotkey": self.hotkey_addr,
            "account_size": 25000.0,
            "asset_class": "crypto",
            "signature": signature,
            "nonce": str(tamper_nonce if tamper_nonce is not None else nonce),
            "timestamp": timestamp,
            "version": "3.1.0",
        }
        if extra:
            body.update(extra)
        return body

    def _post(self, body):
        return self.client.post("/entity/create-subaccount", json=body)

    def test_missing_nonce_rejected(self):
        body = self._signed_body()
        del body["nonce"]
        resp = self._post(body)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("nonce", resp.get_json()["error"])

    def test_valid_request_succeeds_then_replay_rejected(self):
        body = self._signed_body()
        first = self._post(body)
        self.assertEqual(first.status_code, 200, first.get_json())
        self.server._entity_client.create_subaccount.assert_called_once()

        replay = self._post(body)  # byte-identical captured request
        self.assertEqual(replay.status_code, 401, "replayed nonce must be rejected")
        self.server._entity_client.create_subaccount.assert_called_once()  # still once

    def test_tampered_nonce_fails_signature(self):
        # The nonce is signature-covered: an attacker cannot swap in a fresh nonce.
        body = self._signed_body(tamper_nonce=uuid.uuid4().hex)
        resp = self._post(body)
        self.assertEqual(resp.status_code, 401)
        self.server._entity_client.create_subaccount.assert_not_called()

    def test_stale_timestamp_rejected(self):
        # One millisecond past the NonceManager window (read from the real
        # instance, so this tracks the implementation's TTL).
        stale_ms = self.server.nonce_manager.ttl_ms + 1
        body = self._signed_body(timestamp=int(time.time() * 1000) - stale_ms)
        resp = self._post(body)
        self.assertEqual(resp.status_code, 401)
        self.server._entity_client.create_subaccount.assert_not_called()

    def test_same_nonce_different_entity_allowed(self):
        # Nonces are scoped per coldkey::hotkey — a nonce used by one entity
        # must not block a different entity's request.
        shared_nonce = uuid.uuid4().hex
        first = self._post(self._signed_body(nonce=shared_nonce))
        self.assertEqual(first.status_code, 200)

        other_cold = Keypair.create_from_uri("//Charlie")
        other_hot = Keypair.create_from_uri("//Dave").ss58_address
        ts = int(time.time() * 1000)
        sig_dict = {
            "account_size": 25000.0,
            "asset_class": "crypto",
            "entity_coldkey": other_cold.ss58_address,
            "entity_hotkey": other_hot,
            "nonce": shared_nonce,
            "timestamp": ts,
        }
        body = {
            "entity_coldkey": other_cold.ss58_address,
            "entity_hotkey": other_hot,
            "account_size": 25000.0,
            "asset_class": "crypto",
            "signature": other_cold.sign(json.dumps(sig_dict, sort_keys=True).encode()).hex(),
            "nonce": shared_nonce,
            "timestamp": ts,
            "version": "3.1.0",
        }
        second = self._post(body)
        self.assertEqual(second.status_code, 200, second.get_json())

    def test_old_cli_version_rejected_with_upgrade_message(self):
        body = self._signed_body()
        body["version"] = "2.2.1"
        resp = self._post(body)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("upgrade", resp.get_json()["error"].lower())
        self.server._entity_client.create_subaccount.assert_not_called()

    def test_unowned_hotkey_rejected(self):
        # Subtensor coldkey->hotkey ownership failure must 403 before any RPC.
        self.server._verify_coldkey_owns_hotkey = lambda ck, hk: False
        resp = self._post(self._signed_body())
        self.assertEqual(resp.status_code, 403)
        self.server._entity_client.create_subaccount.assert_not_called()

    def test_collateral_exempt_rejected(self):
        resp = self._post(self._signed_body(extra={"collateral_exempt": True}))
        self.assertEqual(resp.status_code, 403)
        self.server._entity_client.create_subaccount.assert_not_called()

    def test_legacy_admin_flag_rejected(self):
        resp = self._post(self._signed_body(extra={"admin": True}))
        self.assertEqual(resp.status_code, 403)
        self.server._entity_client.create_subaccount.assert_not_called()


if __name__ == "__main__":
    unittest.main()
