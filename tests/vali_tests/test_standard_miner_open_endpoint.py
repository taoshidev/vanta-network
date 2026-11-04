"""
POST /account/open on the validator REST server, against the real Flask handler with the contract client mocked:
the request the CLI signs, the tier validation, signature binding, ownership, replay protection, and how an
opening, a rejection and a validator service that did not answer are reported.
"""
import json
import unittest
import uuid
from unittest.mock import MagicMock

from bittensor_wallet import Keypair
from flask import Flask

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from vanta_api.nonce_manager import NonceManager
from vanta_api.validator_rest_server import ValidatorRestServer
from vanta_cli.src.commands.account import opening_request

SIGNED_FIELDS = ("leverage_tier", "miner_coldkey", "miner_hotkey", "nonce", "timestamp")
RECORD = {"opened_at_ms": 1_700_000_000_000, "opening_leverage_tier": 2, "asset_class": "crypto",
          "account_size": 50_000.0, "locked_theta": 5.5, "opening_price_usd": 100.0, "trading_account_size": None,
          "pro_account_size": None, "payout_scale": None, "closed_at_ms": None}
OPENED = (True, "Opened a $50,000.00 account at tier 2, locking 5.5000 theta at $100.0000 per theta", RECORD)


def _client():
    """Flask test client for the endpoint, with the contract client and the ownership check mocked."""
    server = object.__new__(ValidatorRestServer)
    server._contract_client = MagicMock()
    server._contract_client.open_standard_miner_account.return_value = OPENED
    server._verify_coldkey_owns_hotkey = MagicMock(return_value=True)
    server.nonce_manager = NonceManager()
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.route("/account/open", methods=["POST"])(server.open_account)
    return server, app.test_client()


class TestOpenAccountEndpoint(unittest.TestCase):

    def setUp(self):
        self.coldkey = Keypair.create_from_uri("//Alice")
        self.hotkey = Keypair.create_from_uri("//Bob")
        self.server, self.client = _client()
        self.opening = self.server._contract_client.open_standard_miner_account

    def _body(self, signed=None, drop=(), **tampered):
        """A correctly signed request. `signed` overrides fields before signing, `drop` removes them before
        signing, `tampered` overrides after signing."""
        fields = {
            "leverage_tier": 2,
            "miner_coldkey": self.coldkey.ss58_address,
            "miner_hotkey": self.hotkey.ss58_address,
            "nonce": uuid.uuid4().hex,
            "timestamp": TimeUtil.now_in_millis(),
        }
        fields.update(signed or {})
        for field in drop:
            fields.pop(field, None)
        message = json.dumps(fields, sort_keys=True).encode("utf-8")
        body = {**fields, "signature": self.coldkey.sign(message).hex(), "version": ValiConfig.VANTA_CLI_MINIMUM_VERSION}
        body.update(tampered)
        return body

    def _post(self, body):
        response = self.client.post("/account/open", json=body)
        return response.status_code, response.get_json()

    def test_a_valid_request_opens_the_account(self):
        status, reply = self._post(self._body())
        self.assertEqual(status, 200, reply)
        self.assertTrue(reply["successfully_processed"])
        self.assertEqual(reply["success_message"], OPENED[1])
        self.assertEqual(reply["account"], RECORD)
        self.opening.assert_called_once_with(self.hotkey.ss58_address, 2)

    def test_the_request_the_cli_signs_is_accepted(self):
        payload = {**opening_request(self.coldkey, self.hotkey.ss58_address, 3), "version": ValiConfig.VANTA_CLI_MINIMUM_VERSION}
        self.assertEqual(set(payload), set(SIGNED_FIELDS) | {"signature", "version"})
        status, reply = self._post(payload)
        self.assertEqual(status, 200, reply)
        self.opening.assert_called_once_with(self.hotkey.ss58_address, 3)

    def test_the_tier_is_checked_before_the_signature_and_ownership(self):
        for tier in (0, 4, "2", 2.0, True, None):
            with self.subTest(tier=tier):
                status, reply = self._post(self._body(signed={"leverage_tier": tier}))
                self.assertEqual(status, 400)
                self.assertIn("leverage_tier", reply["error"])
        self.server._verify_coldkey_owns_hotkey.assert_not_called()
        self.opening.assert_not_called()

    def test_a_body_that_is_not_a_json_object_is_400(self):
        for body in ("[1, 2]", "42", "not json", ""):
            with self.subTest(body=body):
                response = self.client.post("/account/open", data=body, content_type="application/json")
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())
        self.opening.assert_not_called()

    def test_missing_fields_are_400(self):
        for field in SIGNED_FIELDS + ("signature",):
            with self.subTest(field=field):
                body = self._body()
                del body[field]
                status, reply = self._post(body)
                self.assertEqual(status, 400)
                self.assertIn(field, reply["error"])
        self.opening.assert_not_called()

    def test_malformed_keys_signature_or_timestamp_are_400(self):
        for body in (self._body(signed={"miner_coldkey": "not an address"}),
                     self._body(signed={"miner_hotkey": "not an address"}),
                     self._body(signature="not hex"),
                     self._body(signed={"timestamp": "soon"})):
            with self.subTest(body=body):
                status, reply = self._post(body)
                self.assertEqual(status, 400)
                self.assertIn("error", reply)
        self.opening.assert_not_called()

    def test_a_bad_signature_is_401_and_covers_every_signed_field(self):
        for signature in ("00" * 64, "00" * 10):  # wrong, and of the wrong length
            status, reply = self._post(self._body(signature=signature))
            self.assertEqual(status, 401)
            self.assertIn("signature", reply["error"].lower())
        other_hotkey = Keypair.create_from_uri("//Charlie").ss58_address
        for field, value in (("leverage_tier", 3), ("miner_coldkey", other_hotkey), ("miner_hotkey", other_hotkey),
                             ("nonce", "another"), ("timestamp", TimeUtil.now_in_millis() + 30_000)):
            with self.subTest(field=field):
                status, reply = self._post(self._body(**{field: value}))
                self.assertEqual(status, 401)
        self.opening.assert_not_called()

    def test_a_coldkey_not_owning_the_hotkey_is_403(self):
        self.server._verify_coldkey_owns_hotkey.return_value = False
        status, reply = self._post(self._body())
        self.assertEqual(status, 403)
        self.opening.assert_not_called()

    def test_a_replayed_or_expired_request_is_401(self):
        body = self._body()
        self.assertEqual(self._post(body)[0], 200)
        status, reply = self._post(body)
        self.assertEqual(status, 401)
        self.assertIn("Nonce", reply["error"])
        status, reply = self._post(self._body(signed={"timestamp": TimeUtil.now_in_millis() - 6 * 60 * 1000}))
        self.assertEqual(status, 401)
        self.assertIn("expired", reply["error"])
        self.opening.assert_called_once()

    def test_the_nonce_is_consumed_only_after_the_signature_and_ownership_pass(self):
        nonce = uuid.uuid4().hex
        self.assertEqual(self._post(self._body(signed={"nonce": nonce}, signature="00" * 64))[0], 401)
        self.server._verify_coldkey_owns_hotkey.return_value = False
        self.assertEqual(self._post(self._body(signed={"nonce": nonce}))[0], 403)
        self.server._verify_coldkey_owns_hotkey.return_value = True
        self.assertEqual(self._post(self._body(signed={"nonce": nonce}))[0], 200)

    def test_a_rejection_is_400_with_the_reason(self):
        self.opening.return_value = (False, "Select an asset class before opening the account", None)
        status, reply = self._post(self._body())
        self.assertEqual(status, 400)
        self.assertEqual(reply, {"successfully_processed": False,
                                 "error_message": "Select an asset class before opening the account"})

    def test_a_validator_service_that_did_not_answer_is_503(self):
        self.opening.side_effect = RuntimeError("contract server unavailable")
        status, reply = self._post(self._body())
        self.assertEqual(status, 503)
        self.assertFalse(reply["successfully_processed"])
        self.assertIn("try again", reply["error_message"])

    def test_an_outdated_cli_is_400(self):
        status, reply = self._post(self._body(version="1.0.0"))
        self.assertEqual(status, 400)
        self.assertIn("outdated", reply["error"])
        self.opening.assert_not_called()


if __name__ == '__main__':
    unittest.main()
