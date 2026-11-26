"""
POST /entity/register on the validator REST server, against the real Flask handler with the entity client
mocked: the requests the CLI signs for each role, the request signed before roles existed, role and take
validation, that role and take are covered by the coldkey signature, and the entity API key a success returns
(the same one POST /request-api-key issues, created once even under concurrent requests).
"""
import json
import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from bittensor_wallet import Keypair
from flask import Flask

from vali_objects.enums.entity_role_enum import EntityRole
from vali_objects.vali_config import ValiConfig
from vanta_api.validator_rest_server import ValidatorRestServer
from vanta_cli.src.commands.entity import registration_request

EM = EntityRole.ENTITY_MINER.value
PEM = EntityRole.PERMISSIONLESS_ENTITY_MINER.value


def _client(api_keys_file):
    """Flask test client for the endpoints, with the entity client and the ownership check mocked, and the API
    keys in `api_keys_file`."""
    server = object.__new__(ValidatorRestServer)
    server._entity_client = MagicMock()
    server._entity_client.register_entity.return_value = (True, "registered")
    server._entity_client.get_entity_data.return_value = {"entity_hotkey": "registered"}
    server._verify_coldkey_owns_hotkey = MagicMock(return_value=True)
    server.api_keys_file = api_keys_file
    server.api_key_to_alias = {}
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.route("/entity/register", methods=["POST"])(server.register_entity)
    app.route("/request-api-key", methods=["POST"])(server.request_entity_api_key)
    return server, app.test_client()


class TestEntityRegisterEndpoint(unittest.TestCase):

    def setUp(self):
        self.coldkey = Keypair.create_from_uri("//Alice")
        self.hotkey = Keypair.create_from_uri("//Bob")
        self.tmp_dir = tempfile.mkdtemp()
        self.api_keys_file = os.path.join(self.tmp_dir, "api_keys.json")
        self.server, self.client = _client(self.api_keys_file)
        self.register = self.server._entity_client.register_entity

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _stored_keys(self):
        with open(self.api_keys_file) as f:
            return json.load(f)

    def _signed(self, fields, **tampered):
        """`fields` signed by the coldkey, then `tampered` applied after signing."""
        message = json.dumps(fields, sort_keys=True).encode("utf-8")
        body = {**fields, "signature": self.coldkey.sign(message).hex(), "version": ValiConfig.VANTA_CLI_MINIMUM_VERSION}
        body.update(tampered)
        return body

    def _base(self):
        return {"entity_coldkey": self.coldkey.ss58_address, "entity_hotkey": self.hotkey.ss58_address}

    def _post(self, body):
        response = self.client.post("/entity/register", json=body)
        return response.status_code, response.get_json()

    def test_the_entity_miner_request_the_cli_signs_is_accepted(self):
        payload = {**registration_request(self.coldkey, self.hotkey.ss58_address), "version": ValiConfig.VANTA_CLI_MINIMUM_VERSION}
        self.assertNotIn("role", payload)
        status, reply = self._post(payload)
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["role"], EM)
        self.register.assert_called_once_with(entity_hotkey=self.hotkey.ss58_address, role=EM, pem_take=None)

    def test_the_permissionless_request_the_cli_signs_is_accepted(self):
        payload = {**registration_request(self.coldkey, self.hotkey.ss58_address, PEM, 0.1),
                   "version": ValiConfig.VANTA_CLI_MINIMUM_VERSION}
        status, reply = self._post(payload)
        self.assertEqual(status, 200, reply)
        self.assertEqual(reply["role"], PEM)
        self.register.assert_called_once_with(entity_hotkey=self.hotkey.ss58_address, role=PEM, pem_take=0.1)

    def test_an_explicit_entity_miner_role_is_accepted_when_signed(self):
        status, reply = self._post(self._signed({**self._base(), "role": EM}))
        self.assertEqual(status, 200, reply)
        self.register.assert_called_once_with(entity_hotkey=self.hotkey.ss58_address, role=EM, pem_take=None)

    def test_role_and_take_must_be_signed(self):
        for body in (
            # Signed without a role, sent with one
            self._signed(self._base(), role=PEM, pem_take=0.1),
            # Signed at one take, sent at another
            self._signed({**self._base(), "role": PEM, "pem_take": 0.5}, pem_take=0.1),
            # Signed as a permissionless entity miner, sent as an entity miner
            self._signed({**self._base(), "role": PEM, "pem_take": 0.1}, role=EM, pem_take=None),
        ):
            with self.subTest(body=body):
                if body.get("pem_take") is None:
                    body.pop("pem_take", None)
                status, reply = self._post(body)
                self.assertEqual(status, 401, reply)
        self.register.assert_not_called()

    def test_role_and_take_are_validated_before_the_signature(self):
        for fields in (
            {"role": "pem", "pem_take": 0.1},
            {"role": PEM},
            {"role": PEM, "pem_take": "0.1"},
            {"role": PEM, "pem_take": True},
            {"role": EM, "pem_take": 0.1},
            {"pem_take": 0.1},
        ):
            with self.subTest(fields=fields):
                status, reply = self._post(self._signed({**self._base(), **fields}))
                self.assertEqual(status, 400, reply)
        self.server._verify_coldkey_owns_hotkey.assert_not_called()
        self.register.assert_not_called()

    def test_a_coldkey_that_does_not_own_the_hotkey_is_refused(self):
        self.server._verify_coldkey_owns_hotkey.return_value = False
        status, reply = self._post(self._signed({**self._base(), "role": PEM, "pem_take": 0.1}))
        self.assertEqual(status, 403, reply)
        self.register.assert_not_called()

    def test_a_refusal_from_the_entity_manager_is_400(self):
        self.register.return_value = (False, "pem_take must be at least 0.1 and below 1, got 0.05")
        status, reply = self._post(self._signed({**self._base(), "role": PEM, "pem_take": 0.05}))
        self.assertEqual(status, 400)
        self.assertIn("pem_take", reply["error"])

    # ==================== The entity API key ====================

    def _api_key_request(self):
        fields = self._base()
        message = json.dumps(fields, sort_keys=True).encode("utf-8")
        return {**fields, "signature": self.coldkey.sign(message).hex()}

    def test_a_success_returns_the_entity_api_key(self):
        code, reply = self._post(self._signed({**self._base(), "role": PEM, "pem_take": 0.1}))
        self.assertEqual(code, 200, reply)
        self.assertTrue(reply["api_key"])
        stored = self._stored_keys()[self.hotkey.ss58_address]
        self.assertEqual(stored["key"], reply["api_key"])
        self.assertEqual(stored["tier"], ValiConfig.SUBACCOUNT_SUBSCRIPTION_TIER)

    def test_every_registration_and_request_api_key_return_the_same_key(self):
        first = self._post(self._signed({**self._base(), "role": PEM, "pem_take": 0.1}))[1]["api_key"]
        # Adding the other role, before the alias map has reloaded the file
        second = self._post(self._signed(self._base()))[1]["api_key"]
        response = self.client.post("/request-api-key", json=self._api_key_request())
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual({first, second, response.get_json()["api_key"]}, {first})
        self.assertEqual(len(self._stored_keys()), 1)

    def test_a_key_already_issued_is_returned(self):
        self.server.api_key_to_alias = {"an-existing-key": self.hotkey.ss58_address}
        code, reply = self._post(self._signed(self._base()))
        self.assertEqual(code, 200, reply)
        self.assertEqual(reply["api_key"], "an-existing-key")
        self.assertFalse(os.path.exists(self.api_keys_file))

    def test_a_key_that_cannot_be_issued_leaves_the_registration_standing(self):
        with patch("vanta_api.validator_rest_server.ValiBkpUtils.write_file", side_effect=OSError("disk full")):
            code, reply = self._post(self._signed({**self._base(), "role": PEM, "pem_take": 0.1}))
        self.assertEqual(code, 200, reply)
        self.assertEqual(reply["status"], "success")
        self.assertIsNone(reply["api_key"])
        self.assertIn("/request-api-key", reply["api_key_error"])

    def test_a_failed_registration_returns_no_key(self):
        self.register.return_value = (False, "Insufficient collateral")
        code, reply = self._post(self._signed(self._base()))
        self.assertEqual(code, 400)
        self.assertNotIn("api_key", reply)
        self.assertFalse(os.path.exists(self.api_keys_file))

    def test_concurrent_requests_create_one_key(self):
        bodies = [self._signed(self._base()) if i % 2 else self._api_key_request() for i in range(16)]

        def send(i_body):
            i, body = i_body
            path = "/request-api-key" if i % 2 == 0 else "/entity/register"
            response = self.server_app_client().post(path, json=body)
            return response.get_json()["api_key"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            keys = set(pool.map(send, enumerate(bodies)))
        self.assertEqual(len(keys), 1)
        self.assertEqual(self._stored_keys()[self.hotkey.ss58_address]["key"], keys.pop())

    def server_app_client(self):
        # One test client per request: Flask test clients are not shared across threads
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.route("/entity/register", methods=["POST"])(self.server.register_entity)
        app.route("/request-api-key", methods=["POST"])(self.server.request_entity_api_key)
        return app.test_client()


if __name__ == "__main__":
    unittest.main()
