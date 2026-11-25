"""
POST /entity/register on the validator REST server, against the real Flask handler with the entity client
mocked: the requests the CLI signs for each role, the request signed before roles existed, role and take
validation, and that role and take are covered by the coldkey signature.
"""
import json
import unittest
from unittest.mock import MagicMock

from bittensor_wallet import Keypair
from flask import Flask

from vali_objects.enums.entity_role_enum import EntityRole
from vali_objects.vali_config import ValiConfig
from vanta_api.validator_rest_server import ValidatorRestServer
from vanta_cli.src.commands.entity import registration_request

EM = EntityRole.ENTITY_MINER.value
PEM = EntityRole.PERMISSIONLESS_ENTITY_MINER.value


def _client():
    """Flask test client for the endpoint, with the entity client and the ownership check mocked."""
    server = object.__new__(ValidatorRestServer)
    server._entity_client = MagicMock()
    server._entity_client.register_entity.return_value = (True, "registered")
    server._verify_coldkey_owns_hotkey = MagicMock(return_value=True)
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.route("/entity/register", methods=["POST"])(server.register_entity)
    return server, app.test_client()


class TestEntityRegisterEndpoint(unittest.TestCase):

    def setUp(self):
        self.coldkey = Keypair.create_from_uri("//Alice")
        self.hotkey = Keypair.create_from_uri("//Bob")
        self.server, self.client = _client()
        self.register = self.server._entity_client.register_entity

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


if __name__ == "__main__":
    unittest.main()
