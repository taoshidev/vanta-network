"""
The entity whitelist: hotkeys that register listed roles without the registration fee. The whitelist itself
(add, replace, remove, validation, persistence), the waiver at registration and its role scope, the record on the
entity, and the tier-500 admin endpoints. The fee is never slashed in unit tests (EntityManager skips the
collateral check when running_unit_tests), so the waiver is observed on the entity record and the message.
"""
import json
import os
import unittest
from unittest.mock import MagicMock

from flask import Flask

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.enums.entity_role_enum import EntityRole
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vanta_api.validator_rest_server import ValidatorRestServer

EM = EntityRole.ENTITY_MINER.value
PEM = EntityRole.PERMISSIONLESS_ENTITY_MINER.value


class TestEntityWhitelist(TestBase):

    orchestrator = None
    entity_client = None

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=ValiUtils.get_secrets(running_unit_tests=True))
        cls.entity_client = cls.orchestrator.get_client('entity')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')

    def setUp(self):
        self.orchestrator.clear_all_test_data()
        self.AFFILIATE = "entity_affiliate"
        self.OTHER = "entity_other"
        self.metagraph_client.set_hotkeys([self.AFFILIATE, self.OTHER])

    def tearDown(self):
        self.orchestrator.clear_all_test_data()

    # ==================== The whitelist ====================

    def test_add_list_replace_and_remove(self):
        self.assertEqual(self.entity_client.get_entity_whitelist(), {})

        success, message = self.entity_client.set_entity_whitelist_entry(self.AFFILIATE, [PEM], "Acme Exchange")
        self.assertTrue(success, message)
        entry = self.entity_client.get_entity_whitelist()[self.AFFILIATE]
        self.assertEqual(entry["roles"], [PEM])
        self.assertEqual(entry["note"], "Acme Exchange")
        self.assertIsNotNone(entry["added_at_ms"])

        # Replaced, with duplicates dropped
        self.assertTrue(self.entity_client.set_entity_whitelist_entry(self.AFFILIATE, [PEM, EM, PEM])[0])
        entry = self.entity_client.get_entity_whitelist()[self.AFFILIATE]
        self.assertEqual(entry["roles"], sorted([PEM, EM]))
        self.assertIsNone(entry["note"])

        success, message = self.entity_client.remove_entity_whitelist_entry(self.AFFILIATE)
        self.assertTrue(success, message)
        self.assertEqual(self.entity_client.get_entity_whitelist(), {})
        success, message = self.entity_client.remove_entity_whitelist_entry(self.AFFILIATE)
        self.assertFalse(success)
        self.assertIn("not on the entity whitelist", message)

    def test_invalid_entries_are_refused(self):
        for hotkey, roles, note in (
            ("", [PEM], None),
            (None, [PEM], None),
            (f"{self.AFFILIATE}_0", [PEM], None),       # a synthetic subaccount hotkey
            (self.AFFILIATE, [], None),
            (self.AFFILIATE, PEM, None),                 # not a list
            (self.AFFILIATE, ["pem"], None),
            (self.AFFILIATE, [PEM], "x" * 257),
            (self.AFFILIATE, [PEM], 42),
        ):
            with self.subTest(hotkey=hotkey, roles=roles, note=note):
                success, _ = self.entity_client.set_entity_whitelist_entry(hotkey, roles, note)
                self.assertFalse(success)
        self.assertEqual(self.entity_client.get_entity_whitelist(), {})

    def test_the_whitelist_is_written_to_disk(self):
        self.entity_client.set_entity_whitelist_entry(self.AFFILIATE, [PEM], "Acme Exchange")
        path = ValiBkpUtils.get_entity_whitelist_file_location(running_unit_tests=True)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            on_disk = json.load(f)
        self.assertEqual(on_disk[self.AFFILIATE]["roles"], [PEM])
        self.entity_client.remove_entity_whitelist_entry(self.AFFILIATE)
        with open(path) as f:
            self.assertEqual(json.load(f), {})

    # ==================== The waiver at registration ====================

    def test_a_whitelisted_hotkey_registers_the_role_without_the_fee(self):
        self.entity_client.set_entity_whitelist_entry(self.AFFILIATE, [PEM])
        success, message = self.entity_client.register_entity(entity_hotkey=self.AFFILIATE, role=PEM, pem_take=0.1)
        self.assertTrue(success, message)
        self.assertIn("waived", message)
        data = self.entity_client.get_entity_data(self.AFFILIATE)
        self.assertEqual(data["roles"], [PEM])
        self.assertEqual(data["fee_waived_roles"], [PEM])

    def test_a_hotkey_not_on_the_whitelist_pays(self):
        self.entity_client.set_entity_whitelist_entry(self.OTHER, [PEM])
        success, message = self.entity_client.register_entity(entity_hotkey=self.AFFILIATE, role=PEM, pem_take=0.1)
        self.assertTrue(success, message)
        self.assertIn("theta registration fee slashed", message)
        self.assertEqual(self.entity_client.get_entity_data(self.AFFILIATE)["fee_waived_roles"], [])

    def test_the_waiver_is_scoped_to_the_listed_roles(self):
        self.entity_client.set_entity_whitelist_entry(self.AFFILIATE, [PEM])
        success, message = self.entity_client.register_entity(entity_hotkey=self.AFFILIATE)
        self.assertTrue(success, message)
        self.assertIn("theta registration fee slashed", message)
        self.assertEqual(self.entity_client.get_entity_data(self.AFFILIATE)["fee_waived_roles"], [])

        # The listed role, added to the registered entity, is waived
        success, message = self.entity_client.register_entity(entity_hotkey=self.AFFILIATE, role=PEM, pem_take=0.2)
        self.assertTrue(success, message)
        self.assertIn("waived", message)
        data = self.entity_client.get_entity_data(self.AFFILIATE)
        self.assertEqual(data["roles"], [EM, PEM])
        self.assertEqual(data["fee_waived_roles"], [PEM])

    def test_removing_the_entry_keeps_the_registered_role(self):
        self.entity_client.set_entity_whitelist_entry(self.AFFILIATE, [PEM])
        self.entity_client.register_entity(entity_hotkey=self.AFFILIATE, role=PEM, pem_take=0.1)
        self.entity_client.remove_entity_whitelist_entry(self.AFFILIATE)
        data = self.entity_client.get_entity_data(self.AFFILIATE)
        self.assertEqual(data["roles"], [PEM])
        self.assertEqual(data["fee_waived_roles"], [PEM])
        self.assertTrue(self.entity_client.is_permissionless_entity_miner(self.AFFILIATE))

    def test_registration_checks_still_apply_to_a_whitelisted_hotkey(self):
        self.entity_client.set_entity_whitelist_entry(self.AFFILIATE, [PEM])
        success, message = self.entity_client.register_entity(entity_hotkey=self.AFFILIATE, role=PEM, pem_take=0.05)
        self.assertFalse(success)
        self.assertIn("pem_take", message)
        self.assertTrue(self.entity_client.register_entity(entity_hotkey=self.AFFILIATE, role=PEM, pem_take=0.1)[0])
        success, message = self.entity_client.register_entity(entity_hotkey=self.AFFILIATE, role=PEM, pem_take=0.1)
        self.assertFalse(success)
        self.assertIn("already registered", message)

    def test_the_waiver_is_synced_with_the_entity(self):
        self.entity_client.set_entity_whitelist_entry(self.AFFILIATE, [PEM])
        self.entity_client.register_entity(entity_hotkey=self.AFFILIATE, role=PEM, pem_take=0.1)
        checkpoint = self.entity_client.to_checkpoint_dict()
        self.assertEqual(checkpoint[self.AFFILIATE]["fee_waived_roles"], [PEM])

        self.orchestrator.clear_all_test_data()
        self.entity_client.sync_entity_data(checkpoint)
        self.assertEqual(self.entity_client.get_entity_data(self.AFFILIATE)["fee_waived_roles"], [PEM])
        # The whitelist itself is not part of the entity checkpoint
        self.assertEqual(self.entity_client.get_entity_whitelist(), {})


class TestEntityWhitelistEndpoints(unittest.TestCase):

    def setUp(self):
        server = object.__new__(ValidatorRestServer)
        server._entity_client = MagicMock()
        server._entity_client.get_entity_whitelist.return_value = {"5Aff": {"roles": [PEM], "note": "Acme", "added_at_ms": 1}}
        server._entity_client.set_entity_whitelist_entry.return_value = (True, "5Aff whitelisted")
        server._entity_client.remove_entity_whitelist_entry.return_value = (True, "5Aff removed")
        server.is_valid_api_key = lambda key: key in ("admin", "reader")
        server.can_access_tier = lambda key, tier: key == "admin"
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.route("/admin/entity-whitelist", methods=["GET"])(server.get_entity_whitelist)
        app.route("/admin/entity-whitelist", methods=["POST"])(server.set_entity_whitelist_entry)
        app.route("/admin/entity-whitelist/<hotkey>", methods=["DELETE"])(server.remove_entity_whitelist_entry)
        self.server, self.client = server, app.test_client()

    def _call(self, method, path, key="admin", body=None):
        response = getattr(self.client, method)(path, json=body, headers={"Authorization": f"Bearer {key}"} if key else {})
        return response.status_code, response.get_json()

    def test_admin_access_is_required(self):
        for method, path, body in (("get", "/admin/entity-whitelist", None),
                                   ("post", "/admin/entity-whitelist", {"hotkey": "5Aff", "roles": [PEM]}),
                                   ("delete", "/admin/entity-whitelist/5Aff", None)):
            with self.subTest(method=method):
                self.assertEqual(self._call(method, path, key=None, body=body)[0], 401)
                self.assertEqual(self._call(method, path, key="reader", body=body)[0], 403)
        self.server._entity_client.set_entity_whitelist_entry.assert_not_called()
        self.server._entity_client.remove_entity_whitelist_entry.assert_not_called()

    def test_list(self):
        code, reply = self._call("get", "/admin/entity-whitelist")
        self.assertEqual(code, 200)
        self.assertEqual(reply["whitelist"]["5Aff"]["roles"], [PEM])

    def test_add(self):
        code, reply = self._call("post", "/admin/entity-whitelist", body={"hotkey": "5Aff", "roles": [PEM], "note": "Acme"})
        self.assertEqual(code, 200, reply)
        self.server._entity_client.set_entity_whitelist_entry.assert_called_once_with("5Aff", [PEM], "Acme")

    def test_add_without_hotkey_or_roles_is_400(self):
        for body in ({"roles": [PEM]}, {"hotkey": "5Aff"}, [1, 2]):
            with self.subTest(body=body):
                self.assertEqual(self._call("post", "/admin/entity-whitelist", body=body)[0], 400)
        self.server._entity_client.set_entity_whitelist_entry.assert_not_called()

    def test_a_refused_entry_is_400(self):
        self.server._entity_client.set_entity_whitelist_entry.return_value = (False, "roles must be a non-empty list")
        code, reply = self._call("post", "/admin/entity-whitelist", body={"hotkey": "5Aff", "roles": []})
        self.assertEqual(code, 400)
        self.assertIn("roles", reply["error"])

    def test_remove(self):
        code, _ = self._call("delete", "/admin/entity-whitelist/5Aff")
        self.assertEqual(code, 200)
        self.server._entity_client.remove_entity_whitelist_entry.assert_called_once_with("5Aff")

    def test_removing_an_unknown_hotkey_is_404(self):
        self.server._entity_client.remove_entity_whitelist_entry.return_value = (False, "5Aff is not on the entity whitelist")
        self.assertEqual(self._call("delete", "/admin/entity-whitelist/5Aff")[0], 404)


if __name__ == "__main__":
    unittest.main()
