"""
Unit tests for the leverage_tier field plumbing on standard subaccounts.

Covers:
  * create_subaccount: default tier, explicit tier, invalid tiers rejected before any state is
    written, HL subaccounts get None and reject an explicit tier; the tier lands on both
    SubaccountInfo and the MinerAccount.
  * SubaccountRegistration broadcast receive: tier persisted and pushed to MinerAccount;
    broadcasts from validators without the field leave it None.
  * MinerAccount disk format round trip, including records written before the field existed.
  * update_subaccount_leverage_tier: raise / lower / no-op, the open-position guard when caps may
    drop, ownership and HL guards, and propagation through broadcast receive and checkpoint sync.

  * The validator and gateway HTTP endpoints for the tier update: coldkey signature, field
    validation, and the payload forwarded to the validator.

Order-path behavior lives in test_standard_leverage_tiers.py.
"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bittensor_wallet import Keypair
from flask import Flask

from entity_management.entity_manager import EntityManager, SubaccountInfo
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.miner_account.miner_account_manager import MinerAccount, MinerAccountManager
from vali_objects.trade_pair import TradePair
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.position import Position

HL_ADDRESS = "0x" + "a" * 40


class TestLeverageTierPlumbing(TestBase):

    orchestrator = None
    entity_client = None
    miner_account_client = None
    metagraph_client = None

    ENTITY_HOTKEY = "entity_tiers"
    BROADCAST_ENTITY_HOTKEY = "entity_tier_bcast"

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.entity_client = cls.orchestrator.get_client('entity')
        cls.miner_account_client = cls.orchestrator.get_client('miner_account')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')

    def setUp(self):
        self.orchestrator.clear_all_test_data()
        self.metagraph_client.set_hotkeys([self.ENTITY_HOTKEY, self.BROADCAST_ENTITY_HOTKEY])
        success, msg = self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY)
        self.assertTrue(success, msg)

    def tearDown(self):
        self.orchestrator.clear_all_test_data()

    def _create(self, **kwargs):
        return self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY, account_size=100_000, asset_class="crypto", **kwargs
        )

    # ==================== create_subaccount ====================

    def test_default_tier_when_omitted(self):
        success, info, msg = self._create()
        self.assertTrue(success, msg)
        self.assertEqual(info['leverage_tier'], ValiConfig.STANDARD_LEVERAGE_TIER_DEFAULT)
        account = self.miner_account_client.get_account(info['synthetic_hotkey'])
        self.assertEqual(account.leverage_tier, ValiConfig.STANDARD_LEVERAGE_TIER_DEFAULT)

    def test_explicit_tier_lands_on_subaccount_and_account(self):
        for tier in ValiConfig.STANDARD_LEVERAGE_TIERS:
            with self.subTest(tier=tier):
                success, info, msg = self._create(leverage_tier=tier)
                self.assertTrue(success, msg)
                self.assertEqual(info['leverage_tier'], tier)
                account = self.miner_account_client.get_account(info['synthetic_hotkey'])
                self.assertEqual(account.leverage_tier, tier)

    def test_invalid_tier_rejected_before_any_state_is_written(self):
        for bad in (0, 4, -1, "2", 2.0, True):
            with self.subTest(tier=bad):
                success, info, msg = self._create(leverage_tier=bad)
                self.assertFalse(success)
                self.assertIsNone(info)
                self.assertIn("leverage_tier", msg)
        entity_data = self.entity_client.get_entity_data(self.ENTITY_HOTKEY)
        self.assertEqual(len(entity_data['subaccounts']), 0)

    def test_hl_subaccount_has_no_tier(self):
        success, info, msg = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY, account_size=100_000, hl_address=HL_ADDRESS,
        )
        self.assertTrue(success, msg)
        self.assertIsNone(info['leverage_tier'])
        account = self.miner_account_client.get_account(info['synthetic_hotkey'])
        self.assertIsNone(account.leverage_tier)

    def test_hl_subaccount_rejects_explicit_tier(self):
        manager = self._standalone_manager("guard_hotkey")
        success, info, msg = manager.create_subaccount(
            self.ENTITY_HOTKEY, 100_000, "hl_all", hl_address=HL_ADDRESS, leverage_tier=2,
        )
        self.assertFalse(success)
        self.assertIsNone(info)
        self.assertIn("leverage_tier", msg)

    # ==================== broadcast receive ====================

    def _standalone_manager(self, hotkey: str) -> EntityManager:
        config = SimpleNamespace(
            netuid=116,
            wallet=SimpleNamespace(hotkey=hotkey),
            subtensor=SimpleNamespace(network="test"),
        )
        manager = EntityManager(running_unit_tests=True, config=config, is_backtesting=False)
        manager.is_mothership = False
        return manager

    def _receive_broadcast(self, subaccount_data: dict) -> EntityManager:
        receiver = self._standalone_manager("receiver_hotkey")
        original = ValiConfig.MOTHERSHIP_HOTKEY
        ValiConfig.MOTHERSHIP_HOTKEY = "test_mothership_hotkey"
        try:
            ok = receiver.receive_subaccount_registration_update(
                subaccount_data=subaccount_data,
                sender_hotkey=ValiConfig.MOTHERSHIP_HOTKEY_TESTNET,
            )
        finally:
            ValiConfig.MOTHERSHIP_HOTKEY = original
        self.assertTrue(ok, "broadcast reception failed")
        return receiver

    def _broadcast_data(self, **extra) -> dict:
        data = {
            "entity_hotkey": self.BROADCAST_ENTITY_HOTKEY,
            "subaccount_id": 0,
            "subaccount_uuid": "uuid-tier-0",
            "synthetic_hotkey": f"{self.BROADCAST_ENTITY_HOTKEY}_0",
            "account_size": 50_000.0,
            "asset_class": "crypto",
        }
        data.update(extra)
        return data

    def test_broadcast_with_tier_sets_subaccount_and_account(self):
        receiver = self._receive_broadcast(self._broadcast_data(leverage_tier=3))
        sub = receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY).subaccounts[0]
        self.assertEqual(sub.leverage_tier, 3)
        account = self.miner_account_client.get_account(sub.synthetic_hotkey)
        self.assertEqual(account.leverage_tier, 3)

    def test_broadcast_without_tier_leaves_none(self):
        receiver = self._receive_broadcast(self._broadcast_data())
        sub = receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY).subaccounts[0]
        self.assertIsNone(sub.leverage_tier)
        account = self.miner_account_client.get_account(sub.synthetic_hotkey)
        self.assertIsNone(account.leverage_tier)


class TestLeverageTierUpdate(TestBase):
    """update_subaccount_leverage_tier through the entity client, plus the receive / sync paths
    that carry a changed tier to other validators."""

    orchestrator = None
    entity_client = None
    miner_account_client = None
    metagraph_client = None
    position_client = None

    ENTITY_HOTKEY = "entity_tierupd"
    OTHER_ENTITY_HOTKEY = "entity_other"
    BROADCAST_ENTITY_HOTKEY = "entity_tierupd_bcast"

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.entity_client = cls.orchestrator.get_client('entity')
        cls.miner_account_client = cls.orchestrator.get_client('miner_account')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.position_client = cls.orchestrator.get_client('position_manager')

    def setUp(self):
        self.orchestrator.clear_all_test_data()
        self.metagraph_client.set_hotkeys([self.ENTITY_HOTKEY, self.OTHER_ENTITY_HOTKEY, self.BROADCAST_ENTITY_HOTKEY])
        success, msg = self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY)
        self.assertTrue(success, msg)
        success, info, msg = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY, account_size=100_000, asset_class="crypto",
        )
        self.assertTrue(success, msg)
        self.synthetic = info['synthetic_hotkey']

    def tearDown(self):
        self.orchestrator.clear_all_test_data()

    def _update(self, tier, synthetic=None, entity=None):
        return self.entity_client.update_subaccount_leverage_tier(
            entity or self.ENTITY_HOTKEY, synthetic or self.synthetic, tier
        )

    def _stored_tier(self, synthetic=None, entity=None):
        synthetic = synthetic or self.synthetic
        entity_data = self.entity_client.get_entity_data(entity or self.ENTITY_HOTKEY)
        subs = entity_data['subaccounts'].values() if isinstance(entity_data, dict) else entity_data.subaccounts.values()
        for sub in subs:
            sub_hotkey = sub['synthetic_hotkey'] if isinstance(sub, dict) else sub.synthetic_hotkey
            if sub_hotkey == synthetic:
                return sub['leverage_tier'] if isinstance(sub, dict) else sub.leverage_tier
        self.fail(f"{synthetic} not found")

    def _account_tier(self, synthetic=None):
        return self.miner_account_client.get_account(synthetic or self.synthetic).leverage_tier

    def _open_position(self, synthetic):
        position = Position(
            miner_hotkey=synthetic, position_uuid=f"pos-{synthetic}", open_ms=TimeUtil.now_in_millis(),
            trade_pair=TradePair.BTCUSDC, position_type=OrderType.LONG, account_size=100_000.0,
        )
        self.position_client.save_miner_position(position)
        open_positions = self.position_client.get_positions_for_one_hotkey(synthetic, only_open_positions=True)
        self.assertEqual(len(open_positions), 1, "precondition: one open position")

    # ==================== happy paths ====================

    def test_raise_tier_updates_subaccount_and_account(self):
        success, msg = self._update(3)
        self.assertTrue(success, msg)
        self.assertEqual(self._stored_tier(), 3)
        self.assertEqual(self._account_tier(), 3)

    def test_same_tier_is_a_no_op(self):
        success, msg = self._update(ValiConfig.STANDARD_LEVERAGE_TIER_DEFAULT)
        self.assertTrue(success, msg)
        self.assertIn("already", msg)
        self.assertEqual(self._account_tier(), ValiConfig.STANDARD_LEVERAGE_TIER_DEFAULT)

    def test_lower_tier_without_positions(self):
        self.assertTrue(self._update(3)[0])
        success, msg = self._update(2)
        self.assertTrue(success, msg)
        self.assertEqual(self._stored_tier(), 2)
        self.assertEqual(self._account_tier(), 2)

    def test_raise_tier_with_open_position_is_allowed(self):
        self._open_position(self.synthetic)
        success, msg = self._update(2)
        self.assertTrue(success, msg)
        self.assertEqual(self._account_tier(), 2)

    # ==================== guards ====================

    def test_lower_tier_with_open_position_is_rejected(self):
        self.assertTrue(self._update(3)[0])
        self._open_position(self.synthetic)
        success, msg = self._update(1)
        self.assertFalse(success)
        self.assertIn("open position", msg)
        self.assertEqual(self._stored_tier(), 3)
        self.assertEqual(self._account_tier(), 3)

    def test_invalid_tier_unknown_subaccount_and_wrong_entity(self):
        for bad in (0, 4, "2", True):
            with self.subTest(tier=bad):
                self.assertFalse(self._update(bad)[0])
        success, msg = self._update(2, synthetic=f"{self.ENTITY_HOTKEY}_99")
        self.assertFalse(success)
        self.assertIn("not found", msg)
        success, msg = self._update(2, entity=self.OTHER_ENTITY_HOTKEY)
        self.assertFalse(success)
        self.assertIn("does not belong", msg)
        self.assertEqual(self._account_tier(), ValiConfig.STANDARD_LEVERAGE_TIER_DEFAULT)

    def test_hl_subaccount_is_rejected(self):
        success, info, msg = self.entity_client.create_hl_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY, account_size=100_000, hl_address=HL_ADDRESS,
        )
        self.assertTrue(success, msg)
        success, msg = self._update(2, synthetic=info['synthetic_hotkey'])
        self.assertFalse(success)
        self.assertIn("Hyperliquid", msg)
        self.assertIsNone(self._account_tier(info['synthetic_hotkey']))

    # ==================== pre-tier (legacy) subaccounts ====================

    def _receiver_with_legacy_subaccount(self, synthetic_suffix: int) -> tuple:
        receiver = EntityManager(
            running_unit_tests=True,
            config=SimpleNamespace(netuid=116, wallet=SimpleNamespace(hotkey="receiver_hotkey"),
                                   subtensor=SimpleNamespace(network="test")),
            is_backtesting=False,
        )
        receiver.is_mothership = False
        synthetic = f"{self.BROADCAST_ENTITY_HOTKEY}_{synthetic_suffix}"
        data = {
            "entity_hotkey": self.BROADCAST_ENTITY_HOTKEY,
            "subaccount_id": synthetic_suffix,
            "subaccount_uuid": f"uuid-upd-{synthetic_suffix}",
            "synthetic_hotkey": synthetic,
            "account_size": 50_000.0,
            "asset_class": "crypto",
        }
        original = ValiConfig.MOTHERSHIP_HOTKEY
        ValiConfig.MOTHERSHIP_HOTKEY = "test_mothership_hotkey"
        try:
            self.assertTrue(receiver.receive_subaccount_registration_update(
                subaccount_data=data, sender_hotkey=ValiConfig.MOTHERSHIP_HOTKEY_TESTNET,
            ))
        finally:
            ValiConfig.MOTHERSHIP_HOTKEY = original
        self.assertIsNone(receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY).subaccounts[synthetic_suffix].leverage_tier)
        return receiver, synthetic, data

    def test_legacy_subaccount_moves_onto_tiers_only_without_open_positions(self):
        receiver, synthetic, _ = self._receiver_with_legacy_subaccount(0)
        self._open_position(synthetic)
        success, msg = receiver.update_subaccount_leverage_tier(self.BROADCAST_ENTITY_HOTKEY, synthetic, 1)
        self.assertFalse(success)
        self.assertIn("open position", msg)
        self.assertIsNone(self._account_tier(synthetic))

        receiver, synthetic, _ = self._receiver_with_legacy_subaccount(1)
        success, msg = receiver.update_subaccount_leverage_tier(self.BROADCAST_ENTITY_HOTKEY, synthetic, 1)
        self.assertTrue(success, msg)
        self.assertEqual(receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY).subaccounts[1].leverage_tier, 1)
        self.assertEqual(self._account_tier(synthetic), 1)

    # ==================== propagation ====================

    def test_broadcast_for_existing_subaccount_updates_tier(self):
        receiver, synthetic, data = self._receiver_with_legacy_subaccount(0)
        original = ValiConfig.MOTHERSHIP_HOTKEY
        ValiConfig.MOTHERSHIP_HOTKEY = "test_mothership_hotkey"
        try:
            self.assertTrue(receiver.receive_subaccount_registration_update(
                subaccount_data={**data, "leverage_tier": 3}, sender_hotkey=ValiConfig.MOTHERSHIP_HOTKEY_TESTNET,
            ))
        finally:
            ValiConfig.MOTHERSHIP_HOTKEY = original
        self.assertEqual(receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY).subaccounts[0].leverage_tier, 3)
        self.assertEqual(self._account_tier(synthetic), 3)

    def test_checkpoint_sync_for_existing_subaccount_updates_tier(self):
        receiver, synthetic, _ = self._receiver_with_legacy_subaccount(0)
        checkpoint = {self.BROADCAST_ENTITY_HOTKEY: receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY).model_dump()}
        checkpoint[self.BROADCAST_ENTITY_HOTKEY]["subaccounts"][0]["leverage_tier"] = 2
        stats = receiver.sync_entity_data(checkpoint)
        self.assertEqual(stats['subaccounts_updated'], 1)
        self.assertEqual(receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY).subaccounts[0].leverage_tier, 2)
        self.assertEqual(self._account_tier(synthetic), 2)


class TestValidatorLeverageTierEndpoint(unittest.TestCase):
    """POST /entity/subaccount/leverage-tier on the validator, with a mocked entity client."""

    def setUp(self):
        from vanta_api.validator_rest_server import ValidatorRestServer

        self.coldkey = Keypair.create_from_uri("//Alice")
        self.hotkey = Keypair.create_from_uri("//Bob")
        self.server = object.__new__(ValidatorRestServer)
        self.server._entity_client = MagicMock()
        self.server._entity_client.update_subaccount_leverage_tier.return_value = (True, "updated")
        self.server._verify_coldkey_owns_hotkey = MagicMock(return_value=True)
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.route("/entity/subaccount/leverage-tier", methods=["POST"])(self.server.update_subaccount_leverage_tier)
        self.client = app.test_client()

    def _body(self, **overrides):
        signed = json.dumps({
            "entity_coldkey": self.coldkey.ss58_address,
            "entity_hotkey": self.hotkey.ss58_address,
        }, sort_keys=True).encode("utf-8")
        body = {
            "entity_coldkey": self.coldkey.ss58_address,
            "entity_hotkey": self.hotkey.ss58_address,
            "synthetic_hotkey": f"{self.hotkey.ss58_address}_0",
            "leverage_tier": 2,
            "signature": self.coldkey.sign(signed).hex(),
            "version": "2.2.1",
        }
        body.update(overrides)
        return body

    def _post(self, body):
        resp = self.client.post("/entity/subaccount/leverage-tier", json=body)
        return resp.status_code, json.loads(resp.data)

    def test_valid_request_reaches_entity_client(self):
        status, data = self._post(self._body())
        self.assertEqual(status, 200)
        self.assertEqual(data['leverage_tier'], 2)
        self.server._entity_client.update_subaccount_leverage_tier.assert_called_once_with(
            self.hotkey.ss58_address, f"{self.hotkey.ss58_address}_0", 2
        )

    def test_manager_rejection_is_a_400(self):
        self.server._entity_client.update_subaccount_leverage_tier.return_value = (False, "Close all open positions")
        status, data = self._post(self._body())
        self.assertEqual(status, 400)
        self.assertIn("open positions", data['error'])

    def test_bad_signature_is_401(self):
        other = Keypair.create_from_uri("//Charlie")
        status, _ = self._post(self._body(signature=other.sign(b"anything").hex()))
        self.assertEqual(status, 401)
        self.server._entity_client.update_subaccount_leverage_tier.assert_not_called()

    def test_coldkey_not_owning_hotkey_is_403(self):
        self.server._verify_coldkey_owns_hotkey.return_value = False
        status, _ = self._post(self._body())
        self.assertEqual(status, 403)
        self.server._entity_client.update_subaccount_leverage_tier.assert_not_called()

    def test_invalid_tier_and_missing_fields_are_400(self):
        for bad in (0, 4, "2", True):
            with self.subTest(tier=bad):
                status, _ = self._post(self._body(leverage_tier=bad))
                self.assertEqual(status, 400)
        body = self._body()
        del body["synthetic_hotkey"]
        status, data = self._post(body)
        self.assertEqual(status, 400)
        self.assertIn("synthetic_hotkey", data['error'])
        self.server._entity_client.update_subaccount_leverage_tier.assert_not_called()


class TestGatewayLeverageTierEndpoint(unittest.TestCase):
    """POST /api/update-subaccount-leverage-tier on the gateway, with the validator call patched."""

    def setUp(self):
        from vanta_api.entity_miner_rest_server import EntityMinerRestServer

        self.coldkey = Keypair.create_from_uri("//Alice")
        self.hotkey = Keypair.create_from_uri("//Bob")
        self.gw = object.__new__(EntityMinerRestServer)
        self.gw._coldkey = self.coldkey
        self.gw._hotkey = self.hotkey
        self.gw._validator_url = "http://validator.test"
        self.gw._get_api_key_safe = MagicMock(return_value="key")
        self.gw.is_valid_api_key = MagicMock(return_value=True)
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.route("/api/update-subaccount-leverage-tier", methods=["POST"])(self.gw.update_subaccount_leverage_tier_endpoint)
        self.client = app.test_client()

    def _post(self, body):
        resp = self.client.post("/api/update-subaccount-leverage-tier", json=body)
        return resp.status_code, json.loads(resp.data)

    def test_forwards_signed_payload_to_validator(self):
        validator_resp = MagicMock(status_code=200)
        validator_resp.json.return_value = {"status": "success", "leverage_tier": 3}
        with patch("requests.post", return_value=validator_resp) as post:
            status, data = self._post({"synthetic_hotkey": f"{self.hotkey.ss58_address}_0", "leverage_tier": 3})
        self.assertEqual(status, 200)
        self.assertEqual(data['leverage_tier'], 3)
        url = post.call_args.args[0]
        payload = post.call_args.kwargs['json']
        self.assertEqual(url, "http://validator.test/entity/subaccount/leverage-tier")
        self.assertEqual(payload['synthetic_hotkey'], f"{self.hotkey.ss58_address}_0")
        self.assertEqual(payload['leverage_tier'], 3)
        signed = json.dumps({
            "entity_coldkey": self.coldkey.ss58_address,
            "entity_hotkey": self.hotkey.ss58_address,
        }, sort_keys=True).encode("utf-8")
        self.assertTrue(Keypair(ss58_address=self.coldkey.ss58_address).verify(signed, bytes.fromhex(payload['signature'])))

    def test_validator_error_is_passed_through(self):
        validator_resp = MagicMock(status_code=400)
        validator_resp.json.return_value = {"error": "Close all open positions"}
        with patch("requests.post", return_value=validator_resp):
            status, data = self._post({"synthetic_hotkey": f"{self.hotkey.ss58_address}_0", "leverage_tier": 1})
        self.assertEqual(status, 400)
        self.assertIn("open positions", data['message'])

    def test_invalid_input_never_reaches_validator(self):
        with patch("requests.post") as post:
            for body in (
                {"synthetic_hotkey": f"{self.hotkey.ss58_address}_0", "leverage_tier": 4},
                {"synthetic_hotkey": f"{self.hotkey.ss58_address}_0", "leverage_tier": "2"},
                {"leverage_tier": 2},
            ):
                with self.subTest(body=body):
                    status, _ = self._post(body)
                    self.assertEqual(status, 400)
            post.assert_not_called()

    def test_bad_api_key_is_401(self):
        self.gw.is_valid_api_key.return_value = False
        with patch("requests.post") as post:
            status, _ = self._post({"synthetic_hotkey": f"{self.hotkey.ss58_address}_0", "leverage_tier": 2})
        self.assertEqual(status, 401)
        post.assert_not_called()


class TestLeverageTierModels(unittest.TestCase):
    """Pure model / disk-format checks, no servers."""

    def test_subaccount_info_defaults_and_dumps(self):
        legacy_fields = dict(
            subaccount_id=0, subaccount_uuid="u", synthetic_hotkey="e_0", created_at_ms=1,
            account_size=10_000.0, asset_class="crypto",
        )
        self.assertIsNone(SubaccountInfo(**legacy_fields).leverage_tier)
        info = SubaccountInfo(**legacy_fields, leverage_tier=2)
        self.assertEqual(info.model_dump()['leverage_tier'], 2)

    def test_miner_account_disk_round_trip(self):
        record = MinerAccount(miner_hotkey="ent_1", leverage_tier=2).to_dict()
        self.assertEqual(record['leverage_tier'], 2)
        parsed = MinerAccountManager._parse_accounts_dict({"ent_1": [record]})
        self.assertEqual(parsed["ent_1"].leverage_tier, 2)

    def test_miner_account_record_without_field_parses_to_none(self):
        record = MinerAccount(miner_hotkey="ent_1").to_dict()
        del record['leverage_tier']
        parsed = MinerAccountManager._parse_accounts_dict({"ent_1": [record]})
        self.assertIsNone(parsed["ent_1"].leverage_tier)

    def test_tier_validator(self):
        for good in ValiConfig.STANDARD_LEVERAGE_TIERS:
            self.assertTrue(ValiConfig.is_valid_standard_leverage_tier(good))
        for bad in (0, 4, "2", 2.0, True, None):
            with self.subTest(tier=bad):
                self.assertFalse(ValiConfig.is_valid_standard_leverage_tier(bad))


if __name__ == "__main__":
    unittest.main()
