"""
Unit tests for the leverage_tier field plumbing on standard subaccounts.

Covers:
  * create_subaccount: default tier, explicit tier, invalid tiers rejected before any state is
    written, HL subaccounts get None and reject an explicit tier; the tier lands on both
    SubaccountInfo and the MinerAccount.
  * SubaccountRegistration broadcast receive: tier persisted and pushed to MinerAccount;
    broadcasts from validators without the field leave it None.
  * MinerAccount disk format round trip, including records written before the field existed.

Nothing reads leverage_tier yet, so no order-path behavior is asserted here.
"""
import unittest
from types import SimpleNamespace

from entity_management.entity_manager import EntityManager, SubaccountInfo
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.miner_account.miner_account_manager import MinerAccount, MinerAccountManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig

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
