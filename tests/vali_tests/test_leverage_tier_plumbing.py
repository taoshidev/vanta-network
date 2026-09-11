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
    drop, ownership, HL and hl_all guards, and propagation through broadcast receive and checkpoint
    sync, including the MinerAccount repair when the account lost the field and the rejection of
    out-of-range tiers arriving from other validators.
  * An order for a standard subaccount through MarketOrderManager: per-pair and class caps at
    tier 1, more room after raising the tier, lowering only with a flat book.

  * The validator and gateway HTTP endpoints for the tier update: coldkey signature bound to the
    target subaccount and tier, nonce + timestamp replay protection, field validation, and the
    payload forwarded to the validator.

Order-path behavior lives in test_standard_leverage_tiers.py.
"""
import json
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from bittensor_wallet import Keypair
from flask import Flask

from entity_management.entity_manager import EntityManager, SubaccountInfo
from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.execution_type_enum import ExecutionType
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.miner_account.miner_account_manager import MinerAccount, MinerAccountManager
from vali_objects.trade_pair import TradePair
from vali_objects.utils.limit_order.order_utils import OrderSize
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_dataclasses.price_source import PriceSource

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

    def test_same_tier_repairs_miner_account_that_lost_the_field(self):
        self.assertTrue(self._update(3)[0])
        self.miner_account_client.set_leverage_tier(self.synthetic, None)
        self.assertIsNone(self._account_tier())
        success, msg = self._update(3)
        self.assertTrue(success, msg)
        self.assertEqual(self._account_tier(), 3)

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

    # ==================== subaccounts without a stored tier ====================

    @staticmethod
    def _receive(receiver: EntityManager, data: dict) -> bool:
        """Deliver a SubaccountRegistration broadcast to `receiver` as if sent by the mothership."""
        original = ValiConfig.MOTHERSHIP_HOTKEY
        ValiConfig.MOTHERSHIP_HOTKEY = "test_mothership_hotkey"
        try:
            return receiver.receive_subaccount_registration_update(
                subaccount_data=data, sender_hotkey=ValiConfig.MOTHERSHIP_HOTKEY_TESTNET,
            )
        finally:
            ValiConfig.MOTHERSHIP_HOTKEY = original

    def _receiver_with_legacy_subaccount(self, synthetic_suffix: int, asset_class: str = "crypto") -> tuple:
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
            "asset_class": asset_class,
        }
        self.assertTrue(self._receive(receiver, data))
        self.assertIsNone(receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY).subaccounts[synthetic_suffix].leverage_tier)
        return receiver, synthetic, data

    def test_legacy_subaccount_counts_as_default_tier_when_changed(self):
        # A pre-tier subaccount already trades at the default tier, so recording tier 1 or raising
        # to tier 2 is never a lowering and works even with open positions.
        receiver, synthetic, _ = self._receiver_with_legacy_subaccount(0)
        self._open_position(synthetic)
        success, msg = receiver.update_subaccount_leverage_tier(self.BROADCAST_ENTITY_HOTKEY, synthetic, 1)
        self.assertTrue(success, msg)
        self.assertEqual(receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY).subaccounts[0].leverage_tier, 1)
        self.assertEqual(self._account_tier(synthetic), 1)

        receiver, synthetic, _ = self._receiver_with_legacy_subaccount(1)
        self._open_position(synthetic)
        success, msg = receiver.update_subaccount_leverage_tier(self.BROADCAST_ENTITY_HOTKEY, synthetic, 2)
        self.assertTrue(success, msg)
        self.assertEqual(self._account_tier(synthetic), 2)

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

    # ==================== guards on what other validators send ====================

    def test_hl_all_legacy_subaccount_is_rejected(self):
        # A pre-migration standard subaccount with asset_class hl_all stays on the legacy curve, so a
        # stored tier would never be applied; refuse to store one
        receiver, synthetic, _ = self._receiver_with_legacy_subaccount(0, asset_class="hl_all")
        success, msg = receiver.update_subaccount_leverage_tier(self.BROADCAST_ENTITY_HOTKEY, synthetic, 2)
        self.assertFalse(success)
        self.assertIn("hl_all", msg)
        self.assertIsNone(receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY).subaccounts[0].leverage_tier)
        self.assertIsNone(self._account_tier(synthetic))

    def test_out_of_range_tier_from_broadcast_or_checkpoint_is_ignored(self):
        receiver, synthetic, data = self._receiver_with_legacy_subaccount(0)
        entity = lambda: receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY)  # noqa: E731

        # existing subaccount, broadcast
        self.assertTrue(self._receive(receiver, {**data, "leverage_tier": 7}))
        self.assertIsNone(entity().subaccounts[0].leverage_tier)
        self.assertIsNone(self._account_tier(synthetic))

        # new subaccount, broadcast
        new_sub = {**data, "subaccount_id": 1, "subaccount_uuid": "uuid-upd-1",
                   "synthetic_hotkey": f"{self.BROADCAST_ENTITY_HOTKEY}_1", "leverage_tier": 9}
        self.assertTrue(self._receive(receiver, new_sub))
        self.assertIsNone(entity().subaccounts[1].leverage_tier)
        self.assertIsNone(self._account_tier(new_sub["synthetic_hotkey"]))

        # existing subaccount, checkpoint
        checkpoint = {self.BROADCAST_ENTITY_HOTKEY: entity().model_dump()}
        checkpoint[self.BROADCAST_ENTITY_HOTKEY]["subaccounts"][0]["leverage_tier"] = 0
        stats = receiver.sync_entity_data(checkpoint)
        self.assertEqual(stats['subaccounts_updated'], 0)
        self.assertEqual(stats['leverage_tiers_pushed'], 0)
        self.assertIsNone(entity().subaccounts[0].leverage_tier)
        self.assertIsNone(self._account_tier(synthetic))

    # ==================== repairing a MinerAccount that lost the field ====================

    def test_same_tier_call_repushes_and_rebroadcasts(self):
        receiver, synthetic, _ = self._receiver_with_legacy_subaccount(0)
        self.assertTrue(receiver.update_subaccount_leverage_tier(self.BROADCAST_ENTITY_HOTKEY, synthetic, 2)[0])
        self.miner_account_client.set_leverage_tier(synthetic, None)

        receiver.running_unit_tests = False  # only gates the broadcast in this method
        receiver.broadcast_subaccount_registration = MagicMock()
        success, msg = receiver.update_subaccount_leverage_tier(self.BROADCAST_ENTITY_HOTKEY, synthetic, 2)
        self.assertTrue(success, msg)
        self.assertIn("already", msg)
        self.assertEqual(self._account_tier(synthetic), 2)
        receiver.broadcast_subaccount_registration.assert_called_once()
        self.assertEqual(receiver.broadcast_subaccount_registration.call_args.args[1].leverage_tier, 2)

    def test_broadcast_with_unchanged_tier_repairs_miner_account(self):
        receiver, synthetic, data = self._receiver_with_legacy_subaccount(0)
        self.assertTrue(self._receive(receiver, {**data, "leverage_tier": 3}))
        self.assertEqual(self._account_tier(synthetic), 3)

        self.miner_account_client.set_leverage_tier(synthetic, None)
        self.assertTrue(self._receive(receiver, {**data, "leverage_tier": 3}))
        self.assertEqual(receiver.get_entity_data(self.BROADCAST_ENTITY_HOTKEY).subaccounts[0].leverage_tier, 3)
        self.assertEqual(self._account_tier(synthetic), 3)

    def test_checkpoint_sync_repairs_miner_account_that_lost_the_tier(self):
        # An account-size sync from a validator without the field replaces the MinerAccount; the entity
        # sync that follows puts the stored tier back although nothing changed on the entity side
        self.assertTrue(self._update(3)[0])
        self.miner_account_client.set_leverage_tier(self.synthetic, None)
        self.assertIsNone(self._account_tier())

        checkpoint = {self.ENTITY_HOTKEY: self.entity_client.get_entity_data(self.ENTITY_HOTKEY)}
        stats = self.entity_client.sync_entity_data(checkpoint)
        self.assertEqual(stats['subaccounts_updated'], 0)
        self.assertEqual(stats['leverage_tiers_pushed'], 1)
        self.assertEqual(self._stored_tier(), 3)
        self.assertEqual(self._account_tier(), 3)
        # Nothing left to repair on the next pass
        checkpoint = {self.ENTITY_HOTKEY: self.entity_client.get_entity_data(self.ENTITY_HOTKEY)}
        self.assertEqual(self.entity_client.sync_entity_data(checkpoint)['leverage_tiers_pushed'], 0)


class TestStandardTierOrderPath(TestBase):
    """Orders for a standard subaccount go through MarketOrderManager and are clamped by the standard
    tier caps; raising the tier opens more room and lowering needs a flat book."""

    orchestrator = None
    entity_client = None
    market_order_client = None
    position_client = None
    metagraph_client = None

    ENTITY_HOTKEY = "entity_tierorder"
    ACCOUNT_SIZE = 100_000.0
    PRICE = 50_000.0

    @classmethod
    def setUpClass(cls):
        cls.orchestrator = ServerOrchestrator.get_instance()
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(mode=ServerMode.TESTING, secrets=secrets)
        cls.entity_client = cls.orchestrator.get_client('entity')
        cls.market_order_client = cls.orchestrator.get_client('market_order')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')

    def setUp(self):
        self.orchestrator.clear_all_test_data()
        self.metagraph_client.set_hotkeys([self.ENTITY_HOTKEY])
        self.assertTrue(self.entity_client.register_entity(entity_hotkey=self.ENTITY_HOTKEY)[0])
        success, info, msg = self.entity_client.create_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY, account_size=self.ACCOUNT_SIZE, asset_class="crypto",
        )
        self.assertTrue(success, msg)
        self.synthetic = info['synthetic_hotkey']
        self.now_ms = TimeUtil.now_in_millis()

    def tearDown(self):
        self.orchestrator.clear_all_test_data()

    def _order(self, trade_pair, order_type, order_size):
        """Execute a market order at PRICE; each call moves the clock past the order cooldown."""
        self.now_ms += ValiConfig.ORDER_COOLDOWN_MS + 1_000
        price_source = PriceSource(
            source='test', timespan_ms=0, open=self.PRICE, close=self.PRICE, vwap=None, high=self.PRICE,
            low=self.PRICE, start_ms=self.now_ms, websocket=True, lag_ms=100, bid=self.PRICE - 1, ask=self.PRICE + 1,
        )
        self.market_order_client.execute_order(
            self.synthetic, f"order-{trade_pair.trade_pair_id}-{self.now_ms}", trade_pair, ExecutionType.MARKET,
            order_type, order_size, fill_price=self.PRICE, price_sources=[price_source], slippage=0.0,
            now_ms=self.now_ms,
        )

    def _buy(self, trade_pair, value) -> float:
        """Market buy `value` USD; returns the open position's net value in USD."""
        self._order(trade_pair, OrderType.LONG, OrderSize(value=value))
        position = self.position_client.get_open_position_for_trade_pair(self.synthetic, trade_pair.trade_pair_id)
        return abs(position.net_value)

    def _update(self, tier):
        return self.entity_client.update_subaccount_leverage_tier(self.ENTITY_HOTKEY, self.synthetic, tier)

    def assertClampedTo(self, net_value, cap_multiple):
        # The clamp lands just under cap x balance: fee headroom, quantity rounding, fees already paid
        cap = cap_multiple * self.ACCOUNT_SIZE
        self.assertLess(net_value, cap)
        self.assertGreater(net_value, cap * 0.99)

    def test_tier_one_per_pair_caps_then_class_cap(self):
        # Other coins cap at 0.5x, majors at 1.5x, the crypto class at 1.5x in total (all requested at 3x)
        self.assertClampedTo(self._buy(TradePair.LINKUSDC, 3 * self.ACCOUNT_SIZE), 0.5)
        self.assertClampedTo(self._buy(TradePair.BTCUSDC, 3 * self.ACCOUNT_SIZE), 1.0)
        with self.assertRaises(SignalException) as ctx:
            self._buy(TradePair.LINKUSDC, 3 * self.ACCOUNT_SIZE)
        self.assertIn("No buying power remaining", str(ctx.exception))

    def test_raising_tier_opens_room_and_lowering_needs_a_flat_book(self):
        self.assertClampedTo(self._buy(TradePair.BTCUSDC, 3 * self.ACCOUNT_SIZE), 1.5)

        success, msg = self._update(3)
        self.assertTrue(success, msg)
        self.assertClampedTo(self._buy(TradePair.BTCUSDC, 3 * self.ACCOUNT_SIZE), 2.5)

        success, msg = self._update(1)
        self.assertFalse(success)
        self.assertIn("open position", msg)

        self._order(TradePair.BTCUSDC, OrderType.FLAT, OrderSize(quantity=0.0))
        self.assertIsNone(self.position_client.get_open_position_for_trade_pair(self.synthetic, TradePair.BTCUSDC.trade_pair_id))
        success, msg = self._update(1)
        self.assertTrue(success, msg)


LEVERAGE_TIER_SIGNED_FIELDS = ("entity_coldkey", "entity_hotkey", "synthetic_hotkey", "leverage_tier", "nonce", "timestamp")


def _validator_leverage_tier_client():
    """Flask test client for the validator endpoint with a mocked entity client and ownership check."""
    from vanta_api.nonce_manager import NonceManager
    from vanta_api.validator_rest_server import ValidatorRestServer

    server = object.__new__(ValidatorRestServer)
    server._entity_client = MagicMock()
    server._entity_client.update_subaccount_leverage_tier.return_value = (True, "updated")
    server._verify_coldkey_owns_hotkey = MagicMock(return_value=True)
    server.nonce_manager = NonceManager()
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.route("/entity/subaccount/leverage-tier", methods=["POST"])(server.update_subaccount_leverage_tier)
    return server, app.test_client()


class TestValidatorLeverageTierEndpoint(unittest.TestCase):
    """POST /entity/subaccount/leverage-tier on the validator, with a mocked entity client."""

    def setUp(self):
        self.coldkey = Keypair.create_from_uri("//Alice")
        self.hotkey = Keypair.create_from_uri("//Bob")
        self.server, self.client = _validator_leverage_tier_client()

    def _body(self, signed=None, **tampered):
        """A correctly signed request. `signed` overrides fields before signing (what a gateway would
        have sent), `tampered` overrides after signing (what a man in the middle would change)."""
        fields = {
            "entity_coldkey": self.coldkey.ss58_address,
            "entity_hotkey": self.hotkey.ss58_address,
            "synthetic_hotkey": f"{self.hotkey.ss58_address}_0",
            "leverage_tier": 2,
            "nonce": uuid.uuid4().hex,
            "timestamp": TimeUtil.now_in_millis(),
        }
        fields.update(signed or {})
        message = json.dumps(fields, sort_keys=True).encode("utf-8")
        body = {**fields, "signature": self.coldkey.sign(message).hex(), "version": "2.2.1"}
        body.update(tampered)
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
                status, _ = self._post(self._body(signed={"leverage_tier": bad}))
                self.assertEqual(status, 400)
        for field in ("synthetic_hotkey", "nonce", "timestamp"):
            with self.subTest(missing=field):
                body = self._body()
                del body[field]
                status, data = self._post(body)
                self.assertEqual(status, 400)
                self.assertIn(field, data['error'])
        for bad in ({"nonce": ""}, {"nonce": 123}, {"timestamp": "now"}, {"timestamp": 1.5}, {"timestamp": True}):
            with self.subTest(signed=bad):
                status, _ = self._post(self._body(signed=bad))
                self.assertEqual(status, 400)
        self.server._entity_client.update_subaccount_leverage_tier.assert_not_called()

    # ==================== replay protection ====================

    def test_replayed_request_is_rejected(self):
        body = self._body()
        self.assertEqual(self._post(body)[0], 200)
        status, data = self._post(body)
        self.assertEqual(status, 401)
        self.assertIn("Nonce already used", data['error'])
        self.server._entity_client.update_subaccount_leverage_tier.assert_called_once()

    def test_signature_is_bound_to_every_signed_field(self):
        for field, value in (
            ("leverage_tier", 3),
            ("synthetic_hotkey", f"{self.hotkey.ss58_address}_1"),
            ("nonce", uuid.uuid4().hex),
            ("timestamp", TimeUtil.now_in_millis() + 30_000),
        ):
            with self.subTest(tampered=field):
                body = self._body()
                self.assertNotEqual(body[field], value)
                body[field] = value
                status, _ = self._post(body)
                self.assertEqual(status, 401)
        self.server._entity_client.update_subaccount_leverage_tier.assert_not_called()

    def test_expired_or_future_timestamp_is_rejected(self):
        now = TimeUtil.now_in_millis()
        for timestamp in (now - 6 * 60 * 1000, now + 2 * 60 * 1000):
            with self.subTest(timestamp=timestamp):
                status, _ = self._post(self._body(signed={"timestamp": timestamp}))
                self.assertEqual(status, 401)
        self.server._entity_client.update_subaccount_leverage_tier.assert_not_called()

    def test_nonce_is_consumed_only_after_signature_and_ownership_pass(self):
        body = self._body()
        other = Keypair.create_from_uri("//Charlie")
        self.assertEqual(self._post({**body, "signature": other.sign(b"anything").hex()})[0], 401)
        self.server._verify_coldkey_owns_hotkey.return_value = False
        self.assertEqual(self._post(body)[0], 403)
        self.server._verify_coldkey_owns_hotkey.return_value = True
        self.assertEqual(self._post(body)[0], 200)


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

    def _forward(self, leverage_tier=3):
        """Post to the gateway with the validator call patched; returns (status, data, payload sent)."""
        validator_resp = MagicMock(status_code=200)
        validator_resp.json.return_value = {"status": "success", "leverage_tier": leverage_tier}
        with patch("requests.post", return_value=validator_resp) as post:
            status, data = self._post({"synthetic_hotkey": f"{self.hotkey.ss58_address}_0", "leverage_tier": leverage_tier})
        self.assertEqual(post.call_args.args[0], "http://validator.test/entity/subaccount/leverage-tier")
        return status, data, post.call_args.kwargs['json']

    def test_forwards_signed_payload_to_validator(self):
        status, data, payload = self._forward()
        self.assertEqual(status, 200)
        self.assertEqual(data['leverage_tier'], 3)
        self.assertEqual(payload['entity_coldkey'], self.coldkey.ss58_address)
        self.assertEqual(payload['entity_hotkey'], self.hotkey.ss58_address)
        self.assertEqual(payload['synthetic_hotkey'], f"{self.hotkey.ss58_address}_0")
        self.assertEqual(payload['leverage_tier'], 3)
        self.assertIsInstance(payload['nonce'], str)
        self.assertTrue(payload['nonce'])
        self.assertIsInstance(payload['timestamp'], int)
        self.assertLess(abs(payload['timestamp'] - TimeUtil.now_in_millis()), 60_000)
        signed = json.dumps({k: payload[k] for k in LEVERAGE_TIER_SIGNED_FIELDS}, sort_keys=True).encode("utf-8")
        self.assertTrue(Keypair(ss58_address=self.coldkey.ss58_address).verify(signed, bytes.fromhex(payload['signature'])))

    def test_each_request_gets_a_fresh_nonce(self):
        first = self._forward()[2]
        second = self._forward()[2]
        self.assertNotEqual(first['nonce'], second['nonce'])
        self.assertNotEqual(first['signature'], second['signature'])

    def test_gateway_payload_is_accepted_by_the_validator_endpoint(self):
        payload = self._forward()[2]
        server, client = _validator_leverage_tier_client()
        resp = client.post("/entity/subaccount/leverage-tier", json=payload)
        self.assertEqual(resp.status_code, 200, resp.data)
        server._entity_client.update_subaccount_leverage_tier.assert_called_once_with(
            self.hotkey.ss58_address, f"{self.hotkey.ss58_address}_0", 3
        )
        # The same bytes a second time are a replay
        self.assertEqual(client.post("/entity/subaccount/leverage-tier", json=payload).status_code, 401)

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


class TestTradePairsEndpointStandardTiers(unittest.TestCase):
    """GET /trade-pairs exposes the standard tier values next to the legacy ones."""

    def setUp(self):
        from vanta_api.validator_rest_server import ValidatorRestServer

        server = object.__new__(ValidatorRestServer)
        app = Flask(__name__)
        app.config['TESTING'] = True
        app.route("/trade-pairs", methods=["GET"])(server.get_allowed_trade_pairs)
        self.client = app.test_client()

    def test_per_pair_and_table_values(self):
        resp = self.client.get("/trade-pairs")
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.data)
        by_id = {entry['trade_pair_id']: entry for entry in data['allowed'] + data['disabled']}

        btc = by_id['BTCUSDC']
        self.assertEqual(btc['standard_positional_leverage_by_tier'], {"1": 1.5, "2": 2.0, "3": 2.5})
        self.assertEqual(set(btc['subaccount_positional_leverage_by_tier']), {"1", "2", "3", "4"})
        self.assertEqual(by_id['EURNZD']['standard_positional_leverage_by_tier'], {"1": 5.0, "2": 7.5, "3": 10.0})
        self.assertEqual(by_id['NVDA']['standard_positional_leverage_by_tier'], {"1": 0.5, "2": 1.0, "3": 1.5})

        tiers = data['standard_leverage_tiers']
        self.assertEqual(tiers['class']['1']['crypto'], 1.5)
        self.assertEqual(tiers['class']['3']['equities'], 3.0)
        self.assertEqual(tiers['portfolio']['3']['all_markets'], 25.0)
        self.assertNotIn('hl_all', tiers['portfolio']['1'])


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
