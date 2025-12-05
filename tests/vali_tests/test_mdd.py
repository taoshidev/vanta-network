# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
MDD (Maximum Drawdown) Checker tests using modern RPC infrastructure.

Tests MDDCheckerServer functionality with proper server/client setup.
"""
from copy import deepcopy

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position

from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.price_source import PriceSource


class TestMDDChecker(TestBase):
    """
    Integration tests for MDD Checker using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    position_client = None
    elimination_client = None
    metagraph_client = None
    mdd_checker_client = None

    MINER_HOTKEY = "test_miner"

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator."""
        # Get the singleton orchestrator and start all required servers
        cls.orchestrator = ServerOrchestrator.get_instance()

        # Start all servers in TESTING mode (idempotent - safe if already started by another test class)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        # Get clients from orchestrator (servers guaranteed ready, no connection delays)
        cls.live_price_fetcher_client = cls.orchestrator.get_client('live_price_fetcher')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.mdd_checker_client = cls.orchestrator.get_client('mdd_checker')

        # Initialize metagraph with test hotkey
        cls.metagraph_client.set_hotkeys([cls.MINER_HOTKEY])

    @classmethod
    def tearDownClass(cls):
        """
        One-time teardown: No action needed.

        Note: Servers and clients are managed by ServerOrchestrator singleton and shared
        across all test classes. They will be shut down automatically at process exit.
        """
        pass

    def setUp(self):
        """Per-test setup: Reset data state (fast - no server restarts)."""
        # NOTE: Skip super().setUp() to avoid killing ports (servers already running)

        self.MINER_HOTKEY = "test_miner"

        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Re-initialize metagraph with test hotkey (cleared by clear_all_test_data())
        self.metagraph_client.set_hotkeys([self.MINER_HOTKEY])

        # Create fresh test data
        self._create_test_data()

        # Reset MDD checker state via client
        self.mdd_checker_client.reset_debug_counters()
        self.mdd_checker_client.price_correction_enabled = False  # Disabled by default, enable per test

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_data(self):
        """Helper to create fresh test data for each test."""
        self.DEFAULT_TEST_POSITION_UUID = "test_position"
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis()
        self.DEFAULT_ACCOUNT_SIZE = 100_000
        self.trade_pair_to_default_position = {x: Position(
            miner_hotkey=self.MINER_HOTKEY,
            position_uuid=self.DEFAULT_TEST_POSITION_UUID + str(x.trade_pair_id),
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=x,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        ) for x in TradePair}

    def create_price_source(self, price, bid=None, ask=None, order_time_ms=None):
        """Create a price source for test data injection."""
        if bid is None:
            bid = price
        if ask is None:
            ask = price
        if order_time_ms is None:
            order_time_ms = TimeUtil.now_in_millis()

        return PriceSource(
            source='test',
            timespan_ms=0,
            open=price,
            close=price,
            vwap=None,
            high=price,
            low=price,
            start_ms=order_time_ms,  # Match order time for price correction
            websocket=True,
            lag_ms=100,
            bid=bid,
            ask=ask
        )

    def verify_elimination_data_in_memory_and_disk(self, expected_eliminations):
        """Verify elimination data matches expectations."""
        expected_eliminated_hotkeys = [x['hotkey'] for x in expected_eliminations]
        eliminated_hotkeys = [x['hotkey'] for x in self.elimination_client.get_eliminations_from_memory()]

        self.assertEqual(
            len(eliminated_hotkeys),
            len(expected_eliminated_hotkeys),
            f"Eliminated hotkeys in memory/disk do not match expected. "
            f"eliminated_hotkeys: {eliminated_hotkeys} "
            f"expected_eliminated_hotkeys: {expected_eliminated_hotkeys}"
        )
        self.assertEqual(set(eliminated_hotkeys), set(expected_eliminated_hotkeys))

        for v1, v2 in zip(expected_eliminations, self.elimination_client.get_eliminations_from_memory()):
            self.assertEqual(v1['hotkey'], v2['hotkey'])
            self.assertEqual(v1['reason'], v2['reason'])
            self.assertAlmostEqual(
                v1['elimination_initiated_time_ms'] / 1000.0,
                v2['elimination_initiated_time_ms'] / 1000.0,
                places=1
            )
            self.assertAlmostEqual(v1['dd'], v2['dd'], places=2)

    def verify_positions_on_disk(self, in_memory_positions, assert_all_closed=None, assert_all_open=None,
                                 verify_positions_same=True, assert_price_changes=False):
        """Verify positions on disk match in-memory positions."""
        positions_from_disk = self.position_client.get_positions_for_one_hotkey(self.MINER_HOTKEY)
        self.assertEqual(
            len(positions_from_disk),
            len(in_memory_positions),
            f"Mismatched number of positions. Positions on disk: {positions_from_disk} "
            f"Positions in memory: {in_memory_positions}"
        )

        for position in in_memory_positions:
            matching_disk_position = next(
                (x for x in positions_from_disk if x.position_uuid == position.position_uuid),
                None
            )
            self.assertIsNotNone(matching_disk_position)
            # Use static method for comparison
            if verify_positions_same:
                success, reason = PositionManagerClient.positions_are_the_same(position, matching_disk_position)
                self.assertTrue(success, reason)

            if assert_price_changes:
                self.assertNotEqual(
                    position.orders[-1].price,
                    matching_disk_position.orders[-1].price,
                    f"Expected price change for position {position.position_uuid} "
                    f"but found same price on disk."
                )
                self.assertNotEqual(position.average_entry_price, matching_disk_position.average_entry_price)

            if assert_all_closed:
                self.assertTrue(
                    matching_disk_position.is_closed_position,
                    f"Position in memory: {position} Position on disk: {matching_disk_position}"
                )
            if assert_all_open:
                self.assertFalse(matching_disk_position.is_closed_position)

    def verify_core_position_fields_unchanged(self, position_before, position_after, allow_price_correction=False):
        """
        Rigorously verify that core position fields remain unchanged after mdd_check().

        mdd_check() always recalculates fees/returns, so this method skips validating those fields.
        This method focuses on verifying structural integrity: orders, leverages, quantities, etc.

        Args:
            position_before: Position before mdd_check
            position_after: Position after mdd_check (from disk)
            allow_price_correction: If True, allows order prices and average_entry_price to differ (when price correction is enabled)
        """
        # Structural fields that must NEVER change
        self.assertEqual(position_before.miner_hotkey, position_after.miner_hotkey, "miner_hotkey changed")
        self.assertEqual(position_before.position_uuid, position_after.position_uuid, "position_uuid changed")
        self.assertEqual(position_before.open_ms, position_after.open_ms, "open_ms changed")
        self.assertEqual(position_before.trade_pair, position_after.trade_pair, "trade_pair changed")
        self.assertEqual(position_before.position_type, position_after.position_type, "position_type changed")
        self.assertEqual(position_before.is_closed_position, position_after.is_closed_position, "is_closed_position changed")

        # Order list must be identical in structure
        self.assertEqual(len(position_before.orders), len(position_after.orders), "Number of orders changed")
        for i, (order_before, order_after) in enumerate(zip(position_before.orders, position_after.orders)):
            self.assertEqual(order_before.order_uuid, order_after.order_uuid, f"Order {i} uuid changed")
            self.assertEqual(order_before.order_type, order_after.order_type, f"Order {i} type changed")
            self.assertAlmostEqual(order_before.leverage, order_after.leverage, places=9, msg=f"Order {i} leverage changed")
            if not allow_price_correction:
                self.assertAlmostEqual(order_before.price, order_after.price, places=9, msg=f"Order {i} price changed")
            self.assertEqual(order_before.processed_ms, order_after.processed_ms, f"Order {i} processed_ms changed")

        # Position state fields that must remain unchanged (except average_entry_price and net_value if prices corrected)
        self.assertAlmostEqual(position_before.net_leverage, position_after.net_leverage, places=9, msg="net_leverage changed")
        self.assertAlmostEqual(position_before.net_quantity, position_after.net_quantity, places=9, msg="net_quantity changed")
        if not allow_price_correction:
            self.assertAlmostEqual(position_before.net_value, position_after.net_value, places=6, msg="net_value changed")
            self.assertAlmostEqual(position_before.average_entry_price, position_after.average_entry_price, places=9, msg="average_entry_price changed")
        self.assertAlmostEqual(position_before.cumulative_entry_value, position_after.cumulative_entry_value, places=6, msg="cumulative_entry_value changed")

        # NOTE: We intentionally skip validating current_return and return_at_close because mdd_check() always updates them.
        # If you need to test that fees don't change, write a test that doesn't call mdd_check().

    def add_order_to_position_and_save_to_disk(self, position, order):
        """Add order to position and save to disk."""
        position.add_order(order, self.live_price_fetcher_client)
        self.position_client.save_miner_position(position)

    def test_get_live_prices(self):
        """Test that live price fetching works with injected test data."""
        # Inject test price data
        test_price = 65000.0
        price_source = self.create_price_source(test_price)
        self.live_price_fetcher_client.set_test_price_source(TradePair.BTCUSD, price_source)

        live_price, price_sources = self.live_price_fetcher_client.get_latest_price(
            trade_pair=TradePair.BTCUSD,
            time_ms=TimeUtil.now_in_millis()
        )
        self.assertTrue(live_price > 0)
        self.assertTrue(price_sources)
        self.assertTrue(all([x.close > 0 for x in price_sources]))

    def test_mdd_price_correction(self):
        """
        Comprehensive test that price correction works for all order sources, timestamps, and trade pairs.

        Tests:
        1. Different OrderSource values (ORGANIC, LIMIT_FILLED, BRACKET_FILLED, PRICE_FILLED_ELIMINATION_FLAT)
        2. Recent orders (within 5 minute window) - should be corrected
        3. Old orders (beyond 5 minute window) - should NOT be corrected
        4. Different trade pairs (BTCUSD crypto, ETHUSD crypto, EURUSD forex)
        5. Multiple orders in same position with different characteristics
        """
        from vali_objects.enums.order_source_enum import OrderSource
        from vali_objects.vali_config import ValiConfig

        self.mdd_checker_client.price_correction_enabled = True
        self.verify_elimination_data_in_memory_and_disk([])

        now_ms = TimeUtil.now_in_millis()

        # Define time windows for testing
        recent_time_ms = now_ms - 60000  # 1 minute ago (within 5 minute window)
        old_time_ms = now_ms - ValiConfig.RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS - 10000  # Beyond 5 minute window

        # ==================== Test 1: BTCUSD Position with Multiple OrderSource Types ====================
        btc_position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        correct_btc_price = 65000.0
        wrong_btc_price = 1000.0

        # Order 1: OLD order (beyond 5 minute window) - should NOT be corrected
        # Use LONG to start the position
        btc_order_old = Order(
            order_type=OrderType.LONG,
            leverage=0.1,
            price=wrong_btc_price,
            trade_pair=TradePair.BTCUSD,
            processed_ms=old_time_ms,
            order_uuid="btc_old",
            src=OrderSource.ORGANIC
        )

        # Order 2: ORGANIC order (recent) - should be corrected
        btc_order_organic = Order(
            order_type=OrderType.LONG,
            leverage=0.05,
            price=wrong_btc_price,
            trade_pair=TradePair.BTCUSD,
            processed_ms=recent_time_ms,
            order_uuid="btc_organic",
            src=OrderSource.ORGANIC
        )

        # Order 3: LIMIT_FILLED order (recent) - should be corrected
        btc_order_limit = Order(
            order_type=OrderType.LONG,
            leverage=0.05,
            price=wrong_btc_price,
            trade_pair=TradePair.BTCUSD,
            processed_ms=recent_time_ms + 1000,
            order_uuid="btc_limit",
            src=OrderSource.LIMIT_FILLED
        )

        # Order 4: BRACKET_FILLED order (recent) - should be corrected
        btc_order_bracket = Order(
            order_type=OrderType.LONG,
            leverage=0.05,
            price=wrong_btc_price,
            trade_pair=TradePair.BTCUSD,
            processed_ms=recent_time_ms + 2000,
            order_uuid="btc_bracket",
            src=OrderSource.BRACKET_FILLED
        )

        # Order 5: PRICE_FILLED_ELIMINATION_FLAT (recent) - should be corrected
        # Add another LONG order instead of FLAT to avoid closing position
        btc_order_elimination = Order(
            order_type=OrderType.LONG,
            leverage=0.05,
            price=wrong_btc_price,
            trade_pair=TradePair.BTCUSD,
            processed_ms=recent_time_ms + 3000,
            order_uuid="btc_elimination",
            src=OrderSource.PRICE_FILLED_ELIMINATION_FLAT
        )

        # Inject correct price sources for BTCUSD
        price_source_btc_recent = self.create_price_source(correct_btc_price, order_time_ms=recent_time_ms)
        price_source_btc_old = self.create_price_source(correct_btc_price, order_time_ms=old_time_ms)
        self.live_price_fetcher_client.set_test_price_source(TradePair.BTCUSD, price_source_btc_recent)

        # Add all BTC orders to position
        self.add_order_to_position_and_save_to_disk(btc_position, btc_order_old)
        self.add_order_to_position_and_save_to_disk(btc_position, btc_order_organic)
        self.add_order_to_position_and_save_to_disk(btc_position, btc_order_limit)
        self.add_order_to_position_and_save_to_disk(btc_position, btc_order_bracket)
        self.add_order_to_position_and_save_to_disk(btc_position, btc_order_elimination)

        # ==================== Test 2: ETHUSD Position with Different Order Sources ====================
        eth_position = self.trade_pair_to_default_position[TradePair.ETHUSD]
        correct_eth_price = 3200.0
        wrong_eth_price = 500.0

        # Recent LIMIT_FILLED order - should be corrected
        eth_order_limit = Order(
            order_type=OrderType.LONG,
            leverage=0.2,
            price=wrong_eth_price,
            trade_pair=TradePair.ETHUSD,
            processed_ms=recent_time_ms,
            order_uuid="eth_limit",
            src=OrderSource.LIMIT_FILLED
        )

        # Recent BRACKET_FILLED order - should be corrected
        eth_order_bracket = Order(
            order_type=OrderType.SHORT,
            leverage=-0.1,
            price=wrong_eth_price,
            trade_pair=TradePair.ETHUSD,
            processed_ms=recent_time_ms + 1000,
            order_uuid="eth_bracket",
            src=OrderSource.BRACKET_FILLED
        )

        # Inject correct price sources for ETHUSD
        price_source_eth = self.create_price_source(correct_eth_price, order_time_ms=recent_time_ms)
        self.live_price_fetcher_client.set_test_price_source(TradePair.ETHUSD, price_source_eth)

        # Add ETH orders to position
        self.add_order_to_position_and_save_to_disk(eth_position, eth_order_limit)
        self.add_order_to_position_and_save_to_disk(eth_position, eth_order_bracket)

        # ==================== Test 3: EURUSD Forex Position ====================
        eur_position = self.trade_pair_to_default_position[TradePair.EURUSD]
        correct_eur_price = 1.1000
        wrong_eur_price = 0.5000

        # Recent ORGANIC order - should be corrected
        eur_order_organic = Order(
            order_type=OrderType.LONG,
            leverage=1.0,
            price=wrong_eur_price,
            trade_pair=TradePair.EURUSD,
            processed_ms=recent_time_ms,
            order_uuid="eur_organic",
            src=OrderSource.ORGANIC
        )

        # Inject correct price sources for EURUSD
        price_source_eur = self.create_price_source(correct_eur_price, order_time_ms=recent_time_ms)
        self.live_price_fetcher_client.set_test_price_source(TradePair.EURUSD, price_source_eur)

        # Add EUR order to position
        self.add_order_to_position_and_save_to_disk(eur_position, eur_order_organic)

        # ==================== Run MDD Check with Price Corrections ====================
        self.mdd_checker_client.last_price_fetch_time_ms = now_ms - 1000 * 30
        self.mdd_checker_client.mdd_check()
        self.verify_elimination_data_in_memory_and_disk([])

        # ==================== Verify Price Corrections ====================

        # Verify BTCUSD position
        btc_from_disk = self.position_client.get_miner_position_by_uuid(self.MINER_HOTKEY, btc_position.position_uuid)
        self.assertIsNotNone(btc_from_disk)
        self.assertEqual(len(btc_from_disk.orders), 5)

        # Verify old order was NOT corrected (beyond 5 minute window)
        old_order = next(o for o in btc_from_disk.orders if o.order_uuid == "btc_old")
        self.assertAlmostEqual(
            old_order.price,
            wrong_btc_price,
            delta=10,
            msg=f"Old order should NOT be corrected. Expected ~{wrong_btc_price}, got {old_order.price}"
        )

        # Verify recent ORGANIC order was corrected
        organic_order = next(o for o in btc_from_disk.orders if o.order_uuid == "btc_organic")
        self.assertAlmostEqual(
            organic_order.price,
            correct_btc_price,
            delta=100,
            msg=f"ORGANIC order should be corrected to ~{correct_btc_price}, got {organic_order.price}"
        )

        # Verify recent LIMIT_FILLED order was corrected
        limit_order = next(o for o in btc_from_disk.orders if o.order_uuid == "btc_limit")
        self.assertAlmostEqual(
            limit_order.price,
            correct_btc_price,
            delta=100,
            msg=f"LIMIT_FILLED order should be corrected to ~{correct_btc_price}, got {limit_order.price}"
        )

        # Verify recent BRACKET_FILLED order was corrected
        bracket_order = next(o for o in btc_from_disk.orders if o.order_uuid == "btc_bracket")
        self.assertAlmostEqual(
            bracket_order.price,
            correct_btc_price,
            delta=100,
            msg=f"BRACKET_FILLED order should be corrected to ~{correct_btc_price}, got {bracket_order.price}"
        )

        # Verify recent PRICE_FILLED_ELIMINATION_FLAT order was corrected
        elim_order = next(o for o in btc_from_disk.orders if o.order_uuid == "btc_elimination")
        self.assertAlmostEqual(
            elim_order.price,
            correct_btc_price,
            delta=100,
            msg=f"PRICE_FILLED_ELIMINATION_FLAT order should be corrected to ~{correct_btc_price}, got {elim_order.price}"
        )

        # Verify ETHUSD position - both orders should be corrected
        eth_from_disk = self.position_client.get_miner_position_by_uuid(self.MINER_HOTKEY, eth_position.position_uuid)
        self.assertIsNotNone(eth_from_disk)
        self.assertEqual(len(eth_from_disk.orders), 2)

        eth_limit = next(o for o in eth_from_disk.orders if o.order_uuid == "eth_limit")
        self.assertAlmostEqual(
            eth_limit.price,
            correct_eth_price,
            delta=50,
            msg=f"ETH LIMIT_FILLED order should be corrected to ~{correct_eth_price}, got {eth_limit.price}"
        )

        eth_bracket = next(o for o in eth_from_disk.orders if o.order_uuid == "eth_bracket")
        self.assertAlmostEqual(
            eth_bracket.price,
            correct_eth_price,
            delta=50,
            msg=f"ETH BRACKET_FILLED order should be corrected to ~{correct_eth_price}, got {eth_bracket.price}"
        )

        # Verify EURUSD forex position - order should be corrected
        eur_from_disk = self.position_client.get_miner_position_by_uuid(self.MINER_HOTKEY, eur_position.position_uuid)
        self.assertIsNotNone(eur_from_disk)
        self.assertEqual(len(eur_from_disk.orders), 1)

        eur_organic = eur_from_disk.orders[0]
        self.assertAlmostEqual(
            eur_organic.price,
            correct_eur_price,
            delta=0.01,
            msg=f"EUR ORGANIC order should be corrected to ~{correct_eur_price}, got {eur_organic.price}"
        )

    def test_no_mdd_failures(self):
        self.verify_elimination_data_in_memory_and_disk([])
        self.position = self.trade_pair_to_default_position[TradePair.BTCUSD]

        # Inject test price data
        test_price = 65000.0
        price_source = self.create_price_source(test_price)
        self.live_price_fetcher_client.set_test_price_source(TradePair.BTCUSD, price_source)

        live_price, _ = self.live_price_fetcher_client.get_latest_price(trade_pair=TradePair.BTCUSD)
        o1 = Order(order_type=OrderType.SHORT,
                leverage=1.0,
                price=live_price,
                trade_pair=TradePair.BTCUSD,
                processed_ms=1000,
                order_uuid="1000")

        o2 = Order(order_type=OrderType.LONG,
                leverage=.5,
                price=live_price,
                trade_pair=TradePair.BTCUSD,
                processed_ms=2000,
                order_uuid="2000")

        self.mdd_checker_client.last_price_fetch_time_ms = TimeUtil.now_in_millis()

        relevant_position = self.trade_pair_to_default_position[TradePair.BTCUSD]
        self.mdd_checker_client.mdd_check()
        # Running mdd_check with no positions should not cause any eliminations but it should write an empty list to disk
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(relevant_position, o1)
        # Snapshot position before mdd_check to verify what changes
        position_snapshot = deepcopy(relevant_position)
        self.mdd_checker_client.mdd_check()
        self.assertEqual(relevant_position.is_closed_position, False)
        self.verify_elimination_data_in_memory_and_disk([])
        # Get position from disk and rigorously verify core fields unchanged
        positions_from_disk = self.position_client.get_positions_for_one_hotkey(self.MINER_HOTKEY)
        self.assertEqual(len(positions_from_disk), 1)
        position_from_disk = positions_from_disk[0]
        self.verify_core_position_fields_unchanged(position_snapshot, position_from_disk)
        self.assertFalse(position_from_disk.is_closed_position)

        self.add_order_to_position_and_save_to_disk(relevant_position, o2)
        # Snapshot position before mdd_check
        position_snapshot = deepcopy(relevant_position)
        self.assertEqual(relevant_position.is_closed_position, False)
        self.mdd_checker_client.mdd_check()
        self.verify_elimination_data_in_memory_and_disk([])
        # Get position from disk and rigorously verify core fields unchanged
        positions_from_disk = self.position_client.get_positions_for_one_hotkey(self.MINER_HOTKEY)
        self.assertEqual(len(positions_from_disk), 1)
        position_from_disk = positions_from_disk[0]
        self.verify_core_position_fields_unchanged(position_snapshot, position_from_disk)
        self.assertFalse(position_from_disk.is_closed_position)

    def test_no_mdd_failures_high_leverage_one_order(self):
        """Test that high leverage positions with small losses don't trigger MDD."""
        self.verify_elimination_data_in_memory_and_disk([])
        position_btc = self.trade_pair_to_default_position[TradePair.BTCUSD]

        # Inject test price data for BTC
        btc_price = 65000.0
        btc_price_source = self.create_price_source(btc_price)
        self.live_price_fetcher_client.set_test_price_source(TradePair.BTCUSD, btc_price_source)

        live_btc_price, _ = self.live_price_fetcher_client.get_latest_price(trade_pair=TradePair.BTCUSD)

        o1 = Order(
            order_type=OrderType.LONG,
            leverage=20.0,
            price=live_btc_price * 1.001,  # Down 0.1%
            trade_pair=TradePair.BTCUSD,
            processed_ms=1000,
            order_uuid="1000"
        )

        self.mdd_checker_client.last_price_fetch_time_ms = TimeUtil.now_in_millis()

        # Running mdd_check with no positions should not cause any eliminations
        self.mdd_checker_client.mdd_check()
        self.verify_elimination_data_in_memory_and_disk([])

        self.add_order_to_position_and_save_to_disk(position_btc, o1)
        self.mdd_checker_client.mdd_check()
        self.assertEqual(position_btc.is_closed_position, False)
        self.verify_elimination_data_in_memory_and_disk([])

        # Reload position from disk after mdd_check (prices may have been corrected)
        positions_from_disk = self.position_client.get_positions_for_one_hotkey(self.MINER_HOTKEY)
        self.assertEqual(len(positions_from_disk), 1)
        btc_position_from_disk = positions_from_disk[0]
        self.assertFalse(btc_position_from_disk.is_closed_position)
        self.assertIsNotNone(btc_position_from_disk)

        # Add ETH position
        position_eth = self.trade_pair_to_default_position[TradePair.ETHUSD]

        # Inject test price data for ETH
        eth_price = 3200.0
        eth_price_source = self.create_price_source(eth_price)
        self.live_price_fetcher_client.set_test_price_source(TradePair.ETHUSD, eth_price_source)

        live_eth_price, _ = self.live_price_fetcher_client.get_latest_price(trade_pair=TradePair.ETHUSD)

        o2 = Order(
            order_type=OrderType.LONG,
            leverage=20.0,
            price=live_eth_price * 1.001,  # Down 0.1%
            trade_pair=TradePair.ETHUSD,
            processed_ms=2000,
            order_uuid="2000"
        )

        self.add_order_to_position_and_save_to_disk(position_eth, o2)
        self.mdd_checker_client.mdd_check()
        positions_from_disk = self.position_client.get_positions_for_one_hotkey(self.MINER_HOTKEY)
        self.assertEqual(len(positions_from_disk), 2)

    def test_get_quote_returns_three_values(self):
        """
        Regression test for get_quote return type.

        Tests that get_quote returns exactly 3 values (bid, ask, timestamp)
        and can be properly unpacked. This catches the bug where the type
        annotation was incorrectly set to (float, float, int) instead of
        Tuple[float, float, int], causing RPC serialization errors.
        """
        # Inject test price data with bid/ask
        test_price = 65000.0
        bid_price = 64990.0
        ask_price = 65010.0
        order_time_ms = TimeUtil.now_in_millis()

        price_source = self.create_price_source(
            price=test_price,
            bid=bid_price,
            ask=ask_price,
            order_time_ms=order_time_ms
        )
        self.live_price_fetcher_client.set_test_price_source(TradePair.BTCUSD, price_source)

        # Test that get_quote returns exactly 3 values
        result = self.live_price_fetcher_client.get_quote(TradePair.BTCUSD, order_time_ms)

        # Verify it's a tuple with 3 elements
        self.assertIsInstance(result, tuple, "get_quote should return a tuple")
        self.assertEqual(len(result), 3, "get_quote should return exactly 3 values")

        # Test unpacking works (this is what failed in production)
        bid, ask, timestamp = result

        # Verify the values are correct types (or None)
        self.assertTrue(bid is None or isinstance(bid, (float, int)), "bid should be numeric or None")
        self.assertTrue(ask is None or isinstance(ask, (float, int)), "ask should be numeric or None")
        self.assertTrue(timestamp is None or isinstance(timestamp, (float, int)), "timestamp should be numeric or None")

        # Verify bid/ask relationship if both are present
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            self.assertGreaterEqual(ask, bid, "ask should be >= bid when both are present")


if __name__ == '__main__':
    import unittest
    unittest.main()
