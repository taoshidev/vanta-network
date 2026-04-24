"""
Test equities-specific implementations including:
- Stock splits
- Miner account margins (cash balance, margin loans, interest)
"""
import unittest
from datetime import datetime, timezone, timedelta

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_config import TradePair, ValiConfig, TradePairCategory
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.exceptions.signal_exception import SignalException
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.miner_account.miner_account_manager import MinerAccount, CollateralRecord
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils


class TestEquities(TestBase):
    """
    Test suite for equities-specific functionality.

    Uses ServerOrchestrator singleton pattern for shared server infrastructure.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    miner_account_client = None

    DEFAULT_MINER_HOTKEY = "test_miner"
    DEFAULT_MINER_HOTKEY_2 = "test_miner_2"
    DEFAULT_POSITION_UUID = "test_position"
    # Use timestamp after leverage v3 start (1739937600000) to avoid leverage validation issues
    DEFAULT_OPEN_MS = 1740000000000  # Jan 2025
    DEFAULT_TRADE_PAIR = TradePair.AAPL
    DEFAULT_ACCOUNT_SIZE = 100_000

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator (shared across all test classes)."""
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
        cls.miner_account_client = cls.orchestrator.get_client('miner_account')
        cls.asset_selection_client = cls.orchestrator.get_client('asset_selection')

        # Initialize metagraph with test miners
        cls.metagraph_client.set_hotkeys([cls.DEFAULT_MINER_HOTKEY, cls.DEFAULT_MINER_HOTKEY_2])

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
        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Create fresh test data for this test
        self._create_test_data()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_data(self):
        """Helper to create fresh test data."""
        # Set asset selection to EQUITIES for test miners (required for margin trading)
        self.asset_selection_client.sync_miner_asset_selection_data({
            self.DEFAULT_MINER_HOTKEY: TradePairCategory.EQUITIES.value,
            self.DEFAULT_MINER_HOTKEY_2: TradePairCategory.EQUITIES.value
        })

        # Update the MinerAccount's asset_class field to apply the EQUITIES multiplier
        self.miner_account_client.update_asset_selection(
            self.DEFAULT_MINER_HOTKEY, TradePairCategory.EQUITIES
        )
        self.miner_account_client.update_asset_selection(
            self.DEFAULT_MINER_HOTKEY_2, TradePairCategory.EQUITIES
        )

        # Set account sizes for test miners
        # Use timestamp from yesterday so collateral record is valid today
        yesterday_ms = self.DEFAULT_OPEN_MS - (24 * 60 * 60 * 1000)
        self.miner_account_client.set_miner_account_size(
            self.DEFAULT_MINER_HOTKEY,
            self.DEFAULT_ACCOUNT_SIZE / ValiConfig.COST_PER_THETA,
            timestamp_ms=yesterday_ms
        )
        self.miner_account_client.set_miner_account_size(
            self.DEFAULT_MINER_HOTKEY_2,
            self.DEFAULT_ACCOUNT_SIZE / ValiConfig.COST_PER_THETA,
            timestamp_ms=yesterday_ms
        )

    # Aliases for backward compatibility with test methods
    @property
    def live_price_fetcher(self):
        """Alias for class-level live_price_fetcher_client."""
        return self.live_price_fetcher_client

    @property
    def position_manager(self):
        """Alias for class-level position_client (provides same interface)."""
        return self.position_client

    @property
    def miner_account_manager(self):
        """Alias for class-level miner_account_client."""
        return self.miner_account_client

    # ==================== Stock Split Tests ====================

    def test_stock_split_basic_2_for_1(self):
        """
        Test basic 2-for-1 stock split on a single open position.
        Quantity should double, price should halve, position value should remain the same.
        """
        # Create position with one order
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.AAPL,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        buy_order = Order(
            price=200.0,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="buy_order",
            trade_pair=TradePair.AAPL,
            order_type=OrderType.LONG,
            leverage=1.0,
        )

        position.add_order(buy_order, self.live_price_fetcher)
        self.position_manager.save_miner_position(position)

        # Record pre-split values
        original_quantity = position.orders[0].quantity
        original_price = position.orders[0].price
        original_value = position.orders[0].value

        # Apply 2-for-1 stock split
        split_ratio = 2.0
        self.position_manager.apply_stock_split(TradePair.AAPL.trade_pair_id, split_ratio, "2026-01-23")

        # Reload position and verify
        updated_position = self.position_manager.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            TradePair.AAPL.trade_pair_id
        )

        self.assertIsNotNone(updated_position)
        self.assertEqual(len(updated_position.orders), 1)

        # Quantity should double
        self.assertAlmostEqual(
            updated_position.orders[0].quantity,
            original_quantity * split_ratio,
            places=6,
            msg="Quantity should double after 2-for-1 split"
        )

        # Price should halve
        self.assertAlmostEqual(
            updated_position.orders[0].price,
            original_price / split_ratio,
            places=6,
            msg="Price should halve after 2-for-1 split"
        )

        # Value should remain the same
        self.assertAlmostEqual(
            updated_position.orders[0].value,
            original_value,
            places=2,
            msg="Order value should remain constant after split"
        )

    def test_stock_split_reverse_1_for_10(self):
        """
        Test reverse 1-for-10 stock split (consolidation).
        Quantity should be divided by 10, price should multiply by 10.
        """
        # Create position
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.TSLA,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        buy_order = Order(
            price=10.0,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="buy_order",
            trade_pair=TradePair.TSLA,
            order_type=OrderType.LONG,
            leverage=1.0,
        )

        position.add_order(buy_order, self.live_price_fetcher)
        self.position_manager.save_miner_position(position)

        # Record pre-split values
        original_quantity = position.orders[0].quantity
        original_price = position.orders[0].price

        # Apply 1-for-10 reverse split (ratio = 0.1)
        split_ratio = 0.1
        self.position_manager.apply_stock_split(TradePair.TSLA.trade_pair_id, split_ratio, "2026-01-23")

        # Reload and verify
        updated_position = self.position_manager.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            TradePair.TSLA.trade_pair_id
        )

        self.assertIsNotNone(updated_position)

        # Quantity should be divided by 10
        self.assertAlmostEqual(
            updated_position.orders[0].quantity,
            original_quantity * split_ratio,
            places=6,
            msg="Quantity should be divided by 10 after 1-for-10 reverse split"
        )

        # Price should multiply by 10
        self.assertAlmostEqual(
            updated_position.orders[0].price,
            original_price / split_ratio,
            places=6,
            msg="Price should multiply by 10 after 1-for-10 reverse split"
        )

    def test_stock_split_multiple_orders(self):
        """
        Test stock split on position with multiple orders (buy, partial sell, buy again).
        All orders should be adjusted correctly.
        """
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.NVDA,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        # First buy
        buy_order_1 = Order(
            price=100.0,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="buy_1",
            trade_pair=TradePair.NVDA,
            order_type=OrderType.LONG,
            leverage=1.0,
        )

        # Partial sell
        sell_order = Order(
            price=110.0,
            processed_ms=self.DEFAULT_OPEN_MS + 1000,
            order_uuid="sell_1",
            trade_pair=TradePair.NVDA,
            order_type=OrderType.SHORT,
            leverage=0.5,
        )

        # Second buy
        buy_order_2 = Order(
            price=105.0,
            processed_ms=self.DEFAULT_OPEN_MS + 2000,
            order_uuid="buy_2",
            trade_pair=TradePair.NVDA,
            order_type=OrderType.LONG,
            leverage=0.5,
        )

        position.add_order(buy_order_1, self.live_price_fetcher)
        position.add_order(sell_order, self.live_price_fetcher)
        position.add_order(buy_order_2, self.live_price_fetcher)
        self.position_manager.save_miner_position(position)

        # Record original values
        original_values = [
            (order.quantity, order.price) for order in position.orders
        ]

        # Apply 3-for-1 split
        split_ratio = 3.0
        self.position_manager.apply_stock_split(TradePair.NVDA.trade_pair_id, split_ratio, "2026-01-23")

        # Verify all orders updated
        updated_position = self.position_manager.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            TradePair.NVDA.trade_pair_id
        )

        self.assertEqual(len(updated_position.orders), 3)

        for i, (original_qty, original_price) in enumerate(original_values):
            self.assertAlmostEqual(
                updated_position.orders[i].quantity,
                original_qty * split_ratio,
                places=6,
                msg=f"Order {i} quantity should be multiplied by split ratio"
            )
            self.assertAlmostEqual(
                updated_position.orders[i].price,
                original_price / split_ratio,
                places=6,
                msg=f"Order {i} price should be divided by split ratio"
            )

    def test_stock_split_multiple_miners(self):
        """
        Test stock split affects all miners with open positions in that trade pair.
        """
        # Create positions for two miners
        position_1 = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="pos_1",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.MSFT,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        position_2 = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY_2,
            position_uuid="pos_2",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.MSFT,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        order_1 = Order(
            price=300.0,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="order_1",
            trade_pair=TradePair.MSFT,
            order_type=OrderType.LONG,
            leverage=1.0,
        )

        order_2 = Order(
            price=300.0,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="order_2",
            trade_pair=TradePair.MSFT,
            order_type=OrderType.LONG,
            leverage=1.0,
        )

        position_1.add_order(order_1, self.live_price_fetcher)
        position_2.add_order(order_2, self.live_price_fetcher)

        self.position_manager.save_miner_position(position_1)
        self.position_manager.save_miner_position(position_2)

        # Apply split
        split_ratio = 2.0
        self.position_manager.apply_stock_split(TradePair.MSFT.trade_pair_id, split_ratio, "2026-01-23")

        # Verify both positions updated
        updated_pos_1 = self.position_manager.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            TradePair.MSFT.trade_pair_id
        )
        updated_pos_2 = self.position_manager.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY_2,
            TradePair.MSFT.trade_pair_id
        )

        # Both should be updated
        self.assertAlmostEqual(updated_pos_1.orders[0].price, 300.0 / split_ratio, places=6)
        self.assertAlmostEqual(updated_pos_2.orders[0].price, 300.0 / split_ratio, places=6)

    def test_stock_split_closed_position_unchanged(self):
        """
        Test that closed positions are NOT affected by stock splits.
        Only open positions should be modified.
        """
        # Create a closed position
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.AAPL,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        buy_order = Order(
            price=150.0,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="buy",
            trade_pair=TradePair.AAPL,
            order_type=OrderType.LONG,
            leverage=1.0,
        )

        close_order = Order(
            price=160.0,
            processed_ms=self.DEFAULT_OPEN_MS + 1000,
            order_uuid="close",
            trade_pair=TradePair.AAPL,
            order_type=OrderType.FLAT,
            leverage=0.0,
        )

        position.add_order(buy_order, self.live_price_fetcher)
        position.add_order(close_order, self.live_price_fetcher)
        self.position_manager.save_miner_position(position)

        # Record original values
        original_price = position.orders[0].price

        # Apply split (should not affect closed position)
        split_ratio = 2.0
        self.position_manager.apply_stock_split(TradePair.AAPL.trade_pair_id, split_ratio, "2026-01-23")

        # Verify position unchanged
        positions = self.position_manager.get_positions_for_one_hotkey(
            self.DEFAULT_MINER_HOTKEY
        )

        self.assertEqual(len(positions), 1)
        # Closed position should be unchanged
        self.assertEqual(positions[0].orders[0].price, original_price)

    def test_stock_split_returns_unchanged(self):
        """
        Test that position returns remain the same after a stock split.

        For a LONG position:
        - Entry price: $100, current price: $120 (20% gain)
        - After 2:1 split: Entry price: $50, current price: $60 (still 20% gain)

        The return should be identical before and after the split because:
        - PnL = (current_price - avg_entry) * quantity * lot_size
        - After split: (price/ratio - entry/ratio) * (qty * ratio) = same PnL
        """
        # Create position
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.AAPL,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        entry_price = 100.0
        buy_order = Order(
            price=entry_price,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="buy_order",
            trade_pair=TradePair.AAPL,
            order_type=OrderType.LONG,
            leverage=1.0,
        )

        position.add_order(buy_order, self.live_price_fetcher)
        self.position_manager.save_miner_position(position)

        # Calculate return at a higher price (simulating profit)
        current_price_before_split = 120.0
        return_before_split = position.calculate_pnl(
            current_price_before_split,
            self.live_price_fetcher,
            t_ms=self.DEFAULT_OPEN_MS + 1000
        )

        # Sanity check: should have positive return
        self.assertGreater(return_before_split, 1.0, "Position should be profitable")

        # Apply 2-for-1 stock split
        split_ratio = 2.0
        self.position_manager.apply_stock_split(TradePair.AAPL.trade_pair_id, split_ratio, "2026-01-23")

        # Reload position
        updated_position = self.position_manager.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            TradePair.AAPL.trade_pair_id
        )

        # Calculate return at split-adjusted current price
        # After split, the market price would also be adjusted
        current_price_after_split = current_price_before_split / split_ratio
        return_after_split = updated_position.calculate_pnl(
            current_price_after_split,
            self.live_price_fetcher,
            t_ms=self.DEFAULT_OPEN_MS + 1000
        )

        # Returns should be identical
        self.assertAlmostEqual(
            return_after_split,
            return_before_split,
            places=6,
            msg="Position return should remain unchanged after stock split"
        )

    def test_stock_split_returns_unchanged_short_position(self):
        """
        Test that SHORT position returns remain unchanged after a stock split.

        For a SHORT position:
        - Entry price: $100, current price: $80 (20% profit on short)
        - After 2:1 split: Entry price: $50, current price: $40 (still same return)
        """
        # Create SHORT position
        position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid=self.DEFAULT_POSITION_UUID,
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=TradePair.TSLA,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )

        entry_price = 100.0
        short_order = Order(
            price=entry_price,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="short_order",
            trade_pair=TradePair.TSLA,
            order_type=OrderType.SHORT,
            leverage=-1.0,
        )

        position.add_order(short_order, self.live_price_fetcher)
        self.position_manager.save_miner_position(position)

        # Calculate return at a lower price (profit for short)
        current_price_before_split = 80.0
        return_before_split = position.calculate_pnl(
            current_price_before_split,
            self.live_price_fetcher,
            t_ms=self.DEFAULT_OPEN_MS + 1000
        )

        # Sanity check: should have positive return (price dropped, short profits)
        self.assertGreater(return_before_split, 1.0, "Short position should be profitable when price drops")

        # Apply 4-for-1 stock split
        split_ratio = 4.0
        self.position_manager.apply_stock_split(TradePair.TSLA.trade_pair_id, split_ratio, "2026-01-23")

        # Reload position
        updated_position = self.position_manager.get_open_position_for_trade_pair(
            self.DEFAULT_MINER_HOTKEY,
            TradePair.TSLA.trade_pair_id
        )

        # Calculate return at split-adjusted price
        current_price_after_split = current_price_before_split / split_ratio
        return_after_split = updated_position.calculate_pnl(
            current_price_after_split,
            self.live_price_fetcher,
            t_ms=self.DEFAULT_OPEN_MS + 1000
        )

        # Returns should be identical
        self.assertAlmostEqual(
            return_after_split,
            return_before_split,
            places=6,
            msg="Short position return should remain unchanged after stock split"
        )

    # ==================== Miner Account Margin Tests ====================

    def test_buy_within_buying_power(self):
        """
        Test purchasing equities within buying power and available cash.
        When order <= available_cash, no borrowing occurs.

        Note: Initial buying_power = account_size * 1.5 = $150,000 (Tier 2 EQUITIES multiplier of 1.5).
        Available cash = balance = $100,000.
        """
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        initial_bp = account['buying_power']

        # Verify we start with account_size * multiplier buying power
        self.assertEqual(initial_bp, self.DEFAULT_ACCOUNT_SIZE * 1.5)  # $150K

        # Purchase for $50,000 (within available cash of $100K, no borrowing needed)
        order_value = 50_000.0
        borrowed = 0.0  # within available cash, no borrowing
        self.miner_account_manager.process_order_buy(self.DEFAULT_MINER_HOTKEY, order_value, borrowed)

        # No borrowing when order <= available_cash
        self.assertAlmostEqual(borrowed, 0.0, places=2)

        # bp = (balance - cash_used) * mult = ($100K - $50K) * 1.5 = $75K
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], 75_000.0, places=2)  # $75K
        self.assertAlmostEqual(account['capital_used'], order_value, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], 0.0, places=2)

    def test_buy_large_position(self):
        """
        Test purchasing equities for a large position that exceeds available cash.
        When order > available_cash, equities borrow half the order value.

        Initial buying_power = $100K * 1.5 = $150K, available_cash = $100K.
        """
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        initial_bp = account['buying_power']
        self.assertEqual(initial_bp, self.DEFAULT_ACCOUNT_SIZE * 1.5)  # $150K

        # Purchase for $150,000 (within buying power of $150K, but exceeds available cash)
        order_value = 150_000.0
        expected_borrowed = order_value * 0.5  # $75,000

        borrowed = expected_borrowed  # order > available_cash → borrow half
        self.miner_account_manager.process_order_buy(self.DEFAULT_MINER_HOTKEY, order_value, borrowed)

        # Should borrow half the order value
        self.assertAlmostEqual(borrowed, expected_borrowed, places=2)

        # bp = (balance - cash_used) * mult = ($100K - $75K) * 1.5 = $37.5K
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], 37_500.0, places=2)  # $37.5K
        self.assertAlmostEqual(account['capital_used'], order_value, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], expected_borrowed, places=2)

    def test_insufficient_buying_power_raises_exception(self):
        """
        Test that buying with insufficient buying power raises SignalException.

        Initial buying_power = $100K * 1.5 = $150K (Tier 2).
        """
        # Try to purchase $250,000 worth (buying power is only $150K)
        order_value = 250_000.0

        with self.assertRaises(SignalException) as context:
            self.miner_account_manager.process_order_buy(
                self.DEFAULT_MINER_HOTKEY, order_value, order_value * 0.5
            )

        self.assertIn("Insufficient buying power", str(context.exception))

    def test_sell_repays_loan_and_compounds_pnl(self):
        """
        Test selling equities repays margin loan and compounds realized PNL to balance.
        """
        # Buy $150K: capital_used=$150K, borrowed=$75K (half), bp=($100K-$75K)*1.5=$37.5K
        order_value = 150_000.0
        expected_borrowed = order_value * 0.5  # $75K
        borrowed = expected_borrowed  # order > available_cash → borrow half
        self.miner_account_manager.process_order_buy(self.DEFAULT_MINER_HOTKEY, order_value, borrowed)
        self.assertAlmostEqual(borrowed, expected_borrowed, places=2)

        # Sell with $10K profit: entry=$150K, pnl=$10K
        entry_value = 150_000.0
        realized_pnl = 10_000.0
        self.miner_account_manager.process_order_sell(
            self.DEFAULT_MINER_HOTKEY,
            entry_value,
            realized_pnl,
            borrowed  # full loan repaid (exit_value=$160K > loan=$75K)
        )

        # Balance = $100K + $10K = $110K, buying_power = $110K * 1.5 = $165K
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['balance'], 110_000.0, places=2)
        self.assertAlmostEqual(account['buying_power'], 165_000.0, places=2)
        self.assertAlmostEqual(account['capital_used'], 0.0, places=2)
        self.assertEqual(account['total_borrowed_amount'], 0.0)

    def test_equities_borrows_half_when_exceeds_cash(self):
        """
        Test that equities purchases borrow half the order value only when order exceeds available cash.
        Small purchases within available cash don't use margin.
        """
        # Buy $50K - within available cash ($100K), no borrowing
        order_value = 50_000.0
        borrowed = 0.0  # within available cash, no borrowing
        self.miner_account_manager.process_order_buy(self.DEFAULT_MINER_HOTKEY, order_value, borrowed)

        self.assertAlmostEqual(borrowed, 0.0, places=2)
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], 75_000.0, places=2)  # ($100K-$50K)*1.5
        self.assertAlmostEqual(account['total_borrowed_amount'], 0.0, places=2)

    def test_multiple_positions_cumulative_loan(self):
        """
        Test multiple purchases accumulate total borrowed amount and capital used.
        """
        # First purchase ($150K, borrows $75K - half the order)
        order_value_1 = 150_000.0
        expected_borrowed = order_value_1 * 0.5  # $75K
        borrowed_1 = expected_borrowed  # order > available_cash → borrow half
        self.miner_account_manager.process_order_buy(self.DEFAULT_MINER_HOTKEY, order_value_1, borrowed_1)

        # Should have borrowed $75K (half of $150K)
        self.assertAlmostEqual(borrowed_1, expected_borrowed, places=2)

        # bp = ($100K - $75K) * 1.5 = $37.5K, capital_used = $150K, borrowed = $75K
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], 37_500.0, places=2)
        self.assertAlmostEqual(account['capital_used'], 150_000.0, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], expected_borrowed, places=2)

        # Verify total borrowed tracking
        total_borrowed = self.miner_account_manager.get_total_borrowed_amount(
            self.DEFAULT_MINER_HOTKEY
        )
        self.assertAlmostEqual(total_borrowed, borrowed_1, places=2)

    def test_borrowed_amount_tracking(self):
        """
        Test that total_borrowed_amount tracks cumulative loans correctly.
        """
        # Start with no loan
        initial_borrowed = self.miner_account_manager.get_total_borrowed_amount(
            self.DEFAULT_MINER_HOTKEY
        )
        self.assertEqual(initial_borrowed, 0.0)

        # Buy $150K, borrows $75K (half the order)
        order_value = 150_000.0
        expected_borrowed = order_value * 0.5  # $75K
        borrowed = expected_borrowed  # order > available_cash → borrow half
        self.miner_account_manager.process_order_buy(self.DEFAULT_MINER_HOTKEY, order_value, borrowed)

        # Verify total borrowed amount is tracked
        total_borrowed = self.miner_account_manager.get_total_borrowed_amount(
            self.DEFAULT_MINER_HOTKEY
        )
        self.assertAlmostEqual(total_borrowed, borrowed, places=2)
        self.assertAlmostEqual(total_borrowed, expected_borrowed, places=2)

    def test_collateral_record_updates_buying_power(self):
        """
        Test that adding collateral records updates buying power correctly.
        Buying power = equity * multiplier, and equity = account_size + total_realized_pnl.
        """
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        initial_bp = account['buying_power']
        initial_account_size = account['account_size']

        # Add more collateral (increase account size to $150K)
        # Set it from yesterday so it's valid today
        new_collateral_theta = 150_000 / ValiConfig.COST_PER_THETA
        yesterday_ms = self.DEFAULT_OPEN_MS - (24 * 60 * 60 * 1000) - 1000
        self.miner_account_manager.set_miner_account_size(
            self.DEFAULT_MINER_HOTKEY,
            new_collateral_theta,
            timestamp_ms=yesterday_ms
        )

        # Buying power should increase by delta * multiplier (1.5x for Tier 2 equities)
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        expected_account_size_increase = 150_000 - initial_account_size
        expected_bp_increase = expected_account_size_increase * 1.5  # multiplier=1.5

        self.assertAlmostEqual(
            account['buying_power'],
            initial_bp + expected_bp_increase,
            places=2,
            msg="Buying power should increase by account_size_delta * multiplier"
        )

    def test_position_reduction(self):
        """
        Test position reduction scenario:
        1. Open position at 1.5x leverage ($150,000 position)
        2. Partial sell ($100,000 entry value, no PNL)
        3. Close remaining position with FLAT order ($50,000 entry value, no PNL)

        Starting with $100,000 equity, $150,000 buying power (Tier 2, 1.5x multiplier):
        - After 1.5x open: capital_used=$150K, borrowed=$75K, bp=($100K-$75K)*1.5=$37.5K
        - After partial sell: capital_used=$50K, borrowed=$0, bp=($100K-$50K)*1.5=$75K
        - After FLAT: capital_used=$0, borrowed=$0, bp=$100K*1.5=$150K
        """
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        initial_bp = account['buying_power']
        self.assertEqual(initial_bp, self.DEFAULT_ACCOUNT_SIZE * 1.5)  # $150K

        # Step 1: Open position at 1.5x leverage ($150K position)
        order_value_1 = 150_000.0
        borrowed_1 = order_value_1 * 0.5  # order > cash → borrow half = $75K
        self.miner_account_manager.process_order_buy(self.DEFAULT_MINER_HOTKEY, order_value_1, borrowed_1)

        self.assertAlmostEqual(borrowed_1, 75_000.0, places=2)

        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], 37_500.0, places=2,
                             msg="After 1.5x open: bp should be $37.5K")
        self.assertAlmostEqual(account['total_borrowed_amount'], 75_000.0, places=2,
                             msg="After 1.5x open: Borrowed should be $75K")

        # Step 2: Partial sell (close $100K entry value, no PNL)
        # loan_repaid = min(borrowed_1, exit_value) = min($75K, $100K) = $75K (full loan repaid)
        self.miner_account_manager.process_order_sell(
            self.DEFAULT_MINER_HOTKEY,
            100_000.0,  # entry_value
            0.0,        # realized_pnl
            borrowed_1  # position's full margin loan
        )

        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], 75_000.0, places=2,
                             msg="After partial sell: bp should be $75K")
        self.assertAlmostEqual(account['capital_used'], 50_000.0, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], 0.0, places=2,
                             msg="After partial sell: Borrowed should be $0")

        # Step 3: FLAT order closes remaining position ($50K entry value, no PNL)
        self.miner_account_manager.process_order_sell(
            self.DEFAULT_MINER_HOTKEY,
            50_000.0,  # entry_value
            0.0,       # realized_pnl
            0.0        # no remaining loan
        )

        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], 150_000.0, places=2,
                             msg="After FLAT: bp should return to $150K")
        self.assertAlmostEqual(account['capital_used'], 0.0, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], 0.0, places=2,
                             msg="After FLAT: Borrowed should be $0")



    # ==================== SUBACCOUNT_CHALLENGE Buying Power Tests ====================

    EQUITIES_MULTIPLIER = ValiConfig.TIER_PORTFOLIO_LEVERAGE[2][TradePairCategory.EQUITIES]   # 1.5 (Tier 2, <$200K)
    REDUCED_MULTIPLIER = ValiConfig.TIER_PORTFOLIO_LEVERAGE[1][TradePairCategory.EQUITIES]    # 1.0 (Tier 1, challenge)

    def test_subaccount_challenge_buying_power_reduced(self):
        """
        SUBACCOUNT_CHALLENGE (Tier 1) has a lower buying power multiplier than non-challenge (Tier 2).
        """
        # Verify normal buying power first
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        normal_bp = self.DEFAULT_ACCOUNT_SIZE * self.EQUITIES_MULTIPLIER
        self.assertAlmostEqual(account['buying_power'], normal_bp, places=2)

        # Set bucket to SUBACCOUNT_CHALLENGE
        self.miner_account_client.set_miner_bucket(
            self.DEFAULT_MINER_HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE
        )

        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        expected_bp = self.DEFAULT_ACCOUNT_SIZE * self.REDUCED_MULTIPLIER
        self.assertAlmostEqual(account['buying_power'], expected_bp, places=2)

    def test_subaccount_challenge_buying_power_with_capital_used(self):
        """
        SUBACCOUNT_CHALLENGE buying power accounts for capital_used.
        """
        # Set bucket to SUBACCOUNT_CHALLENGE
        self.miner_account_client.set_miner_bucket(
            self.DEFAULT_MINER_HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE
        )

        reduced_bp = self.DEFAULT_ACCOUNT_SIZE * self.REDUCED_MULTIPLIER
        order_value = 30_000.0

        # Buy within reduced buying power (within available cash, no borrowing)
        self.miner_account_manager.process_order_buy(self.DEFAULT_MINER_HOTKEY, order_value, 0.0)

        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], reduced_bp - order_value, places=2)
        self.assertAlmostEqual(account['capital_used'], order_value, places=2)

    def test_subaccount_challenge_insufficient_buying_power(self):
        """
        SUBACCOUNT_CHALLENGE with reduced buying power should reject orders exceeding it.
        Without the bucket, the same order would succeed under normal buying power.
        """
        reduced_bp = self.DEFAULT_ACCOUNT_SIZE * self.REDUCED_MULTIPLIER
        order_value = reduced_bp + 10_000.0  # exceeds reduced buying power

        # Set bucket to SUBACCOUNT_CHALLENGE
        self.miner_account_client.set_miner_bucket(
            self.DEFAULT_MINER_HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE
        )

        with self.assertRaises(SignalException):
            self.miner_account_manager.process_order_buy(
                self.DEFAULT_MINER_HOTKEY, order_value, order_value * 0.5
            )

        # Remove bucket — Tier 2 buying power ($150K) > order_value, same order should succeed
        self.miner_account_client.set_miner_bucket(self.DEFAULT_MINER_HOTKEY, None)
        borrowed = order_value * 0.5  # order ($110K) > cash ($100K) → borrow half
        self.miner_account_manager.process_order_buy(self.DEFAULT_MINER_HOTKEY, order_value, borrowed)
        self.assertAlmostEqual(borrowed, 55_000.0, places=2)  # half of $110K

    def test_buying_power_restored_after_bucket_change(self):
        """
        Changing bucket from SUBACCOUNT_CHALLENGE to SUBACCOUNT_FUNDED should
        restore normal buying power.
        """
        normal_bp = self.DEFAULT_ACCOUNT_SIZE * self.EQUITIES_MULTIPLIER
        reduced_bp = self.DEFAULT_ACCOUNT_SIZE * self.REDUCED_MULTIPLIER

        # Set bucket to SUBACCOUNT_CHALLENGE
        self.miner_account_client.set_miner_bucket(
            self.DEFAULT_MINER_HOTKEY, MinerBucket.SUBACCOUNT_CHALLENGE
        )

        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], reduced_bp, places=2)

        # Promote to SUBACCOUNT_FUNDED
        self.miner_account_client.set_miner_bucket(
            self.DEFAULT_MINER_HOTKEY, MinerBucket.SUBACCOUNT_FUNDED
        )

        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], normal_bp, places=2)


if __name__ == '__main__':
    unittest.main()
