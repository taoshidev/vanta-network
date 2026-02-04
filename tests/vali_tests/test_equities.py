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
        Test purchasing equities within buying power.
        Buying power should decrease by order value, and half is borrowed.

        Note: Initial buying_power = account_size * 2 = $200,000 (EQUITIES multiplier of 2).
        """
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        initial_bp = account['buying_power']

        # Verify we start with account_size * multiplier buying power
        self.assertEqual(initial_bp, self.DEFAULT_ACCOUNT_SIZE * 2)  # $200K

        # Purchase for $50,000 (within buying power)
        order_value = 50_000.0
        borrowed = self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value
        )

        # Should borrow half for equities
        self.assertAlmostEqual(borrowed, order_value / 2, places=2)  # $25K

        # Buying power should decrease by order value, capital_used should increase
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], initial_bp - order_value, places=2)  # $150K
        self.assertAlmostEqual(account['capital_used'], order_value, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], order_value / 2, places=2)

    def test_buy_large_position(self):
        """
        Test purchasing equities for a large position.
        Equities always borrow half the order value.

        Initial buying_power = $100K * 2 = $200K.
        """
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        initial_bp = account['buying_power']
        self.assertEqual(initial_bp, self.DEFAULT_ACCOUNT_SIZE * 2)  # $200K

        # Purchase for $150,000 (within buying power of $200K)
        order_value = 150_000.0
        expected_borrowed = order_value / 2  # $75,000

        borrowed = self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value
        )

        # Should borrow half of order value
        self.assertAlmostEqual(borrowed, expected_borrowed, places=2)

        # Buying power decreases, capital_used increases
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], initial_bp - order_value, places=2)  # $50K
        self.assertAlmostEqual(account['capital_used'], order_value, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], expected_borrowed, places=2)

    def test_insufficient_buying_power_raises_exception(self):
        """
        Test that buying with insufficient buying power raises SignalException.

        Initial buying_power = $100K * 2 = $200K.
        """
        # Try to purchase $250,000 worth (buying power is only $200K)
        order_value = 250_000.0

        with self.assertRaises(SignalException) as context:
            self.miner_account_manager.process_order_buy(
                self.DEFAULT_MINER_HOTKEY,
                order_value
            )

        self.assertIn("Insufficient buying power", str(context.exception))

    def test_sell_repays_loan_and_compounds_pnl(self):
        """
        Test selling equities repays margin loan and compounds realized PNL to equity.
        """
        # Buy $150K: capital_used=$150K, borrowed=$75K, bp=$50K
        order_value = 150_000.0
        borrowed = self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value
        )
        self.assertAlmostEqual(borrowed, 75_000.0, places=2)

        # Sell with $10K profit: entry=$150K, pnl=$10K
        entry_value = 150_000.0
        realized_pnl = 10_000.0
        loan_repaid = self.miner_account_manager.process_order_sell(
            self.DEFAULT_MINER_HOTKEY,
            entry_value,
            realized_pnl,
            borrowed  # position margin loan
        )

        # Should repay full loan ($75K)
        self.assertAlmostEqual(loan_repaid, borrowed, places=2)

        # Equity = $100K + $10K = $110K, buying_power = $110K * 2 = $220K
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['equity'], 110_000.0, places=2)
        self.assertAlmostEqual(account['buying_power'], 220_000.0, places=2)
        self.assertAlmostEqual(account['capital_used'], 0.0, places=2)
        self.assertEqual(account['total_borrowed_amount'], 0.0)

    def test_sell_at_loss_repays_loan_reduces_equity(self):
        """
        Test selling at a loss still repays full margin loan, equity absorbs the loss.
        """
        # Buy $150K: capital_used=$150K, borrowed=$75K
        order_value = 150_000.0
        borrowed = self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value
        )

        # Sell at massive loss: entry=$150K, pnl=-$100K (sold for $50K)
        entry_value = 150_000.0
        realized_pnl = -100_000.0
        loan_repaid = self.miner_account_manager.process_order_sell(
            self.DEFAULT_MINER_HOTKEY,
            entry_value,
            realized_pnl,
            borrowed
        )

        # Full loan repaid (not capped by proceeds in new model)
        self.assertAlmostEqual(loan_repaid, borrowed, places=2)  # $75K

        # Equity absorbs the loss: $100K + (-$100K) = $0
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['equity'], 0.0, places=2)
        self.assertAlmostEqual(account['buying_power'], 0.0, places=2)
        self.assertAlmostEqual(account['capital_used'], 0.0, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], 0.0, places=2)

    def test_equities_always_borrows_half(self):
        """
        Test that equities purchases always borrow half the order value.
        Even small purchases use margin in the equity-based model.
        """
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        initial_bp = account['buying_power']

        # Buy $50K - should still borrow $25K (half) for equities
        order_value = 50_000.0
        borrowed = self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value
        )

        self.assertAlmostEqual(borrowed, 25_000.0, places=2)
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], initial_bp - order_value, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], 25_000.0, places=2)

    def test_multiple_positions_cumulative_loan(self):
        """
        Test multiple purchases accumulate total borrowed amount and capital used.
        """
        # First purchase ($150K, borrows $75K)
        order_value_1 = 150_000.0
        borrowed_1 = self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value_1
        )

        # Should have borrowed $75K (half of $150K for equities)
        self.assertAlmostEqual(borrowed_1, 75_000.0, places=2)

        # bp = $200K - $150K = $50K, capital_used = $150K, borrowed = $75K
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], 50_000.0, places=2)
        self.assertAlmostEqual(account['capital_used'], 150_000.0, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], 75_000.0, places=2)

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

        # Buy $150K, borrows $75K (half for equities)
        order_value = 150_000.0
        borrowed = self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value
        )

        # Verify total borrowed amount is tracked
        total_borrowed = self.miner_account_manager.get_total_borrowed_amount(
            self.DEFAULT_MINER_HOTKEY
        )
        self.assertAlmostEqual(total_borrowed, borrowed, places=2)
        self.assertAlmostEqual(total_borrowed, 75_000.0, places=2)

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

        # Buying power should increase by delta * multiplier (2x for equities)
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        expected_account_size_increase = 150_000 - initial_account_size
        expected_bp_increase = expected_account_size_increase * 2  # multiplier=2

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

        Starting with $100,000 equity, $200,000 buying power (2x multiplier):
        - After 1.5x open: capital_used=$150K, borrowed=$75K, bp=$50K
        - After partial sell: capital_used=$50K, borrowed=$0, bp=$150K
        - After FLAT: capital_used=$0, borrowed=$0, bp=$200K
        """
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        initial_bp = account['buying_power']
        self.assertEqual(initial_bp, self.DEFAULT_ACCOUNT_SIZE * 2)  # $200K

        # Step 1: Open position at 1.5x leverage ($150K position)
        order_value_1 = 150_000.0
        borrowed_1 = self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value_1
        )

        self.assertAlmostEqual(borrowed_1, 75_000.0, places=2)

        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], 50_000.0, places=2,
                             msg="After 1.5x open: bp should be $50K")
        self.assertAlmostEqual(account['total_borrowed_amount'], 75_000.0, places=2,
                             msg="After 1.5x open: Borrowed should be $75K")

        # Step 2: Partial sell (close $100K entry value, no PNL)
        loan_repaid_1 = self.miner_account_manager.process_order_sell(
            self.DEFAULT_MINER_HOTKEY,
            100_000.0,  # entry_value
            0.0,        # realized_pnl
            borrowed_1  # position's margin loan
        )

        # Full loan repaid
        self.assertAlmostEqual(loan_repaid_1, 75_000.0, places=2)

        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], 150_000.0, places=2,
                             msg="After partial sell: bp should be $150K")
        self.assertAlmostEqual(account['capital_used'], 50_000.0, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], 0.0, places=2,
                             msg="After partial sell: Borrowed should be $0")

        # Step 3: FLAT order closes remaining position ($50K entry value, no PNL)
        remaining_loan = borrowed_1 - loan_repaid_1  # Should be 0
        loan_repaid_2 = self.miner_account_manager.process_order_sell(
            self.DEFAULT_MINER_HOTKEY,
            50_000.0,       # entry_value
            0.0,            # realized_pnl
            remaining_loan  # 0
        )

        self.assertAlmostEqual(loan_repaid_2, 0.0, places=2)

        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertAlmostEqual(account['buying_power'], 200_000.0, places=2,
                             msg="After FLAT: bp should return to $200K")
        self.assertAlmostEqual(account['capital_used'], 0.0, places=2)
        self.assertAlmostEqual(account['total_borrowed_amount'], 0.0, places=2,
                             msg="After FLAT: Borrowed should be $0")


    # ==================== Transaction History Tests ====================

    def test_transaction_history_buy(self):
        """
        Test that buy creates BUY transaction with correct deltas.
        For equities, loan_delta = order_value / 2 (borrow half).
        """
        # Clear any existing transactions
        tx_path = ValiBkpUtils.get_miner_transactions_path(
            self.DEFAULT_MINER_HOTKEY, running_unit_tests=True
        )
        ValiBkpUtils.clear_transactions(tx_path)

        # Make a purchase
        order_value = 50_000.0
        self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value
        )

        # Read transactions
        transactions = ValiBkpUtils.read_transactions(tx_path)

        self.assertEqual(len(transactions), 1)
        tx = transactions[0]
        self.assertEqual(tx['type'], 'BUY')
        self.assertAlmostEqual(tx['cash_delta'], -order_value, places=2)
        self.assertAlmostEqual(tx['loan_delta'], order_value / 2, places=2)  # borrow half for equities
        self.assertIn('timestamp_ms', tx)

    def test_transaction_history_buy_large(self):
        """
        Test that large buy creates BUY transaction with correct deltas.
        """
        tx_path = ValiBkpUtils.get_miner_transactions_path(
            self.DEFAULT_MINER_HOTKEY, running_unit_tests=True
        )
        ValiBkpUtils.clear_transactions(tx_path)

        # Make a large purchase ($150K, borrows $75K)
        order_value = 150_000.0
        expected_borrowed = order_value / 2  # $75K

        self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value
        )

        transactions = ValiBkpUtils.read_transactions(tx_path)

        self.assertEqual(len(transactions), 1)
        tx = transactions[0]
        self.assertEqual(tx['type'], 'BUY')
        self.assertAlmostEqual(tx['cash_delta'], -order_value, places=2)
        self.assertAlmostEqual(tx['loan_delta'], expected_borrowed, places=2)

    def test_transaction_history_sell(self):
        """
        Test that selling creates SELL transaction with correct deltas.
        """
        tx_path = ValiBkpUtils.get_miner_transactions_path(
            self.DEFAULT_MINER_HOTKEY, running_unit_tests=True
        )
        ValiBkpUtils.clear_transactions(tx_path)

        # First buy
        order_value = 150_000.0
        borrowed = self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value
        )

        # Then sell with profit
        entry_value = 150_000.0
        realized_pnl = 10_000.0
        loan_repaid = self.miner_account_manager.process_order_sell(
            self.DEFAULT_MINER_HOTKEY,
            entry_value,
            realized_pnl,
            borrowed
        )

        transactions = ValiBkpUtils.read_transactions(tx_path)

        self.assertEqual(len(transactions), 2)

        # Check SELL transaction: cash_delta = entry_value + realized_pnl
        sell_tx = transactions[1]
        self.assertEqual(sell_tx['type'], 'SELL')
        self.assertAlmostEqual(sell_tx['cash_delta'], entry_value + realized_pnl, places=2)
        self.assertAlmostEqual(sell_tx['loan_delta'], -loan_repaid, places=2)

    def test_transaction_history_interest_initialization(self):
        """
        Test that interest system initializes correctly.
        First day marks loan start but doesn't charge interest.
        """
        tx_path = ValiBkpUtils.get_miner_transactions_path(
            self.DEFAULT_MINER_HOTKEY, running_unit_tests=True
        )
        ValiBkpUtils.clear_transactions(tx_path)

        # Buy to create a loan (equities always borrows half)
        order_value = 150_000.0
        borrowed = self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            order_value
        )
        self.assertGreater(borrowed, 0)

        # Apply interest (first day initializes, no interest charged)
        accounts_processed = self.miner_account_manager.apply_daily_interest()

        # First call initializes but doesn't charge interest (no INTEREST tx)
        transactions = ValiBkpUtils.read_transactions(tx_path)

        # Should have 1 BUY transaction from the purchase
        self.assertEqual(len(transactions), 1)
        self.assertEqual(transactions[0]['type'], 'BUY')

        # Verify the account processed (initialization counts as processed)
        self.assertEqual(accounts_processed, 1)

        # Verify last_interest_date_ms was set
        account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        self.assertIsNotNone(account['last_interest_date_ms'])

    def test_reconstruct_account_from_transactions(self):
        """
        Test that account can be reconstructed from collateral records and transactions.
        """
        tx_path = ValiBkpUtils.get_miner_transactions_path(
            self.DEFAULT_MINER_HOTKEY, running_unit_tests=True
        )
        ValiBkpUtils.clear_transactions(tx_path)

        # Execute some operations
        # 1. Buy $50K (capital_used=$50K, borrowed=$25K)
        self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            50_000.0
        )

        # 2. Buy $80K (capital_used=$130K, borrowed=$65K)
        borrowed = self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            80_000.0
        )

        # 3. Sell: entry=$80K, pnl=$10K (repays $40K loan from 2nd position)
        self.miner_account_manager.process_order_sell(
            self.DEFAULT_MINER_HOTKEY,
            80_000.0,   # entry_value
            10_000.0,   # realized_pnl
            borrowed    # position margin loan ($40K)
        )

        # Get actual account state
        actual_account = self.miner_account_manager.get_account(self.DEFAULT_MINER_HOTKEY)
        actual_borrowed = actual_account['total_borrowed_amount']

        # Reconstruct account from transactions
        reconstructed = self.miner_account_manager.reconstruct_account_from_transactions(
            self.DEFAULT_MINER_HOTKEY
        )

        self.assertIsNotNone(reconstructed)
        self.assertAlmostEqual(reconstructed['total_borrowed_amount'], actual_borrowed, places=2,
                             msg="Reconstructed borrowed amount should match actual")

    def test_transaction_file_format_ndjson(self):
        """
        Test that transactions are stored in NDJSON format (one JSON object per line).
        """
        tx_path = ValiBkpUtils.get_miner_transactions_path(
            self.DEFAULT_MINER_HOTKEY, running_unit_tests=True
        )
        ValiBkpUtils.clear_transactions(tx_path)

        # Make multiple transactions
        self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            30_000.0
        )
        self.miner_account_manager.process_order_buy(
            self.DEFAULT_MINER_HOTKEY,
            20_000.0
        )

        # Read file directly and verify format
        import os
        self.assertTrue(os.path.exists(tx_path))

        with open(tx_path, 'r') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 2)
        for line in lines:
            # Each line should be valid JSON
            import json
            tx = json.loads(line.strip())
            self.assertIn('type', tx)
            self.assertIn('timestamp_ms', tx)
            self.assertIn('cash_delta', tx)
            self.assertIn('loan_delta', tx)


if __name__ == '__main__':
    unittest.main()
