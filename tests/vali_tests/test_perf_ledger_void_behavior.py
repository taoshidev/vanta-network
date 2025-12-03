"""
Performance ledger void behavior tests.

This file consolidates all void-related tests:
- Void filling and bypass logic
- Floating point drift prevention
- Multi-trade pair void scenarios
- Edge cases during void periods
"""

import unittest

from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil, MS_IN_24_HOURS
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import (
    PerfLedger,
    PerfCheckpoint,
    TP_ID_PORTFOLIO,
    TradePairReturnStatus,
)


class TestPerfLedgerVoidBehavior(TestBase):
    """
    Tests for performance ledger void period behavior using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    This eliminates redundant server spawning and dramatically reduces test startup time.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    perf_ledger_server = None  # Keep server handle for internal access
    elimination_client = None
    challenge_period_client = None
    plagiarism_client = None

    DEFAULT_ACCOUNT_SIZE = 100_000
    test_hotkey = "test_miner_void"

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
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.plagiarism_client = cls.orchestrator.get_client('plagiarism')

        # Get server handle for internal access (needed for test_bypass_logic_direct)
        cls.perf_ledger_server = cls.orchestrator._servers.get('perf_ledger')

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

        self.test_hotkey = "test_miner_void"
        self.now_ms = TimeUtil.now_in_millis()

        # Set up metagraph with test hotkey
        self.metagraph_client.set_hotkeys([self.test_hotkey])

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def validate_void_checkpoint(self, cp: PerfCheckpoint, context: str = ""):
        """Validate void checkpoint has expected characteristics."""
        # Void checkpoints should have 0 updates
        self.assertEqual(cp.n_updates, 0, f"{context}: void checkpoint should have 0 updates")
        
        # Void checkpoints should have no new gains/losses
        self.assertEqual(cp.gain, 0.0, f"{context}: void checkpoint should have no gain")
        self.assertEqual(cp.loss, 0.0, f"{context}: void checkpoint should have no loss")
        
        # Portfolio values should be reasonable
        self.assertGreater(cp.prev_portfolio_ret, 0.0, f"{context}: portfolio return should be positive")
        self.assertGreater(cp.prev_portfolio_spread_fee, 0.0, f"{context}: spread fee should be positive")
        self.assertGreater(cp.prev_portfolio_carry_fee, 0.0, f"{context}: carry fee should be positive")
        
        # Risk metrics should be reasonable
        self.assertGreater(cp.mdd, 0.0, f"{context}: MDD should be positive")
        self.assertGreater(cp.mpv, 0.0, f"{context}: MPV should be positive")
        
        # Carry fee loss during void should be 0 (this was the original bug)
        self.assertEqual(cp.carry_fee_loss, 0.0, f"{context}: void checkpoint should have 0 carry_fee_loss")

    def test_void_filling_prevents_drift(self):
        """
        Test that void filling with bypass logic prevents floating point drift.
        This is the core test for the original bug fix.
        """
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        for boundary_offset_ms in [0, 1000, 60000]:
            # Clear data for each sub-test
            self.perf_ledger_client.clear_all_ledger_data()
            self.position_client.clear_all_miner_positions_and_disk()

            base_time = (self.now_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS - (365 * MS_IN_24_HOURS)
            close_ms = base_time + (3 * MS_IN_24_HOURS)

            # Create position that will generate carry fees
            position = Position(
                miner_hotkey=self.test_hotkey,
                position_uuid="drift_test",
                open_ms=base_time,
                close_ms=close_ms,
                trade_pair=TradePair.BTCUSD,
                account_size=self.DEFAULT_ACCOUNT_SIZE,
                orders=[
                    Order(
                        price=50000.0,
                        processed_ms=base_time,
                        order_uuid="open",
                        trade_pair=TradePair.BTCUSD,
                        order_type=OrderType.LONG,
                        leverage=1.0,
                    ),
                    Order(
                        price=50000.0,
                        processed_ms=close_ms,
                        order_uuid="close",
                        trade_pair=TradePair.BTCUSD,
                        order_type=OrderType.FLAT,
                        leverage=0.0,
                    )
                ],
                position_type=OrderType.FLAT,
                is_closed_position=True,
            )
            position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)
            self.position_client.save_miner_position(position)

            # Process position
            self.perf_ledger_client.update(t_ms=close_ms + 5000)

            # Get checkpoint values at close
            bundles = self.perf_ledger_client.get_perf_ledgers(portfolio_only=False)
            btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]

            # Find last active checkpoint
            close_checkpoint = None
            for i, cp in enumerate(btc_ledger.cps):
                if close_checkpoint is None and cp.prev_portfolio_spread_fee == .998:
                    close_checkpoint = cp
                    print('@@@@@ found close cp', i, cp)
                    break

            assert close_checkpoint
            self.assertEqual(close_checkpoint.n_updates, 1)

            self.assertIsNotNone(close_checkpoint)

            print('------------------------------')
            for i, cp in enumerate(btc_ledger.cps):
                print(TimeUtil.millis_to_formatted_date_str(cp.last_update_ms), i, cp)

            print('------------------------------')
            # Perform many void updates
            for update_round_idx in range(1, 50):  # 50 days of void
                void_checkpoints = []
                self.perf_ledger_client.update(t_ms=base_time + (3 + update_round_idx) * MS_IN_24_HOURS + boundary_offset_ms)

                bundles = self.perf_ledger_client.get_perf_ledgers(portfolio_only=False)
                btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]
                portfolio_ledger = bundles[self.test_hotkey][TP_ID_PORTFOLIO]

                assert len(btc_ledger.cps) == len(portfolio_ledger.cps)
                lb = 6 + update_round_idx * 2
                assert len(btc_ledger.cps) in list(range(lb + 3))
                for cp_btc, cp_portfolio in zip(btc_ledger.cps, portfolio_ledger.cps):
                    self.assertEqual(cp_btc, cp_portfolio)

                print(f'-------------- update round index {update_round_idx} boundary offset {boundary_offset_ms}----------------')
                for i, cp in enumerate(btc_ledger.cps):
                    print(TimeUtil.millis_to_formatted_date_str(cp.last_update_ms), i, cp)
                print('-----------------------------------------------------------')

                for cp in btc_ledger.cps:
                    if cp.last_update_ms > close_checkpoint.last_update_ms:
                        void_checkpoints.append(cp)

                # Verify no drift - all void checkpoints should be identical
                n = len(void_checkpoints)
                lb = update_round_idx * 2
                self.assertIn(n, list(range(lb, lb+3)))

                for i, cp in enumerate(void_checkpoints):
                    # Exact equality - no tolerance
                    self.assertEqual(cp.prev_portfolio_ret, close_checkpoint.prev_portfolio_ret,
                                   f"Void checkpoint {i}/{n}: return drifted")
                    self.assertEqual(cp.prev_portfolio_carry_fee, close_checkpoint.prev_portfolio_carry_fee,
                                   f"Void checkpoint {i}/{n}: carry fee drifted")
                    self.assertEqual(cp.prev_portfolio_spread_fee, close_checkpoint.prev_portfolio_spread_fee,
                                     f"Void checkpoint {i}/{n}: spread fee drifted. update round index {update_round_idx}")
                    self.assertEqual(cp.mdd, close_checkpoint.mdd,
                                   f"Void checkpoint {i}/{n}: MDD drifted")

                    # Validate this as a proper void checkpoint
                    self.validate_void_checkpoint(cp, f"Void checkpoint {i}")

    def test_multi_tp_staggered_void_periods(self):
        """Test void behavior with multiple trade pairs having different timings."""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        base_time = (self.now_ms // MS_IN_24_HOURS) * MS_IN_24_HOURS - (30 * MS_IN_24_HOURS)

        # Create staggered positions
        positions = [
            ("btc", TradePair.BTCUSD, 0, 10),    # Days 0-10
            ("eth", TradePair.ETHUSD, 5, 15),    # Days 5-15 (overlaps BTC)
            ("jpy", TradePair.USDJPY, 12, 18),   # Days 12-18
        ]

        for name, tp, start_day, end_day in positions:
            position = self._create_position(
                name, tp,
                base_time + (start_day * MS_IN_24_HOURS),
                base_time + (end_day * MS_IN_24_HOURS),
                1000.0, 1000.0, OrderType.LONG
            )
            self.position_client.save_miner_position(position)

        # Update to day 25
        self.perf_ledger_client.update(t_ms=base_time + (25 * MS_IN_24_HOURS))

        bundles = self.perf_ledger_client.get_perf_ledgers(portfolio_only=False)
        bundle = bundles[self.test_hotkey]

        # Verify each TP has correct void period
        btc_ledger = bundle[TradePair.BTCUSD.trade_pair_id]
        eth_ledger = bundle[TradePair.ETHUSD.trade_pair_id]
        jpy_ledger = bundle[TradePair.USDJPY.trade_pair_id]

        # Count void checkpoints for each
        btc_void = sum(1 for cp in btc_ledger.cps if cp.n_updates == 0 and
                      cp.last_update_ms > base_time + (10 * MS_IN_24_HOURS))
        eth_void = sum(1 for cp in eth_ledger.cps if cp.n_updates == 0 and
                      cp.last_update_ms > base_time + (15 * MS_IN_24_HOURS))
        jpy_void = sum(1 for cp in jpy_ledger.cps if cp.n_updates == 0 and
                      cp.last_update_ms > base_time + (18 * MS_IN_24_HOURS))

        # BTC should have most void checkpoints (closed earliest)
        self.assertGreater(btc_void, eth_void)
        self.assertGreater(eth_void, jpy_void)

    def test_bypass_logic_direct(self):
        """Test the bypass logic utility function via RPC client."""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        # Create test ledger with checkpoint
        ledger = PerfLedger(initialization_time_ms=self.now_ms)
        prev_cp = PerfCheckpoint(
            last_update_ms=self.now_ms,
            prev_portfolio_ret=0.95,
            prev_portfolio_spread_fee=0.999,
            prev_portfolio_carry_fee=0.998,
            mdd=0.95,
            mpv=1.0
        )
        ledger.cps.append(prev_cp)

        # Test case 1: Should use bypass
        ret, spread, carry = self.perf_ledger_client.get_bypass_values_if_applicable(
            ledger, "BTCUSD", TradePairReturnStatus.TP_NO_OPEN_POSITIONS,
            1.0, .999, .998, {"BTCUSD": None}
        )
        self.assertEqual(ret, 0.95)
        self.assertEqual(spread, 0.999)
        self.assertEqual(carry, 0.998)

        # Test case 2: Should NOT use bypass (position just closed)
        # Create a closed position to simulate a position that just closed
        closed_position = self._create_position(
            "closed_test", TradePair.BTCUSD,
            self.now_ms - MS_IN_24_HOURS, self.now_ms,
            50000.0, 50000.0, OrderType.LONG
        )
        ret, spread, carry = self.perf_ledger_client.get_bypass_values_if_applicable(
            ledger, "BTCUSD", TradePairReturnStatus.TP_NO_OPEN_POSITIONS,
            1.0, 1.0, 1.0, {"BTCUSD": closed_position}
        )
        self.assertEqual(ret, 1.0)

        # Test case 3: Should NOT use bypass (positions open)
        ret, spread, carry = self.perf_ledger_client.get_bypass_values_if_applicable(
            ledger, "BTCUSD", TradePairReturnStatus.TP_MARKET_OPEN_PRICE_CHANGE,
            1.0, 1.0, 1.0, {"BTCUSD": None}
        )
        self.assertEqual(ret, 1.0)

        # Test case 4: Should NOT use bypass (different TP)
        ret, spread, carry = self.perf_ledger_client.get_bypass_values_if_applicable(
            ledger, "ETHUSD", TradePairReturnStatus.TP_NO_OPEN_POSITIONS,
            1.0, 1.0, 1.0, {"BTCUSD": None}
        )
        self.assertEqual(ret, 1.0)

    def test_void_checkpoint_characteristics(self):
        """Test that void checkpoints have expected characteristics."""
        # No mocking needed - LivePriceFetcherClient with running_unit_tests=True handles test data

        base_time = self.now_ms - (10 * MS_IN_24_HOURS)

        # Create and close position
        position = self._create_position(
            "char_test", TradePair.BTCUSD,
            base_time, base_time + MS_IN_24_HOURS,
            50000.0, 50000.0, OrderType.LONG
        )
        self.position_client.save_miner_position(position)

        # Update through void period
        self.perf_ledger_client.update(t_ms=base_time + (5 * MS_IN_24_HOURS))

        bundles = self.perf_ledger_client.get_perf_ledgers(portfolio_only=False)
        btc_ledger = bundles[self.test_hotkey][TradePair.BTCUSD.trade_pair_id]

        # Check void checkpoint characteristics
        void_checkpoint_count = 0
        for cp in btc_ledger.cps:
            if cp.n_updates == 0 and cp.last_update_ms > base_time + MS_IN_24_HOURS:
                void_checkpoint_count += 1
                # Validate void checkpoint characteristics
                self.validate_void_checkpoint(cp, f"Void checkpoint at {cp.last_update_ms}")

        self.assertGreater(void_checkpoint_count, 0, "Should have found at least one void checkpoint")

    def _create_position(self, position_id: str, trade_pair: TradePair,
                        open_ms: int, close_ms: int, open_price: float,
                        close_price: float, order_type: OrderType) -> Position:
        """Helper to create a position."""
        position = Position(
            miner_hotkey=self.test_hotkey,
            position_uuid=position_id,
            open_ms=open_ms,
            close_ms=close_ms,
            trade_pair=trade_pair,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
            orders=[
                Order(
                    price=open_price,
                    processed_ms=open_ms,
                    order_uuid=f"{position_id}_open",
                    trade_pair=trade_pair,
                    order_type=order_type,
                    leverage=1.0 if order_type == OrderType.LONG else -1.0,
                ),
                Order(
                    price=close_price,
                    processed_ms=close_ms,
                    order_uuid=f"{position_id}_close",
                    trade_pair=trade_pair,
                    order_type=OrderType.FLAT,
                    leverage=0.0,
                )
            ],
            position_type=OrderType.FLAT,
            is_closed_position=True,
        )
        position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)
        return position


if __name__ == '__main__':
    unittest.main()
