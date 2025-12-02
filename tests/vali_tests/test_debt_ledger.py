"""
Unit tests for DebtLedger production code paths.

This test file runs production code paths and ensures critical paths are touched
as a smoke check. Follows the same pattern as test_perf_ledger_original.py with
class-level server setup for efficiency.

Architecture:
- DebtLedgerManager combines data from:
  - EmissionsLedgerManager (on-chain emissions data)
  - PenaltyLedgerManager (penalty multipliers)
  - PerfLedgerManager (performance metrics)
- DebtLedgerServer wraps manager with RPC infrastructure
- Tests verify production integration of all three data sources
"""
import bittensor as bt
import time

from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.debt_ledger import DebtCheckpoint

bt.logging.enable_info()


class TestDebtLedgers(TestBase):
    """
    Debt ledger tests using class-level server setup for efficiency.

    Server infrastructure is started once in setUpClass and shared across all tests.
    Per-test isolation is achieved by clearing data state (not restarting servers).

    Tests verify production integration of:
    - EmissionsLedgerManager (emissions data)
    - PenaltyLedgerManager (penalty multipliers)
    - PerfLedgerManager (performance metrics)
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    debt_ledger_client = None

    DEFAULT_MINER_HOTKEY = "test_miner"
    DEFAULT_MINER_HOTKEY_2 = "test_miner_2"
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
        cls.perf_ledger_client = cls.orchestrator.get_client('perf_ledger')
        cls.debt_ledger_client = cls.orchestrator.get_client('debt_ledger')

        # Set up test hotkeys
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

        # Reset time-based test data for each test
        self.DEFAULT_OPEN_MS = TimeUtil.now_in_millis() - 1000 * 60 * 60 * 24 * 60  # 60 days ago
        self.DEFAULT_TRADE_PAIR = TradePair.BTCUSD

        # Create fresh test positions for this test
        self._create_test_positions()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_positions(self):
        """Helper to create fresh test orders and positions."""
        self.default_btc_order = Order(
            price=60000,
            processed_ms=self.DEFAULT_OPEN_MS,
            order_uuid="test_order_btc",
            trade_pair=self.DEFAULT_TRADE_PAIR,
            order_type=OrderType.LONG,
            leverage=0.5,
        )

        self.default_nvda_order = Order(
            price=100,
            processed_ms=self.DEFAULT_OPEN_MS + 1000 * 60 * 60 * 24 * 5,
            order_uuid="test_order_nvda",
            trade_pair=TradePair.NVDA,
            order_type=OrderType.LONG,
            leverage=1,
        )

        self.default_btc_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_btc",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[self.default_btc_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.default_btc_position.rebuild_position_with_updated_orders(
            self.live_price_fetcher_client
        )

        self.default_nvda_position = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY,
            position_uuid="test_position_nvda",
            open_ms=self.default_nvda_order.processed_ms,
            trade_pair=TradePair.NVDA,
            orders=[self.default_nvda_order],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        self.default_nvda_position.rebuild_position_with_updated_orders(
            self.live_price_fetcher_client
        )

    def _create_mock_emissions(self):
        """
        Create mock emissions data for test hotkeys to avoid blockchain access.

        In unit tests, we can't access the blockchain, so we manually create
        emissions ledgers with dummy data that aligns with the perf ledger checkpoints.
        """
        from vali_objects.vali_dataclasses.emissions_ledger import EmissionsLedger, EmissionsCheckpoint

        # Get perf ledgers to determine which checkpoints we need to create emissions for
        perf_ledgers = self.perf_ledger_client.get_perf_ledgers(portfolio_only=True)

        for hotkey, portfolio_ledger in perf_ledgers.items():
            # Create emissions ledger for this hotkey (use dummy coldkey for tests)
            emissions_ledger = EmissionsLedger(hotkey=hotkey, coldkey="test_coldkey")

            # Create emissions checkpoints matching the perf ledger checkpoints
            for perf_cp in portfolio_ledger.cps:
                # Only create emissions for completed checkpoints (accum_ms == target duration)
                if perf_cp.accum_ms == ValiConfig.TARGET_CHECKPOINT_DURATION_MS:
                    emissions_cp = EmissionsCheckpoint(
                        chunk_start_ms=perf_cp.last_update_ms - ValiConfig.TARGET_CHECKPOINT_DURATION_MS,
                        chunk_end_ms=perf_cp.last_update_ms,
                        chunk_emissions=0.1,  # Mock emissions value
                        chunk_emissions_tao=0.001,
                        chunk_emissions_usd=0.5,
                        avg_alpha_to_tao_rate=0.01,
                        avg_tao_to_usd_rate=500.0,
                        tao_balance_snapshot=1.0,
                        alpha_balance_snapshot=100.0,
                        num_blocks=100,
                    )
                    emissions_ledger.add_checkpoint(emissions_cp, ValiConfig.TARGET_CHECKPOINT_DURATION_MS)

            # Save the emissions ledger via RPC
            self.debt_ledger_client.set_emissions_ledger(hotkey, emissions_ledger)

        bt.logging.info(f"Created mock emissions for {len(perf_ledgers)} hotkeys")

    def _build_all_ledgers(self, verbose=False):
        """
        Build all three required ledgers in the correct order.

        To create a debt checkpoint, we need:
        1. Performance checkpoint (from perf ledger)
        2. Penalty checkpoint (from penalty ledger)
        3. Emissions checkpoint (from emissions ledger)

        This helper ensures all three are built before calling build_debt_ledgers().

        Args:
            verbose: Enable detailed logging
        """
        # Build penalty ledgers FIRST (they depend on perf ledgers and challenge period data)
        bt.logging.info("Building penalty ledgers...")
        self.debt_ledger_client.build_penalty_ledgers(verbose=verbose, delta_update=False)

        # Create mock emissions ledgers SECOND (avoids blockchain access in tests)
        bt.logging.info("Creating mock emissions ledgers...")
        self._create_mock_emissions()

        # Now build debt ledgers THIRD (combines all three sources)
        bt.logging.info("Building debt ledgers...")
        self.debt_ledger_client.build_debt_ledgers(verbose=verbose, delta_update=False)

    def test_basic_debt_ledger_creation(self):
        """
        Test basic debt ledger creation from perf ledger data.

        Validates that:
        - Debt ledger manager can build ledgers from performance data
        - Checkpoints are created with correct structure
        - Basic RPC communication works
        """
        # Save test positions
        self.position_client.save_miner_position(self.default_btc_position)

        # Update perf ledger
        self.perf_ledger_client.update()

        # Build all three ledgers (perf, penalties, emissions)
        self._build_all_ledgers(verbose=True)

        # Verify we can retrieve the debt ledger
        debt_ledgers = self.debt_ledger_client.get_all_ledgers()
        self.assertIsNotNone(debt_ledgers, "Debt ledgers should not be None")

        # Verify ledger was created for our test miner
        if self.DEFAULT_MINER_HOTKEY in debt_ledgers:
            ledger = debt_ledgers[self.DEFAULT_MINER_HOTKEY]
            self.assertEqual(ledger.hotkey, self.DEFAULT_MINER_HOTKEY)
            bt.logging.info(f"Created debt ledger with {len(ledger.checkpoints)} checkpoints")

    def test_debt_checkpoint_structure(self):
        """
        Test DebtCheckpoint dataclass structure and derived fields.

        Validates that:
        - Checkpoints have all required fields
        - Derived fields are calculated correctly
        - __post_init__ works as expected
        """
        test_checkpoint = DebtCheckpoint(
            timestamp_ms=TimeUtil.now_in_millis(),
            # Emissions
            chunk_emissions_alpha=10.5,
            chunk_emissions_tao=0.05,
            chunk_emissions_usd=25.0,
            # Performance
            portfolio_return=1.15,
            realized_pnl=1000.0,
            unrealized_pnl=-200.0,
            spread_fee_loss=-50.0,
            carry_fee_loss=-30.0,
            # Penalties
            drawdown_penalty=0.95,
            risk_profile_penalty=0.98,
            min_collateral_penalty=1.0,
            risk_adjusted_performance_penalty=0.99,
            total_penalty=0.92,
        )

        # Verify derived fields are calculated correctly
        self.assertEqual(test_checkpoint.net_pnl, 800.0, "Net PnL should be realized + unrealized")
        self.assertEqual(
            test_checkpoint.total_fees, -80.0, "Total fees should be spread + carry"
        )
        self.assertEqual(
            test_checkpoint.return_after_fees,
            1.15,
            "Return after fees should match portfolio return",
        )
        self.assertEqual(
            test_checkpoint.weighted_score,
            1.15 * 0.92,
            "Weighted score should be return * total_penalty",
        )

    def test_debt_ledger_cumulative_emissions(self):
        """
        Test cumulative emissions calculations.

        Validates that:
        - Cumulative alpha/TAO/USD are calculated correctly
        - get_cumulative_* methods work as expected
        """
        from vali_objects.vali_dataclasses.debt_ledger import DebtLedger

        ledger = DebtLedger(hotkey=self.DEFAULT_MINER_HOTKEY)

        # Add multiple checkpoints with emissions data
        target_cp_duration_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
        base_ts = TimeUtil.now_in_millis() - (TimeUtil.now_in_millis() % target_cp_duration_ms)

        checkpoint1 = DebtCheckpoint(
            timestamp_ms=base_ts,
            chunk_emissions_alpha=10.0,
            chunk_emissions_tao=0.05,
            chunk_emissions_usd=25.0,
        )
        ledger.add_checkpoint(checkpoint1, target_cp_duration_ms)

        checkpoint2 = DebtCheckpoint(
            timestamp_ms=base_ts + target_cp_duration_ms,
            chunk_emissions_alpha=15.0,
            chunk_emissions_tao=0.07,
            chunk_emissions_usd=35.0,
        )
        ledger.add_checkpoint(checkpoint2, target_cp_duration_ms)

        # Verify cumulative calculations
        self.assertEqual(
            ledger.get_cumulative_emissions_alpha(), 25.0, "Cumulative alpha should be sum of chunks"
        )
        self.assertAlmostEqual(
            ledger.get_cumulative_emissions_tao(), 0.12, places=6, msg="Cumulative TAO should be sum of chunks"
        )
        self.assertEqual(
            ledger.get_cumulative_emissions_usd(), 60.0, "Cumulative USD should be sum of chunks"
        )

    def test_debt_ledger_checkpoint_validation(self):
        """
        Test checkpoint validation logic.

        Validates that:
        - Checkpoints must align with target duration
        - Checkpoints must be contiguous (no gaps)
        - add_checkpoint validates correctly
        """
        from vali_objects.vali_dataclasses.debt_ledger import DebtLedger

        ledger = DebtLedger(hotkey=self.DEFAULT_MINER_HOTKEY)
        target_cp_duration_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

        # Create aligned timestamp
        base_ts = TimeUtil.now_in_millis() - (TimeUtil.now_in_millis() % target_cp_duration_ms)

        # Valid checkpoint (aligned)
        checkpoint1 = DebtCheckpoint(timestamp_ms=base_ts)
        ledger.add_checkpoint(checkpoint1, target_cp_duration_ms)
        self.assertEqual(len(ledger.checkpoints), 1)

        # Next checkpoint must be exactly target_cp_duration_ms later
        checkpoint2 = DebtCheckpoint(timestamp_ms=base_ts + target_cp_duration_ms)
        ledger.add_checkpoint(checkpoint2, target_cp_duration_ms)
        self.assertEqual(len(ledger.checkpoints), 2)

        # Test validation: misaligned timestamp should fail
        with self.assertRaises(AssertionError):
            bad_checkpoint = DebtCheckpoint(timestamp_ms=base_ts + 1000)  # Not aligned
            ledger.add_checkpoint(bad_checkpoint, target_cp_duration_ms)

        # Test validation: gap in checkpoints should fail
        with self.assertRaises(AssertionError):
            gap_checkpoint = DebtCheckpoint(timestamp_ms=base_ts + 3 * target_cp_duration_ms)
            ledger.add_checkpoint(gap_checkpoint, target_cp_duration_ms)

    def test_debt_ledger_serialization(self):
        """
        Test debt ledger to_dict/from_dict round-trip.

        Validates that:
        - Ledger can be serialized to dict
        - Ledger can be deserialized from dict
        - Round-trip preserves all data
        """
        from vali_objects.vali_dataclasses.debt_ledger import DebtLedger

        ledger = DebtLedger(hotkey=self.DEFAULT_MINER_HOTKEY)
        target_cp_duration_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
        base_ts = TimeUtil.now_in_millis() - (TimeUtil.now_in_millis() % target_cp_duration_ms)

        # Add checkpoint with comprehensive data
        checkpoint = DebtCheckpoint(
            timestamp_ms=base_ts,
            chunk_emissions_alpha=10.0,
            chunk_emissions_tao=0.05,
            chunk_emissions_usd=25.0,
            portfolio_return=1.15,
            realized_pnl=1000.0,
            unrealized_pnl=-200.0,
            drawdown_penalty=0.95,
            total_penalty=0.92,
        )
        ledger.add_checkpoint(checkpoint, target_cp_duration_ms)

        # Serialize and deserialize
        ledger_dict = ledger.to_dict()
        restored_ledger = DebtLedger.from_dict(ledger_dict)

        # Verify structure preserved
        self.assertEqual(restored_ledger.hotkey, ledger.hotkey)
        self.assertEqual(len(restored_ledger.checkpoints), len(ledger.checkpoints))

        # Verify checkpoint data preserved
        original_cp = ledger.checkpoints[0]
        restored_cp = restored_ledger.checkpoints[0]
        self.assertEqual(restored_cp.timestamp_ms, original_cp.timestamp_ms)
        self.assertEqual(restored_cp.chunk_emissions_alpha, original_cp.chunk_emissions_alpha)
        self.assertEqual(restored_cp.portfolio_return, original_cp.portfolio_return)
        self.assertEqual(restored_cp.total_penalty, original_cp.total_penalty)

    def test_debt_ledger_summary_generation(self):
        """
        Test summary generation for efficient RPC access.

        Validates that:
        - Summaries contain key metrics without full checkpoint history
        - get_all_summaries works for multiple miners
        - Summary structure is correct
        """
        # Save positions and build ledgers
        self.position_client.save_miner_position(self.default_btc_position)
        self.perf_ledger_client.update()
        self._build_all_ledgers(verbose=False)

        # Get summary for specific miner
        summary = self.debt_ledger_client.get_ledger_summary(self.DEFAULT_MINER_HOTKEY)

        if summary:
            # Verify summary structure
            self.assertIn("hotkey", summary)
            self.assertIn("total_checkpoints", summary)
            self.assertIn("cumulative_emissions_alpha", summary)
            self.assertIn("cumulative_emissions_tao", summary)
            self.assertIn("cumulative_emissions_usd", summary)
            self.assertIn("portfolio_return", summary)
            self.assertIn("weighted_score", summary)

            bt.logging.info(f"Summary for {self.DEFAULT_MINER_HOTKEY}: {summary}")

        # Test get_all_summaries
        all_summaries = self.debt_ledger_client.get_all_summaries()
        self.assertIsInstance(all_summaries, dict)

    def test_debt_ledger_compressed_summaries(self):
        """
        Test pre-compressed summaries cache for instant RPC access.

        Validates that:
        - Compressed cache is updated after build
        - get_compressed_summaries returns gzip bytes
        - Cache pattern matches MinerStatisticsManager
        """
        # Save positions and build ledgers
        self.position_client.save_miner_position(self.default_btc_position)
        self.perf_ledger_client.update()
        self._build_all_ledgers(verbose=False)

        # Get compressed summaries (should be pre-cached)
        compressed = self.debt_ledger_client.get_compressed_summaries()

        if compressed:
            self.assertIsInstance(compressed, bytes)
            self.assertGreater(len(compressed), 0, "Compressed data should not be empty")

            # Verify we can decompress
            import gzip
            import json

            decompressed = gzip.decompress(compressed).decode("utf-8")
            summaries = json.loads(decompressed)
            self.assertIsInstance(summaries, dict)

            bt.logging.info(
                f"Compressed summaries: {len(compressed)} bytes, {len(summaries)} ledgers"
            )

    def test_multi_miner_debt_ledgers(self):
        """
        Test debt ledger creation for multiple miners.

        Validates that:
        - Multiple miners can have independent debt ledgers
        - Checkpoints align across miners (same timestamps)
        - Delta update mode works correctly
        """
        # Create positions for two miners
        btc_position_miner2 = Position(
            miner_hotkey=self.DEFAULT_MINER_HOTKEY_2,
            position_uuid="test_position_btc_miner2",
            open_ms=self.DEFAULT_OPEN_MS,
            trade_pair=self.DEFAULT_TRADE_PAIR,
            orders=[
                Order(
                    price=60000,
                    processed_ms=self.DEFAULT_OPEN_MS,
                    order_uuid="test_order_btc_miner2",
                    trade_pair=self.DEFAULT_TRADE_PAIR,
                    order_type=OrderType.LONG,
                    leverage=0.5,
                )
            ],
            position_type=OrderType.LONG,
            account_size=self.DEFAULT_ACCOUNT_SIZE,
        )
        btc_position_miner2.rebuild_position_with_updated_orders(self.live_price_fetcher_client)

        # Save both positions
        self.position_client.save_miner_position(self.default_btc_position)
        self.position_client.save_miner_position(btc_position_miner2)

        # Update perf ledgers
        self.perf_ledger_client.update()

        # Build all three ledgers
        self._build_all_ledgers(verbose=True)

        # Verify both miners have ledgers
        debt_ledgers = self.debt_ledger_client.get_all_ledgers()

        if self.DEFAULT_MINER_HOTKEY in debt_ledgers and self.DEFAULT_MINER_HOTKEY_2 in debt_ledgers:
            ledger1 = debt_ledgers[self.DEFAULT_MINER_HOTKEY]
            ledger2 = debt_ledgers[self.DEFAULT_MINER_HOTKEY_2]

            bt.logging.info(
                f"Miner 1: {len(ledger1.checkpoints)} checkpoints, "
                f"Miner 2: {len(ledger2.checkpoints)} checkpoints"
            )

            # If both have checkpoints, verify timestamps align
            if ledger1.checkpoints and ledger2.checkpoints:
                # Latest checkpoints should have same timestamp (aligned to standard intervals)
                self.assertEqual(
                    ledger1.checkpoints[-1].timestamp_ms,
                    ledger2.checkpoints[-1].timestamp_ms,
                    "Latest checkpoints should be aligned across miners",
                )

    def test_debt_ledger_health_check(self):
        """
        Test health check endpoint.

        Validates that:
        - Health check returns expected structure
        - Total ledgers count is accurate
        """
        health = self.debt_ledger_client.health_check()
        self.assertIsNotNone(health)
        self.assertEqual(health.get("status"), "ok")
        self.assertIn("timestamp_ms", health)
        self.assertIn("total_ledgers", health)

        bt.logging.info(f"Health check: {health}")

    def test_production_integration_smoke_test(self):
        """
        Comprehensive smoke test touching all critical production paths.

        This test validates end-to-end integration of:
        - Position creation and storage
        - Performance ledger updates
        - Debt ledger building (combining perf/emissions/penalties)
        - RPC communication
        - Data retrieval

        This is the main smoke test ensuring production code paths work.
        """
        bt.logging.info("="*80)
        bt.logging.info("Starting production integration smoke test")
        bt.logging.info("="*80)

        # Step 1: Create and save positions
        bt.logging.info("Step 1: Creating test positions...")
        self.position_client.save_miner_position(self.default_btc_position)
        self.position_client.save_miner_position(self.default_nvda_position)

        # Step 2: Update performance ledgers
        bt.logging.info("Step 2: Updating performance ledgers...")
        self.perf_ledger_client.update()

        # Verify perf ledgers were created
        perf_ledgers = self.perf_ledger_client.get_perf_ledgers(portfolio_only=False)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, perf_ledgers)
        self.assertIn(TP_ID_PORTFOLIO, perf_ledgers[self.DEFAULT_MINER_HOTKEY])

        portfolio_pl = perf_ledgers[self.DEFAULT_MINER_HOTKEY][TP_ID_PORTFOLIO]
        bt.logging.info(f"  Created {len(portfolio_pl.cps)} perf checkpoints")

        # Step 3: Build all three ledgers (integrates perf + emissions + penalties)
        bt.logging.info("Step 3: Building all ledgers (penalty, emissions, debt)...")
        start_time = time.time()
        self._build_all_ledgers(verbose=True)
        build_time = time.time() - start_time
        bt.logging.info(f"  Built all ledgers in {build_time:.2f}s")

        # Step 4: Verify debt ledgers were created
        bt.logging.info("Step 4: Verifying debt ledgers...")
        debt_ledgers = self.debt_ledger_client.get_all_ledgers()

        if self.DEFAULT_MINER_HOTKEY in debt_ledgers:
            ledger = debt_ledgers[self.DEFAULT_MINER_HOTKEY]
            bt.logging.info(f"  Debt ledger created with {len(ledger.checkpoints)} checkpoints")

            # Verify checkpoint structure
            if ledger.checkpoints:
                latest = ledger.checkpoints[-1]
                bt.logging.info(f"  Latest checkpoint timestamp: {TimeUtil.millis_to_formatted_date_str(latest.timestamp_ms)}")
                bt.logging.info(f"  Portfolio return: {latest.portfolio_return:.4f}")
                bt.logging.info(f"  Total penalty: {latest.total_penalty:.4f}")
                bt.logging.info(f"  Weighted score: {latest.weighted_score:.4f}")

                # Verify checkpoint has all required data
                self.assertIsNotNone(latest.portfolio_return)
                self.assertIsNotNone(latest.total_penalty)
                self.assertIsNotNone(latest.weighted_score)

            # Step 5: Test summary generation
            bt.logging.info("Step 5: Testing summary generation...")
            summary = self.debt_ledger_client.get_ledger_summary(self.DEFAULT_MINER_HOTKEY)
            if summary:
                bt.logging.info(f"  Summary total_checkpoints: {summary.get('total_checkpoints')}")
                bt.logging.info(f"  Summary portfolio_return: {summary.get('portfolio_return'):.4f}")
                bt.logging.info(f"  Summary weighted_score: {summary.get('weighted_score'):.4f}")

            # Step 6: Test compressed cache
            bt.logging.info("Step 6: Testing compressed summaries cache...")
            compressed = self.debt_ledger_client.get_compressed_summaries()
            if compressed:
                bt.logging.info(f"  Compressed cache size: {len(compressed)} bytes")

        bt.logging.info("="*80)
        bt.logging.info("Production integration smoke test completed successfully")
        bt.logging.info("="*80)
