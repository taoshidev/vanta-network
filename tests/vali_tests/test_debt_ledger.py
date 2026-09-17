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
import time
from types import SimpleNamespace

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import MS_IN_WEEK, TimeUtil
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtCheckpoint, DebtLedger, WeekTrack, apply_deferral
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_manager import DebtLedgerManager
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.vali_dataclasses.ledger.debt.weekly_seal_ledger import (
    SealedWeek, SettledSegment, WeeklySealLedger,
)
import logging
import os
from shared_objects.log import logger

logger.setLevel(logging.INFO)


def _clear_seal_ledger():
    """Drop the on-disk seal ledger so each test starts with nothing settled.

    The seal ledger deliberately outlives every other ledger, including across process restarts,
    so a test that seals a week would otherwise pin that decision for every later test and run.
    """
    path = WeeklySealLedger(running_unit_tests=True)._get_path()
    if os.path.exists(path):
        os.remove(path)


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
        from vali_objects.vali_dataclasses.ledger.emission.emissions_ledger import EmissionsLedger, EmissionsCheckpoint

        # Get perf ledgers to determine which checkpoints we need to create emissions for
        perf_ledgers = self.perf_ledger_client.get_perf_ledgers()

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

        logger.info(f"Created mock emissions for {len(perf_ledgers)} hotkeys")

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
        logger.info("Building penalty ledgers...")
        self.debt_ledger_client.build_penalty_ledgers(verbose=verbose, delta_update=False)

        # Create mock emissions ledgers SECOND (avoids blockchain access in tests)
        logger.info("Creating mock emissions ledgers...")
        self._create_mock_emissions()

        # Now build debt ledgers THIRD (combines all three sources)
        logger.info("Building debt ledgers...")
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
            logger.info(f"Created debt ledger with {len(ledger.checkpoints)} checkpoints")

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
            # Penalties
            drawdown_penalty=0.95,
            risk_profile_penalty=0.98,
            min_collateral_penalty=1.0,
            risk_adjusted_performance_penalty=0.99,
            total_penalty=0.92,
        )

        # Verify derived fields are calculated correctly
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
        from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtLedger

        ledger = DebtLedger(hotkey=self.DEFAULT_MINER_HOTKEY)

        # Add multiple checkpoints with emissions data
        target_cp_duration_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
        now_ms = TimeUtil.now_in_millis()
        base_ts = now_ms - (now_ms % target_cp_duration_ms)

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
        from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtLedger

        ledger = DebtLedger(hotkey=self.DEFAULT_MINER_HOTKEY)
        target_cp_duration_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

        # Create aligned timestamp
        now_ms = TimeUtil.now_in_millis()
        base_ts = now_ms - (now_ms % target_cp_duration_ms)

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
        from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtLedger

        ledger = DebtLedger(hotkey=self.DEFAULT_MINER_HOTKEY)
        target_cp_duration_ms = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
        now_ms = TimeUtil.now_in_millis()
        base_ts = now_ms - (now_ms % target_cp_duration_ms)

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

            logger.info(f"Summary for {self.DEFAULT_MINER_HOTKEY}: {summary}")

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

            logger.info(
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

            logger.info(
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

        logger.info(f"Health check: {health}")

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
        logger.info("="*80)
        logger.info("Starting production integration smoke test")
        logger.info("="*80)

        # Step 1: Create and save positions
        logger.info("Step 1: Creating test positions...")
        self.position_client.save_miner_position(self.default_btc_position)
        self.position_client.save_miner_position(self.default_nvda_position)

        # Step 2: Update performance ledgers
        logger.info("Step 2: Updating performance ledgers...")
        self.perf_ledger_client.update()

        # Verify perf ledgers were created
        perf_ledgers = self.perf_ledger_client.get_perf_ledgers()
        self.assertIn(self.DEFAULT_MINER_HOTKEY, perf_ledgers)
        self.assertIn(self.DEFAULT_MINER_HOTKEY, perf_ledgers)

        portfolio_pl = perf_ledgers[self.DEFAULT_MINER_HOTKEY]
        logger.info(f"  Created {len(portfolio_pl.cps)} perf checkpoints")

        # Step 3: Build all three ledgers (integrates perf + emissions + penalties)
        logger.info("Step 3: Building all ledgers (penalty, emissions, debt)...")
        start_time = time.time()
        self._build_all_ledgers(verbose=True)
        build_time = time.time() - start_time
        logger.info(f"  Built all ledgers in {build_time:.2f}s")

        # Step 4: Verify debt ledgers were created
        logger.info("Step 4: Verifying debt ledgers...")
        debt_ledgers = self.debt_ledger_client.get_all_ledgers()

        if self.DEFAULT_MINER_HOTKEY in debt_ledgers:
            ledger = debt_ledgers[self.DEFAULT_MINER_HOTKEY]
            logger.info(f"  Debt ledger created with {len(ledger.checkpoints)} checkpoints")

            # Verify checkpoint structure
            if ledger.checkpoints:
                latest = ledger.checkpoints[-1]
                logger.info(f"  Latest checkpoint timestamp: {TimeUtil.millis_to_formatted_date_str(latest.timestamp_ms)}")
                logger.info(f"  Portfolio return: {latest.portfolio_return:.4f}")
                logger.info(f"  Total penalty: {latest.total_penalty:.4f}")
                logger.info(f"  Weighted score: {latest.weighted_score:.4f}")

                # Verify checkpoint has all required data
                self.assertIsNotNone(latest.portfolio_return)
                self.assertIsNotNone(latest.total_penalty)
                self.assertIsNotNone(latest.weighted_score)

            # Step 5: Test summary generation
            logger.info("Step 5: Testing summary generation...")
            summary = self.debt_ledger_client.get_ledger_summary(self.DEFAULT_MINER_HOTKEY)
            if summary:
                logger.info(f"  Summary total_checkpoints: {summary.get('total_checkpoints')}")
                logger.info(f"  Summary portfolio_return: {summary.get('portfolio_return'):.4f}")
                logger.info(f"  Summary weighted_score: {summary.get('weighted_score'):.4f}")

            # Step 6: Test compressed cache
            logger.info("Step 6: Testing compressed summaries cache...")
            compressed = self.debt_ledger_client.get_compressed_summaries()
            if compressed:
                logger.info(f"  Compressed cache size: {len(compressed)} bytes")

        logger.info("="*80)
        logger.info("Production integration smoke test completed successfully")
        logger.info("="*80)

    def test_settled_segments_survive_deleting_the_debt_ledger(self):
        """The wipe _switch_account runs: the debt and penalty ledgers go, the settled money stays.

        Exercises the real RPC path the promotion uses, so a settled segment is proven to outlive
        the delete that happens moments after it is written.
        """
        _clear_seal_ledger()
        hotkey = "wounddown_1"
        week_start_ms = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis())
        promotion_ms = week_start_ms + MS_IN_WEEK // 2
        try:
            self.assertTrue(self.debt_ledger_client.record_settled_segment(
                hotkey, week_start_ms, week_start_ms, promotion_ms,
                MinerBucket.PRO_CHALLENGE_TRANSITION.value, 1843.20, 1843.20, 1.0, 1.0,
            ))

            self.debt_ledger_client.delete_debt_ledger(hotkey)

            segments = self.debt_ledger_client.get_settled_segments(hotkey)
            self.assertEqual(len(segments), 1)
            self.assertEqual(segments[0].payout_usd, 1843.20)
            self.assertEqual(segments[0].bucket, MinerBucket.PRO_CHALLENGE_TRANSITION.value)
        finally:
            self.debt_ledger_client.remove_settled_segment(hotkey, promotion_ms)
            _clear_seal_ledger()


class TestApplyDeferral(TestBase):
    """One payout week of escrow bookkeeping: every dollar is released, carried, or forfeited."""

    def test_clean_pro_week_releases_balance(self):
        self.assertEqual(apply_deferral(100.0, 0.0, track=WeekTrack.ON_TRACK, week_penalty=1.0), (100.0, 0.0, 0.0))

    def test_breach_week_holds_balance_and_withheld(self):
        self.assertEqual(apply_deferral(100.0, 40.0, track=WeekTrack.ON_TRACK, week_penalty=0.0), (0.0, 140.0, 0.0))

    def test_quiet_week_carries_balance(self):
        self.assertEqual(apply_deferral(100.0, 0.0, track=WeekTrack.NO_DATA, week_penalty=1.0), (0.0, 100.0, 0.0))

    def test_off_track_week_forfeits_balance_and_withheld(self):
        self.assertEqual(apply_deferral(100.0, 40.0, track=WeekTrack.OFF_TRACK, week_penalty=0.0), (0.0, 0.0, 140.0))

    def test_off_track_week_with_nothing_held_forfeits_nothing(self):
        self.assertEqual(apply_deferral(0.0, 0.0, track=WeekTrack.OFF_TRACK, week_penalty=1.0), (0.0, 0.0, 0.0))

    def test_forfeited_escrow_does_not_return_when_the_account_comes_back_on_track(self):
        # Breach week holds 140; leaving the track forfeits it all
        released, balance, forfeited = apply_deferral(0.0, 140.0, track=WeekTrack.ON_TRACK, week_penalty=0.0)
        self.assertEqual((released, balance, forfeited), (0.0, 140.0, 0.0))
        released, balance, forfeited = apply_deferral(balance, 0.0, track=WeekTrack.OFF_TRACK, week_penalty=1.0)
        self.assertEqual((released, balance, forfeited), (0.0, 0.0, 140.0))
        # Back on the track: a new breach holds only its own withheld amount
        released, balance, forfeited = apply_deferral(balance, 30.0, track=WeekTrack.ON_TRACK, week_penalty=0.0)
        self.assertEqual((released, balance, forfeited), (0.0, 30.0, 0.0))
        # ... and the next clean week releases only that, never the forfeited 140
        released, balance, forfeited = apply_deferral(balance, 0.0, track=WeekTrack.ON_TRACK, week_penalty=1.0)
        self.assertEqual((released, balance, forfeited), (30.0, 0.0, 0.0))


class TestEntityWeeklyPenaltyAggregation(TestBase):
    """Entity aggregation honors a subaccount's weekly penalty for the whole payout week."""

    ENTITY_HOTKEY = "entity"
    SUBACCOUNT_HOTKEY = "entity_1"
    CP_DURATION_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    def setUp(self):
        super().setUp()
        _clear_seal_ledger()

    def _build_manager(self, subaccount_ledger):
        """Bare manager with only the collaborators aggregate_entity_debt_ledgers touches."""
        manager = object.__new__(DebtLedgerManager)
        # Keeps the lazily built weekly seal ledger inside the test validation directory
        manager.running_unit_tests = True
        manager.debt_ledgers = {self.SUBACCOUNT_HOTKEY: subaccount_ledger}
        manager.emissions_ledger_manager = SimpleNamespace(get_ledger=lambda _hotkey: None)
        manager._entity_client = SimpleNamespace(get_all_entities=lambda: {
            self.ENTITY_HOTKEY: {'subaccounts': {'1': {
                'status': 'active', 'synthetic_hotkey': self.SUBACCOUNT_HOTKEY, 'reg_fee_theta': 1.0,
            }}}
        })
        manager._perf_ledger_client = SimpleNamespace(get_frozen_ledgers=lambda: {})
        manager._challengeperiod_client = SimpleNamespace(
            get_miner_bucket=lambda _hotkey: MinerBucket.ENTITY
        )
        return manager

    def _subaccount_ledger(self, blocked_checkpoint_indices=(), bucket_by_index=None):
        """One earning checkpoint per 12h for two weeks, each realizing 10 USD.

        `bucket_by_index` overrides the bucket stamped on individual checkpoints, which is how a
        promotion lands mid-week; everything else is PRO_FUNDED.
        """
        week_0_start = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK
        bucket_by_index = bucket_by_index or {}
        checkpoints = []
        for i in range(2 * MS_IN_WEEK // self.CP_DURATION_MS):
            checkpoints.append(DebtCheckpoint(
                timestamp_ms=week_0_start + (i + 1) * self.CP_DURATION_MS,
                realized_pnl=10.0,
                accum_ms=self.CP_DURATION_MS,
                max_portfolio_value=1000.0,
                challenge_period_status=bucket_by_index.get(i, MinerBucket.PRO_FUNDED).value,
                weekly_penalty=0.0 if i in blocked_checkpoint_indices else 1.0,
            ))
        return DebtLedger(self.SUBACCOUNT_HOTKEY, checkpoints=checkpoints), week_0_start

    def _aggregate(self, blocked_checkpoint_indices=(), bucket_by_index=None):
        ledger, week_0_start = self._subaccount_ledger(blocked_checkpoint_indices, bucket_by_index)
        manager = self._build_manager(ledger)
        manager.aggregate_entity_debt_ledgers(self.CP_DURATION_MS)

        entity_ledger = manager.debt_ledgers[self.ENTITY_HOTKEY]
        per_week = [0.0, 0.0]
        for cp in entity_ledger.checkpoints:
            per_week[(cp.timestamp_ms - 1 - week_0_start) // MS_IN_WEEK] += cp.realized_pnl
        return per_week

    def test_unblocked_weeks_aggregate_fully(self):
        week_0, week_1 = self._aggregate()
        self.assertAlmostEqual(week_0, 140.0)
        self.assertAlmostEqual(week_1, 140.0)

    def test_a_breach_after_a_mid_week_promotion_keeps_the_pre_promotion_pnl_in_its_own_week(self):
        """The weight-setting half of the same rule: a breach cannot reach back past the promotion.

        The account promotes halfway through week 0 and breaches after. Week 0 still pays the
        70 USD it earned under the challenge rules; only the 70 earned under the pro rules is
        held, and week 1 - clean and still on track - settles it alongside its own 140.
        """
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        promoted_at = cps_per_week // 2
        week_0, week_1 = self._aggregate(
            blocked_checkpoint_indices=(promoted_at + 3,),
            bucket_by_index={
                i: MinerBucket.PRO_CHALLENGE_FROM_STANDARD for i in range(promoted_at)
            },
        )
        self.assertAlmostEqual(week_0, 70.0)
        self.assertAlmostEqual(week_1, 210.0)

    def test_the_pre_promotion_pnl_reaches_the_weight_setter_in_its_own_week(self):
        """The gate has to move weights, not just the payout report.

        WeightCalculatorManager._compute_miner_weights prices an entity week as the difference
        between calculate_payout_from_checkpoints run through this week and through the prior one,
        over exactly the aggregated ledger this test builds. Reproduced here: withholding the
        pre-promotion stretch would push its 70 USD out of week 0 and into week 1, moving the
        emissions target for both weeks.
        """
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        promoted_at = cps_per_week // 2
        ledger, week_0_start = self._subaccount_ledger(
            blocked_checkpoint_indices=(promoted_at + 3,),
            bucket_by_index={
                i: MinerBucket.PRO_CHALLENGE_FROM_STANDARD for i in range(promoted_at)
            },
        )
        manager = self._build_manager(ledger)
        manager.aggregate_entity_debt_ledgers(self.CP_DURATION_MS)
        entity_checkpoints = manager.debt_ledgers[self.ENTITY_HOTKEY].checkpoints

        def weekly_target(week_close_ms):
            """The weight setter's own arithmetic: this week's cumulative minus the prior one's."""
            through_this = DebtBasedScoring.calculate_payout_from_checkpoints(
                [cp for cp in entity_checkpoints if cp.timestamp_ms <= week_close_ms]
            )
            through_prior = DebtBasedScoring.calculate_payout_from_checkpoints(
                [cp for cp in entity_checkpoints if cp.timestamp_ms <= week_close_ms - MS_IN_WEEK]
            )
            return max(0.0, through_this - through_prior)

        # The Monday after the promotion week: the challenge half is the week's whole target
        self.assertAlmostEqual(weekly_target(week_0_start + MS_IN_WEEK), 70.0)
        # ...and the following Monday pays week 1 plus the escrow the breach deferred
        self.assertAlmostEqual(weekly_target(week_0_start + 2 * MS_IN_WEEK), 210.0)

    def test_a_settled_segment_reaches_the_entity_aggregate(self):
        """The weight-setting half: money settled at an account switch still reaches the entity
        aggregate, in the payout week it was earned in, even though the subaccount ledger it was
        computed from was deleted by that switch."""
        ledger, week_0_start = self._subaccount_ledger()
        manager = self._build_manager(ledger)
        manager.weekly_seal_ledger.record_settled(
            self.SUBACCOUNT_HOTKEY,
            week_start_ms=week_0_start,
            segment_start_ms=week_0_start,
            segment_end_ms=week_0_start + MS_IN_WEEK // 2,
            bucket=MinerBucket.PRO_CHALLENGE_TRANSITION.value,
            payout_usd=1000.0,
            gross_payout_usd=1000.0,
            weekly_penalty=1.0,
            payout_scale=1.0,
        )
        try:
            manager.aggregate_entity_debt_ledgers(self.CP_DURATION_MS)

            per_week = [0.0, 0.0]
            for cp in manager.debt_ledgers[self.ENTITY_HOTKEY].checkpoints:
                per_week[(cp.timestamp_ms - 1 - week_0_start) // MS_IN_WEEK] += cp.realized_pnl
            self.assertAlmostEqual(per_week[0], 140.0 + 1000.0)
            self.assertAlmostEqual(per_week[1], 140.0)
        finally:
            _clear_seal_ledger()


def _frozen_subaccount_manager(
    *,
    subaccount_status='eliminated',
    standard_account_size=100_000.0,
    pro_account_size=500_000.0,
    bucket_by_index=None,
    entity_hotkey='entity',
    subaccount_hotkey='entity_1',
):
    """A manager whose only subaccount reaches the aggregation through a frozen ledger.

    Two closed payout weeks of 12h checkpoints realizing 10 USD each. `bucket_by_index` overrides
    the bucket stamped on individual checkpoints; the rest are PRO_CHALLENGE_FROM_STANDARD, the
    one bucket whose payouts are scaled by the standard/pro ratio.
    """
    cp_duration = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
    week_0_start = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK
    bucket_by_index = bucket_by_index or {}
    timestamps = [
        week_0_start + (i + 1) * cp_duration
        for i in range(2 * MS_IN_WEEK // cp_duration)
    ]

    perf_ledger = SimpleNamespace(cps=[
        SimpleNamespace(
            last_update_ms=ts, accum_ms=cp_duration, realized_pnl=10.0, cumulative_fees_usd=0.0,
            mpv=1000.0, mdd=1.0, gain=0.0, open_ms=cp_duration, n_updates=1,
        )
        for ts in timestamps
    ])
    # The frozen ledger takes its buckets from the penalty ledger, which outlives elimination
    penalty_cps = {
        ts: SimpleNamespace(
            timestamp_ms=ts, weekly_penalty=1.0,
            challenge_period_status=bucket_by_index.get(
                i, MinerBucket.PRO_CHALLENGE_FROM_STANDARD
            ).value,
        )
        for i, ts in enumerate(timestamps)
    }
    penalty_ledger = SimpleNamespace(
        checkpoints=list(penalty_cps.values()),
        get_checkpoint_at_time=lambda ts, _duration: penalty_cps.get(ts),
    )

    manager = object.__new__(DebtLedgerManager)
    # Keeps the lazily built weekly seal ledger inside the test validation directory
    manager.running_unit_tests = True
    manager.debt_ledgers = {}
    manager.emissions_ledger_manager = SimpleNamespace(get_ledger=lambda _hotkey: None)
    manager.penalty_ledger_manager = SimpleNamespace(get_penalty_ledger=lambda _hotkey: penalty_ledger)
    manager._entity_client = SimpleNamespace(get_all_entities=lambda: {
        entity_hotkey: {'subaccounts': {'1': {
            'status': subaccount_status,
            'synthetic_hotkey': subaccount_hotkey,
            'reg_fee_theta': 1.0,
            'standard_account_size': standard_account_size,
            'pro_account_size': pro_account_size,
        }}}
    })
    manager._perf_ledger_client = SimpleNamespace(
        get_frozen_ledgers=lambda: {subaccount_hotkey: perf_ledger}
    )
    manager._challengeperiod_client = SimpleNamespace(
        get_miner_bucket=lambda _hotkey: MinerBucket.ENTITY
    )
    return manager, week_0_start


class TestEliminatedSubaccountPayoutScale(TestBase):
    """An eliminated subaccount's closed weeks are sealed at its real payout scale.

    An eliminated subaccount still reaches the aggregation through its frozen ledger, and its
    closed weeks are sealed there. Sealing them at the default 1.0 would pin the wrong scale into
    settled history, and revert-elimination could not undo it: `seal` is write-once and a later
    build computing the right scale would only log the disagreement.
    """

    ENTITY_HOTKEY = "entity"
    SUBACCOUNT_HOTKEY = "entity_1"
    CP_DURATION_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
    STANDARD_ACCOUNT_SIZE = 100_000.0
    PRO_ACCOUNT_SIZE = 500_000.0

    @property
    def expected_scale(self):
        return (ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER
                * self.STANDARD_ACCOUNT_SIZE / self.PRO_ACCOUNT_SIZE)

    def setUp(self):
        super().setUp()
        _clear_seal_ledger()

    def tearDown(self):
        _clear_seal_ledger()
        super().tearDown()

    def _aggregate(self, subaccount_status):
        manager, week_0_start = _frozen_subaccount_manager(
            subaccount_status=subaccount_status,
            standard_account_size=self.STANDARD_ACCOUNT_SIZE,
            pro_account_size=self.PRO_ACCOUNT_SIZE,
            entity_hotkey=self.ENTITY_HOTKEY,
            subaccount_hotkey=self.SUBACCOUNT_HOTKEY,
        )
        manager.aggregate_entity_debt_ledgers(self.CP_DURATION_MS)

        sealed = manager.weekly_seal_ledger.get_sealed(self.SUBACCOUNT_HOTKEY)
        realized = sum(cp.realized_pnl for cp in manager.debt_ledgers[self.ENTITY_HOTKEY].checkpoints)
        return sealed, realized, week_0_start

    def test_eliminated_subaccount_seals_its_real_payout_scale(self):
        sealed, _realized, week_0_start = self._aggregate('eliminated')
        self.assertEqual(sorted(sealed), [week_0_start, week_0_start + MS_IN_WEEK])
        for week in sealed.values():
            self.assertAlmostEqual(week.payout_scale, self.expected_scale)

    def test_elimination_does_not_change_what_gets_sealed(self):
        """Revert-elimination replays the sealed weeks, so they must match the active verdict."""
        active_sealed, active_realized, _ = self._aggregate('active')
        _clear_seal_ledger()
        eliminated_sealed, eliminated_realized, _ = self._aggregate('eliminated')

        self.assertEqual(
            {ms: week.payout_scale for ms, week in eliminated_sealed.items()},
            {ms: week.payout_scale for ms, week in active_sealed.items()},
        )
        self.assertAlmostEqual(eliminated_realized, active_realized)

    def test_the_sealed_scale_is_what_the_entity_is_paid_on(self):
        _sealed, realized, _ = self._aggregate('eliminated')
        # 28 checkpoints realizing 10 USD each, paid on the standard account's basis
        self.assertAlmostEqual(realized, 280.0 * self.expected_scale)


class TestSealedScaleGovernsPayment(TestBase):
    """The ratio a week settled at is the ratio it keeps being paid at.

    `get_payout_scale` reads the subaccount's sizes live, so without the seal, resizing a pro
    account silently reprices every week it ever traded.
    """

    ENTITY_HOTKEY = "entity"
    SUBACCOUNT_HOTKEY = "entity_1"
    CP_DURATION_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
    STANDARD = 100_000.0
    PRO = 500_000.0
    # PRO_TRANSITION_PAYOUT_MULTIPLIER * 100k / 500k
    RATIO = ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER * 0.2
    # ... and the ratio after the pro account is doubled to 1M
    RESIZED_RATIO = ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER * 0.1

    def setUp(self):
        super().setUp()
        _clear_seal_ledger()

    def tearDown(self):
        _clear_seal_ledger()
        super().tearDown()

    def _aggregate(self, pro_account_size, bucket_by_index=None):
        manager, week_0_start = _frozen_subaccount_manager(
            standard_account_size=self.STANDARD,
            pro_account_size=pro_account_size,
            bucket_by_index=bucket_by_index,
            entity_hotkey=self.ENTITY_HOTKEY,
            subaccount_hotkey=self.SUBACCOUNT_HOTKEY,
        )
        manager.aggregate_entity_debt_ledgers(self.CP_DURATION_MS)
        sealed = manager.weekly_seal_ledger.get_sealed(self.SUBACCOUNT_HOTKEY)
        realized = sum(cp.realized_pnl for cp in manager.debt_ledgers[self.ENTITY_HOTKEY].checkpoints)
        return sealed, realized, week_0_start

    def test_a_resize_cannot_reprice_a_sealed_week(self):
        _sealed, first_realized, week_0_start = self._aggregate(self.PRO)
        self.assertAlmostEqual(first_realized, 280.0 * self.RATIO)

        # The pro account is doubled, halving the live ratio. Both weeks already settled.
        sealed, realized, _ = self._aggregate(2 * self.PRO)
        self.assertAlmostEqual(realized, 280.0 * self.RATIO)
        self.assertAlmostEqual(sealed[week_0_start].payout_scale, self.RATIO)

    def test_the_same_resize_does_reprice_a_week_that_was_never_sealed(self):
        """The seal is what holds the ratio - without it the rebuild follows the new sizes."""
        _sealed, realized, _ = self._aggregate(2 * self.PRO)
        self.assertAlmostEqual(realized, 280.0 * self.RESIZED_RATIO)

    def test_a_mid_week_promotion_is_still_priced_per_checkpoint(self):
        """A week-level seal must not flatten the per-checkpoint bucket gate.

        The account promotes halfway through week 0: the first half is paid on the standard
        basis, everything after it at full pro scale.
        """
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        promoted = {i: MinerBucket.PRO_FUNDED for i in range(cps_per_week // 2, 2 * cps_per_week)}
        sealed, realized, week_0_start = self._aggregate(self.PRO, bucket_by_index=promoted)

        scaled_cps, funded_cps = cps_per_week // 2, 2 * cps_per_week - cps_per_week // 2
        self.assertAlmostEqual(realized, 10.0 * scaled_cps * self.RATIO + 10.0 * funded_cps)
        # The week still seals the account's ratio, ungated: the gate is the checkpoint's bucket
        self.assertAlmostEqual(sealed[week_0_start].payout_scale, self.RATIO)


class TestWeeklySealLedger(TestBase):
    """A closed payout week is settled money: rebuilding the ledgers must not move it between
    paid and withheld, and deleting the ledgers must not lose the record that it was settled."""

    HOTKEY = "entity_1"

    def setUp(self):
        super().setUp()
        _clear_seal_ledger()
        self.week_start_ms = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - MS_IN_WEEK
        self.ledger = WeeklySealLedger(running_unit_tests=True)

    def tearDown(self):
        _clear_seal_ledger()
        super().tearDown()

    def _seal(self, ledger, weekly_penalty=1.0, payout_scale=1.0, track='ON_TRACK'):
        return ledger.seal(
            self.HOTKEY, self.week_start_ms,
            weekly_penalty=weekly_penalty, payout_scale=payout_scale, track=track,
        )

    def test_first_seal_records_the_week(self):
        self.assertTrue(self._seal(self.ledger, weekly_penalty=0.0))
        sealed = self.ledger.get_sealed(self.HOTKEY)[self.week_start_ms]
        self.assertEqual(sealed.weekly_penalty, 0.0)
        self.assertEqual(sealed.track, 'ON_TRACK')

    def test_a_disagreeing_rebuild_never_overwrites_the_sealed_week(self):
        """The whole point: a later build computing a different verdict keeps the settled one."""
        self._seal(self.ledger, weekly_penalty=0.0)
        # A rebuild against a worse ratcheted drawdown / a resized account now says "clean"
        self.assertFalse(self._seal(self.ledger, weekly_penalty=1.0, payout_scale=0.2))
        sealed = self.ledger.get_sealed(self.HOTKEY)[self.week_start_ms]
        self.assertEqual(sealed.weekly_penalty, 0.0)
        self.assertEqual(sealed.payout_scale, 1.0)

    def test_sealed_weeks_survive_losing_every_other_ledger(self):
        """delete_debt_ledger drops the debt and penalty ledgers; the seals outlive that."""
        self._seal(self.ledger, weekly_penalty=0.0)
        self.ledger.save_to_disk()

        reloaded = WeeklySealLedger(running_unit_tests=True)
        self.assertEqual(reloaded.get_sealed(self.HOTKEY)[self.week_start_ms].weekly_penalty, 0.0)

    def test_unseal_allows_a_deliberate_correction(self):
        self._seal(self.ledger, weekly_penalty=0.0)
        self.assertTrue(self.ledger.unseal(self.HOTKEY, self.week_start_ms))
        self.assertEqual(self.ledger.get_sealed(self.HOTKEY), {})
        # Now a rebuild is free to record a new verdict
        self.assertTrue(self._seal(self.ledger, weekly_penalty=1.0))

    def test_unseal_of_an_unknown_week_is_a_no_op(self):
        self.assertFalse(self.ledger.unseal(self.HOTKEY, self.week_start_ms))

    def test_weekly_payout_context_replays_the_sealed_verdict(self):
        """A breached ledger still reads as paid when the week was sealed clean, and vice versa."""
        cp_duration = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
        checkpoints = [
            DebtCheckpoint(
                timestamp_ms=self.week_start_ms + (i + 1) * cp_duration,
                challenge_period_status=MinerBucket.PRO_FUNDED.value,
                weekly_penalty=0.0,
            )
            for i in range(MS_IN_WEEK // cp_duration)
        ]
        debt_ledger = DebtLedger(self.HOTKEY, checkpoints=checkpoints)

        recomputed = debt_ledger.weekly_payout_context()
        self.assertEqual(recomputed[self.week_start_ms].weekly_penalty, 0.0)

        sealed = {self.week_start_ms: SealedWeek(
            week_start_ms=self.week_start_ms, weekly_penalty=1.0, payout_scale=1.0,
            track='ON_TRACK', first_earning_ms=None, sealed_ms=0,
        )}
        replayed = debt_ledger.weekly_payout_context(sealed=sealed)
        self.assertEqual(replayed[self.week_start_ms].weekly_penalty, 1.0)
        self.assertIs(replayed[self.week_start_ms].track, WeekTrack.ON_TRACK)


class TestDebtLedgerBucketHelpers(TestBase):
    """The payout paths read each checkpoint's own bucket instead of wiping challenge history."""

    HOTKEY = "entity_1"
    CP_DURATION_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    def _ledger(self, statuses, fees=None):
        fees = fees or [0.0] * len(statuses)
        base_ms = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - MS_IN_WEEK
        return DebtLedger(self.HOTKEY, checkpoints=[
            DebtCheckpoint(
                timestamp_ms=base_ms + (i + 1) * self.CP_DURATION_MS,
                challenge_period_status=status.value,
                cumulative_fees_usd=fee,
            )
            for i, (status, fee) in enumerate(zip(statuses, fees))
        ]), base_ms

    def test_first_earning_checkpoint_skips_the_challenge(self):
        ledger, base_ms = self._ledger([
            MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_CHALLENGE_DIRECT,
            MinerBucket.PRO_FUNDED,
        ])
        self.assertEqual(ledger.first_earning_checkpoint_ms(), base_ms + 3 * self.CP_DURATION_MS)

    def test_fee_baseline_excludes_fees_paid_during_the_challenge(self):
        """Fees run cumulatively from inception while realized PnL restarts at the first earning
        checkpoint, so the challenge's fees must not be charged against funded earnings."""
        ledger, _ = self._ledger(
            [MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_CHALLENGE_DIRECT, MinerBucket.PRO_FUNDED],
            fees=[5.0, 12.0, 20.0],
        )
        # The baseline is the checkpoint *before* the first earning one: fees inside the first
        # earning window still count against it
        self.assertEqual(ledger.fee_baseline_at_first_earning(), 12.0)

    def test_fee_baseline_is_zero_when_the_account_earned_from_the_start(self):
        ledger, _ = self._ledger([MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.SUBACCOUNT_FUNDED],
                                 fees=[3.0, 7.0])
        self.assertEqual(ledger.fee_baseline_at_first_earning(), 0.0)

    def test_bucket_change_times_marks_the_promotion_boundary(self):
        ledger, base_ms = self._ledger([
            MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_FUNDED, MinerBucket.PRO_FUNDED,
        ])
        # The switch happened at the end of the last challenge window
        self.assertEqual(ledger.bucket_change_times(), [base_ms + self.CP_DURATION_MS])

    def test_bucket_change_times_is_empty_for_a_stable_account(self):
        ledger, _ = self._ledger([MinerBucket.SUBACCOUNT_FUNDED] * 4)
        self.assertEqual(ledger.bucket_change_times(), [])

    def test_checkpoint_weekly_penalty_only_bites_where_the_soft_breach_rule_applies(self):
        """The week hands over one penalty; the checkpoint's own bucket decides if it applies.

        Without this, a breach committed after a mid-week promotion would withhold the earnings
        the subaccount made before it, under rules that did not govern it at the time.
        """
        ledger, _ = self._ledger([
            MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_FUNDED,
        ])
        challenge_cp, funded_cp = ledger.checkpoints
        self.assertEqual(DebtLedger.checkpoint_weekly_penalty(challenge_cp, 0.0), 1.0)
        self.assertEqual(DebtLedger.checkpoint_weekly_penalty(funded_cp, 0.0), 0.0)
        # A clean week passes through untouched either way
        self.assertEqual(DebtLedger.checkpoint_weekly_penalty(challenge_cp, 1.0), 1.0)
        self.assertEqual(DebtLedger.checkpoint_weekly_penalty(funded_cp, 1.0), 1.0)

    def test_bucket_at_reads_the_segment_that_starts_on_a_boundary(self):
        ledger, base_ms = self._ledger([
            MinerBucket.PRO_CHALLENGE_FROM_STANDARD, MinerBucket.PRO_FUNDED,
        ])
        boundary_ms = base_ms + self.CP_DURATION_MS
        # An event exactly on the boundary belongs to the window that just closed...
        self.assertIs(ledger.bucket_at(boundary_ms), MinerBucket.PRO_CHALLENGE_FROM_STANDARD)
        # ...while the segment starting there is governed by the next one
        self.assertIs(ledger.bucket_at(boundary_ms + 1), MinerBucket.PRO_FUNDED)


class TestSettledSegment(TestBase):
    """A subaccount promoted mid-week traded part of that week on the account the promotion wipes.

    The switch archives its positions and deletes its perf, debt and penalty ledgers, so nothing
    can recompute what it earned. The settled segment is the only surviving record of that money,
    which makes surviving a rebuild the whole point of it.
    """

    HOTKEY = "settledsub_1"

    def setUp(self):
        super().setUp()
        _clear_seal_ledger()
        self.week_start_ms = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis())
        self.promotion_ms = self.week_start_ms + MS_IN_WEEK // 2
        self.ledger = WeeklySealLedger(running_unit_tests=True)

    def tearDown(self):
        _clear_seal_ledger()
        super().tearDown()

    def _settle(self, ledger, payout_usd=1843.20):
        return ledger.record_settled(
            self.HOTKEY,
            week_start_ms=self.week_start_ms,
            segment_start_ms=self.week_start_ms,
            segment_end_ms=self.promotion_ms,
            bucket=MinerBucket.PRO_CHALLENGE_TRANSITION.value,
            payout_usd=payout_usd,
            gross_payout_usd=payout_usd,
            weekly_penalty=1.0,
            payout_scale=1.0,
        )

    def test_a_settled_segment_records_the_wound_down_week(self):
        self.assertTrue(self._settle(self.ledger))
        segment = self.ledger.get_settled(self.HOTKEY)[0]
        self.assertEqual(segment.payout_usd, 1843.20)
        self.assertEqual(segment.bucket, MinerBucket.PRO_CHALLENGE_TRANSITION.value)
        self.assertEqual(segment.segment_end_ms, self.promotion_ms)

    def test_resettling_the_same_switch_never_double_pays(self):
        """A retried promotion must not settle the same stretch twice."""
        self._settle(self.ledger, payout_usd=1843.20)
        self.assertFalse(self._settle(self.ledger, payout_usd=9999.99))
        segments = self.ledger.get_settled(self.HOTKEY)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].payout_usd, 1843.20)

    def test_a_retried_switch_settles_the_week_once(self):
        """A switch that fails after settling is retried with a fresh timestamp. The week and
        bucket are already settled, so the retry must not file the same dollars again."""
        self._settle(self.ledger)
        self.promotion_ms += 30_000
        self.assertFalse(self._settle(self.ledger))
        segments = self.ledger.get_settled(self.HOTKEY)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].payout_usd, 1843.20)

    def test_settled_segments_survive_a_reload(self):
        """The rebuild case: every other ledger is regenerated, this one is replayed from disk."""
        self._settle(self.ledger)
        reloaded = WeeklySealLedger(running_unit_tests=True)
        self.assertEqual(reloaded.get_settled(self.HOTKEY)[0].payout_usd, 1843.20)

    def test_a_seal_file_written_before_settled_segments_still_loads(self):
        """Format 1.0 has no `settled` key; it must load as a ledger with no segments."""
        import gzip
        import json
        path = self.ledger._get_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump({
                "format_version": "1.0",
                "last_update_ms": TimeUtil.now_in_millis(),
                "sealed": {self.HOTKEY: {str(self.week_start_ms): SealedWeek(
                    week_start_ms=self.week_start_ms, weekly_penalty=1.0, payout_scale=1.0,
                    track='ON_TRACK', first_earning_ms=None, sealed_ms=0,
                ).to_dict()}},
            }, f)

        reloaded = WeeklySealLedger(running_unit_tests=True)
        self.assertEqual(reloaded.get_settled(self.HOTKEY), [])
        self.assertEqual(reloaded.get_sealed(self.HOTKEY)[self.week_start_ms].weekly_penalty, 1.0)
        # A segment recorded onto it upgrades the file rather than tripping over the old shape
        self.assertTrue(self._settle(reloaded))
        self.assertEqual(
            WeeklySealLedger(running_unit_tests=True).get_settled(self.HOTKEY)[0].payout_usd, 1843.20
        )

    def test_amend_and_remove_allow_a_deliberate_correction(self):
        """Nothing recomputes these records, so the tier-500 door is the only way to fix one."""
        self._settle(self.ledger)
        self.assertTrue(self.ledger.amend_settled(self.HOTKEY, self.promotion_ms, 500.0))
        amended = self.ledger.get_settled(self.HOTKEY)[0]
        self.assertEqual(amended.payout_usd, 500.0)
        self.assertIsNotNone(amended.amended_ms)
        # The correction is on disk, not just in memory
        self.assertEqual(
            WeeklySealLedger(running_unit_tests=True).get_settled(self.HOTKEY)[0].payout_usd, 500.0
        )

        self.assertTrue(self.ledger.remove_settled(self.HOTKEY, self.promotion_ms))
        self.assertEqual(self.ledger.get_settled(self.HOTKEY), [])
        self.assertFalse(self.ledger.remove_settled(self.HOTKEY, self.promotion_ms))

    def test_a_segment_round_trips_through_serialization(self):
        segment = SettledSegment(
            week_start_ms=self.week_start_ms,
            segment_start_ms=self.week_start_ms,
            segment_end_ms=self.promotion_ms,
            bucket=MinerBucket.PRO_CHALLENGE_TRANSITION.value,
            payout_usd=1843.20,
            gross_payout_usd=1843.20,
            weekly_penalty=1.0,
            payout_scale=1.0,
            recorded_ms=123,
        )
        self.assertEqual(SettledSegment.from_dict(segment.to_dict()), segment)
