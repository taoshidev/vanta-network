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
from vali_objects.enums.miner_bucket_enum import BucketPhase, MinerBucket
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger import DebtCheckpoint, DebtLedger, WeekTrack, apply_deferral
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_manager import DebtLedgerManager
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
import logging
import os
from shared_objects.log import logger

logger.setLevel(logging.INFO)


# Ledger status of a checkpoint stamped during the SUBACCOUNT_FUNDED transition week
TRANSITION_STATUS = BucketPhase(MinerBucket.SUBACCOUNT_FUNDED, in_transition=True).status


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

class TestRetiredWeeklySealLedger(TestBase):

    def test_the_retired_file_is_removed_on_startup(self):
        path = (ValiConfig.BASE_DIR + ValiBkpUtils.get_test_dir_suffix(True)
                + "/validation/weekly_seal_ledger.json.gz")
        ValiBkpUtils.write_compressed_json(path, {"sealed": {}, "settled": {}})
        manager = object.__new__(DebtLedgerManager)
        manager.running_unit_tests = True

        manager._remove_retired_weekly_seal_ledger()

        self.assertFalse(os.path.exists(path))


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

    def _build_manager(self, subaccount_ledger, segments=None):
        """Bare manager with only the collaborators aggregate_entity_debt_ledgers touches.
        `segments` are the subaccount's bucket segments (none: one account in one bucket)."""
        manager = object.__new__(DebtLedgerManager)
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
            get_miner_bucket=lambda _hotkey: MinerBucket.ENTITY,
            get_bucket_history_checkpoint_dict=lambda: {self.SUBACCOUNT_HOTKEY: segments or []},
        )
        return manager

    def _subaccount_ledger(self, blocked_checkpoint_indices=(), bucket_by_index=None, fees_by_index=None):
        """One earning checkpoint per 12h for two weeks, each realizing 10 USD.

        `bucket_by_index` overrides the bucket stamped on individual checkpoints, which is how a
        promotion lands mid-week; everything else is PRO_FUNDED.
        """
        week_0_start = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - 2 * MS_IN_WEEK
        bucket_by_index = bucket_by_index or {}
        fees_by_index = fees_by_index or {}
        checkpoints = []
        for i in range(2 * MS_IN_WEEK // self.CP_DURATION_MS):
            checkpoints.append(DebtCheckpoint(
                timestamp_ms=week_0_start + (i + 1) * self.CP_DURATION_MS,
                realized_pnl=10.0,
                cumulative_fees_usd=fees_by_index.get(i, 0.0),
                accum_ms=self.CP_DURATION_MS,
                max_portfolio_value=1000.0,
                challenge_period_status=bucket_by_index.get(i, MinerBucket.PRO_FUNDED).value,
                weekly_penalty=0.0 if i in blocked_checkpoint_indices else 1.0,
            ))
        return DebtLedger(self.SUBACCOUNT_HOTKEY, checkpoints=checkpoints), week_0_start

    def _aggregate(self, blocked_checkpoint_indices=(), bucket_by_index=None, segments=None,
                   fees_by_index=None, ledger_from_index=0):
        ledger, week_0_start = self._subaccount_ledger(blocked_checkpoint_indices, bucket_by_index, fees_by_index)
        # A ledger starting later is what an account switch leaves behind
        ledger.checkpoints = ledger.checkpoints[ledger_from_index:]
        manager = self._build_manager(ledger, segments(week_0_start) if segments else None)
        manager.aggregate_entity_debt_ledgers(self.CP_DURATION_MS)

        entity_ledger = manager.debt_ledgers[self.ENTITY_HOTKEY]
        per_week = [0.0, 0.0]
        for cp in entity_ledger.checkpoints:
            per_week[(cp.timestamp_ms - 1 - week_0_start) // MS_IN_WEEK] += cp.realized_pnl
        return per_week

    @staticmethod
    def _segment(bucket, start_ms, end_ms=None, record_rows=()):
        """A bucket segment summary; record_rows are (end_ms, payout) of its payout record."""
        record = None
        if record_rows:
            record = {'end_ms': end_ms, 'payout_usd': sum(p for _, p in record_rows),
                      'settlements': [{'end_ms': e, 'payout': p} for e, p in record_rows]}
        return {'bucket': bucket.value, 'start_ms': start_ms, 'end_ms': end_ms, 'reverted_ms': None,
                'payout_scale': None, 'payout_record': record}

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
                i: MinerBucket.PRO_CHALLENGE_FUNDED for i in range(promoted_at)
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
                i: MinerBucket.PRO_CHALLENGE_FUNDED for i in range(promoted_at)
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

    def test_a_payout_record_replaces_the_checkpoints_it_covers(self):
        """PRO_CHALLENGE_FUNDED -> PRO_FUNDED keeps the ledger. The checkpoints of the closed
        segment are paid from its record, in the week the record row ends, and never again;
        PRO_FUNDED is paid from its own start."""
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        promoted_at = cps_per_week // 2
        week_0, week_1 = self._aggregate(
            bucket_by_index={i: MinerBucket.PRO_CHALLENGE_FUNDED for i in range(promoted_at)},
            segments=lambda w0: [
                self._segment(MinerBucket.PRO_CHALLENGE_FUNDED, w0 - MS_IN_WEEK, w0 + promoted_at * self.CP_DURATION_MS,
                              record_rows=[(w0, 500.0), (w0 + promoted_at * self.CP_DURATION_MS, 999.0)]),
                self._segment(MinerBucket.PRO_FUNDED, w0 + promoted_at * self.CP_DURATION_MS),
            ],
        )
        # The record row ending at week 0's start belongs to the week before and is not paid again
        self.assertAlmostEqual(week_0, 999.0 + 70.0)
        self.assertAlmostEqual(week_1, 140.0)

    def test_a_record_reaches_the_aggregate_after_its_ledger_was_deleted(self):
        """An account switch deletes the ledger the record was computed from."""
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        switch_at = cps_per_week // 2
        week_0, week_1 = self._aggregate(
            bucket_by_index={i: MinerBucket.PRO_CHALLENGE_FUNDED for i in range(2 * cps_per_week)},
            ledger_from_index=switch_at,
            segments=lambda w0: [
                self._segment(MinerBucket.SUBACCOUNT_FUNDED, w0 - MS_IN_WEEK, w0 + switch_at * self.CP_DURATION_MS,
                              record_rows=[(w0 + switch_at * self.CP_DURATION_MS, 1000.0)]),
                self._segment(MinerBucket.PRO_CHALLENGE_FUNDED, w0 + switch_at * self.CP_DURATION_MS),
            ],
        )
        self.assertAlmostEqual(week_0, 1000.0 + 70.0)
        self.assertAlmostEqual(week_1, 140.0)

    def test_a_new_segment_is_not_charged_the_fees_of_the_one_before(self):
        """cumulative_fees_usd runs from ledger inception; PRO_FUNDED starts from its own baseline."""
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        week_0, week_1 = self._aggregate(
            bucket_by_index={i: MinerBucket.PRO_CHALLENGE_DIRECT for i in range(cps_per_week)},
            fees_by_index={i: 50.0 for i in range(2 * cps_per_week)},
            segments=lambda w0: [
                self._segment(MinerBucket.PRO_CHALLENGE_DIRECT, w0, w0 + MS_IN_WEEK),
                self._segment(MinerBucket.PRO_FUNDED, w0 + MS_IN_WEEK),
            ],
        )
        self.assertAlmostEqual(week_0, 0.0)
        self.assertAlmostEqual(week_1, 140.0)


def _frozen_subaccount_manager(
    *,
    subaccount_status='eliminated',
    standard_account_size=100_000.0,
    pro_account_size=500_000.0,
    payout_scale=None,
    bucket_by_index=None,
    entity_hotkey='entity',
    subaccount_hotkey='entity_1',
    segment_payout_scale=None,
):
    """A manager whose only subaccount reaches the aggregation through a frozen ledger.

    Two closed payout weeks of 12h checkpoints realizing 10 USD each. `bucket_by_index` overrides
    the bucket stamped on individual checkpoints; the rest are PRO_CHALLENGE_FUNDED (out of
    transition), the one bucket whose payouts are scaled by the standard/pro ratio.
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
                i, MinerBucket.PRO_CHALLENGE_FUNDED
            ).value,
        )
        for i, ts in enumerate(timestamps)
    }
    penalty_ledger = SimpleNamespace(
        checkpoints=list(penalty_cps.values()),
        get_checkpoint_at_time=lambda ts, _duration: penalty_cps.get(ts),
    )

    manager = object.__new__(DebtLedgerManager)
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
            'payout_scale': payout_scale,
        }}}
    })
    manager._perf_ledger_client = SimpleNamespace(
        get_frozen_ledgers=lambda: {subaccount_hotkey: perf_ledger}
    )
    # One live segment, opened at segment_payout_scale (None: the live ratio applies)
    segments = [{'bucket': MinerBucket.PRO_CHALLENGE_FUNDED.value, 'start_ms': week_0_start, 'end_ms': None,
                 'reverted_ms': None, 'payout_scale': segment_payout_scale, 'payout_record': None}]
    manager._challengeperiod_client = SimpleNamespace(
        get_miner_bucket=lambda _hotkey: MinerBucket.ENTITY,
        get_bucket_history_checkpoint_dict=lambda: {subaccount_hotkey: segments},
    )
    return manager, week_0_start


class TestEliminatedSubaccountPayoutScale(TestBase):
    """An eliminated subaccount still reaches the aggregation through its frozen ledger, at its
    real payout scale rather than the default 1.0."""

    CP_DURATION_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
    SCALE = ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER * 100_000.0 / 500_000.0

    def _realized(self, subaccount_status):
        manager, _ = _frozen_subaccount_manager(subaccount_status=subaccount_status)
        manager.aggregate_entity_debt_ledgers(self.CP_DURATION_MS)
        return sum(cp.realized_pnl for cp in manager.debt_ledgers['entity'].checkpoints)

    def test_the_eliminated_subaccount_is_paid_on_its_real_scale(self):
        # 28 checkpoints realizing 10 USD each, paid on the standard account's basis
        self.assertAlmostEqual(self._realized('eliminated'), 280.0 * self.SCALE)

    def test_elimination_does_not_change_the_scale(self):
        self.assertAlmostEqual(self._realized('eliminated'), self._realized('active'))


class TestSegmentScaleGovernsPayment(TestBase):
    """The ratio a segment opened with is the ratio it is paid at, so resizing the pro account
    cannot reprice it."""

    CP_DURATION_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
    RATIO = ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER * 0.2
    RESIZED_RATIO = ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER * 0.1

    def _realized(self, pro_account_size, segment_payout_scale=None, bucket_by_index=None):
        manager, _ = _frozen_subaccount_manager(pro_account_size=pro_account_size,
                                                segment_payout_scale=segment_payout_scale,
                                                bucket_by_index=bucket_by_index)
        manager.aggregate_entity_debt_ledgers(self.CP_DURATION_MS)
        return sum(cp.realized_pnl for cp in manager.debt_ledgers['entity'].checkpoints)

    def test_a_resize_cannot_reprice_the_segment(self):
        self.assertAlmostEqual(self._realized(1_000_000.0, segment_payout_scale=self.RATIO), 280.0 * self.RATIO)

    def test_without_a_captured_scale_the_live_ratio_applies(self):
        self.assertAlmostEqual(self._realized(1_000_000.0), 280.0 * self.RESIZED_RATIO)

    def test_a_mid_week_promotion_is_still_priced_per_checkpoint(self):
        cps_per_week = MS_IN_WEEK // self.CP_DURATION_MS
        promoted = {i: MinerBucket.PRO_FUNDED for i in range(cps_per_week // 2, 2 * cps_per_week)}
        scaled_cps, funded_cps = cps_per_week // 2, 2 * cps_per_week - cps_per_week // 2
        self.assertAlmostEqual(self._realized(500_000.0, bucket_by_index=promoted),
                               10.0 * scaled_cps * self.RATIO + 10.0 * funded_cps)


class TestSubaccountPayoutScaleInWeights(TestBase):
    """A payout_scale chosen at creation replaces PRO_TRANSITION_PAYOUT_MULTIPLIER in the entity's
    aggregated ledger, which the weight calculator reads."""

    CP_DURATION_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS

    def _realized(self, payout_scale, bucket_by_index=None):
        manager, _ = _frozen_subaccount_manager(
            subaccount_status='active', payout_scale=payout_scale, bucket_by_index=bucket_by_index)
        manager.aggregate_entity_debt_ledgers(self.CP_DURATION_MS)
        return sum(cp.realized_pnl for cp in manager.debt_ledgers['entity'].checkpoints)

    def test_the_chosen_payout_scale_prices_the_entity_ledger(self):
        # 28 checkpoints realizing 10 USD each at 1.0 * 100k / 500k
        self.assertAlmostEqual(self._realized(1.0), 280.0 * 0.2)

    def test_no_chosen_payout_scale_keeps_the_network_multiplier(self):
        self.assertAlmostEqual(self._realized(None), 280.0 * ValiConfig.PRO_TRANSITION_PAYOUT_MULTIPLIER * 0.2)

    def test_the_chosen_payout_scale_only_applies_in_pro_challenge_funded(self):
        funded = {i: MinerBucket.PRO_FUNDED for i in range(2 * MS_IN_WEEK // self.CP_DURATION_MS)}
        self.assertAlmostEqual(self._realized(1.0, bucket_by_index=funded), 280.0)


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

    def test_checkpoint_weekly_penalty_only_bites_where_the_soft_breach_rule_applies(self):
        """The week hands over one penalty; the checkpoint's own bucket decides if it applies.

        Without this, a breach committed after a mid-week promotion would withhold the earnings
        the subaccount made before it, under rules that did not govern it at the time.
        """
        ledger, _ = self._ledger([
            MinerBucket.PRO_CHALLENGE_FUNDED, MinerBucket.PRO_FUNDED,
        ])
        challenge_cp, funded_cp = ledger.checkpoints
        self.assertEqual(DebtLedger.checkpoint_weekly_penalty(challenge_cp, 0.0), 1.0)
        self.assertEqual(DebtLedger.checkpoint_weekly_penalty(funded_cp, 0.0), 0.0)
        # A clean week passes through untouched either way
        self.assertEqual(DebtLedger.checkpoint_weekly_penalty(challenge_cp, 1.0), 1.0)
        self.assertEqual(DebtLedger.checkpoint_weekly_penalty(funded_cp, 1.0), 1.0)

    def _status_ledger(self, statuses):
        base_ms = TimeUtil.ms_at_start_of_week(TimeUtil.now_in_millis()) - MS_IN_WEEK
        return DebtLedger(self.HOTKEY, checkpoints=[
            DebtCheckpoint(timestamp_ms=base_ms + (i + 1) * self.CP_DURATION_MS, challenge_period_status=status)
            for i, status in enumerate(statuses)
        ]), base_ms

    def test_the_transition_is_paid_unscaled_and_the_pro_phase_scaled(self):
        ledger, _ = self._status_ledger([
            TRANSITION_STATUS, TRANSITION_STATUS, MinerBucket.PRO_CHALLENGE_FUNDED.value,
        ])
        transition_cp, _, pro_cp = ledger.checkpoints
        self.assertEqual(DebtLedger.checkpoint_payout_scale(transition_cp, 0.4), 1.0)
        self.assertEqual(DebtLedger.checkpoint_payout_scale(pro_cp, 0.4), 0.4)

    def test_checkpoints_stamped_with_the_replaced_names_keep_their_phase(self):
        """Ledgers built before the merge carry PRO_CHALLENGE_TRANSITION / PRO_CHALLENGE_FROM_STANDARD."""
        ledger, _ = self._status_ledger(["PRO_CHALLENGE_TRANSITION", "PRO_CHALLENGE_FROM_STANDARD"])
        transition_cp, pro_cp = ledger.checkpoints
        self.assertEqual(ledger.checkpoint_bucket(transition_cp), BucketPhase.from_status(TRANSITION_STATUS))
        self.assertEqual(DebtLedger.checkpoint_payout_scale(transition_cp, 0.4), 1.0)
        self.assertEqual(DebtLedger.checkpoint_payout_scale(pro_cp, 0.4), 0.4)
        for status in (TRANSITION_STATUS, "PRO_CHALLENGE_TRANSITION", "PRO_CHALLENGE_FROM_STANDARD"):
            self.assertIn(status, DebtBasedScoring.EARNING_MINER_BUCKETS)
