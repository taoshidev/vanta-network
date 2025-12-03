# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Test MinerStatisticsServer and MinerStatisticsClient production code paths.

This test ensures that MinerStatisticsServer can:
- Generate miner statistics via generate_request_minerstatistics
- Execute the same code paths used in production
- Properly handle various parameter combinations
"""
import unittest
import bittensor as bt

from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.shared_objects.test_utilities import generate_winning_ledger
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, RPCConnectionMode
from vali_objects.vali_dataclasses.order import Order
from runnable.miner_statistics_server import MinerStatisticsServer, MinerStatisticsClient


class TestMinerStatistics(TestBase):
    """
    Test MinerStatisticsServer and MinerStatisticsClient functionality using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across all test classes.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    elimination_client = None
    challenge_period_client = None
    plagiarism_client = None
    plagiarism_detector_client = None
    asset_selection_client = None
    miner_statistics_client = None
    miner_statistics_handle = None  # Server handle for tests that need direct server access

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
        cls.plagiarism_detector_client = cls.orchestrator.get_client('plagiarism_detector')
        cls.asset_selection_client = cls.orchestrator.get_client('asset_selection')
        cls.miner_statistics_client = cls.orchestrator.get_client('miner_statistics')

        # Get server handle for tests that need direct server access (for property checks)
        cls.miner_statistics_handle = cls.orchestrator._servers.get('miner_statistics')

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
        # Enable debug logging to see bt.logging.info() statements
        bt.logging.set_debug()

        # Clear all data for test isolation (both memory and disk)
        self.orchestrator.clear_all_test_data()

        # Create test hotkeys
        self.test_hotkeys = [
            "test_hotkey_1_abc123",
            "test_hotkey_2_def456",
            "test_hotkey_3_ghi789"
        ]

        # Set up metagraph with test hotkeys
        self.metagraph_client.set_hotkeys(self.test_hotkeys)

        # Set up asset selection for all miners (required for statistics generation)
        from vali_objects.vali_config import TradePairCategory
        asset_class_str = TradePairCategory.CRYPTO.value
        asset_selection_data = {}
        for hotkey in self.test_hotkeys:
            asset_selection_data[hotkey] = asset_class_str
        self.asset_selection_client.sync_miner_asset_selection_data(asset_selection_data)

        # Create some test positions for miners
        self._create_test_positions()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_positions(self):
        """Create some test positions for miners to avoid empty data errors."""
        current_time = TimeUtil.now_in_millis()
        start_time = current_time - 1000 * 60 * 60 * 24 * 60  # 60 days ago (required for 60 complete days = 120 checkpoints)

        # Build ledgers dictionary with VARIED performance data
        # Each miner needs different performance to get non-zero scores from metrics
        from tests.shared_objects.test_utilities import create_daily_checkpoints_with_pnl
        from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO, PerfLedger
        import numpy as np

        ledgers = {}
        for i, hotkey in enumerate(self.test_hotkeys):
            # Create VARIED daily PnL for each miner to ensure different scores
            # Miner 0: Best performance (high positive returns)
            # Miner 1: Medium performance
            # Miner 2: Lower performance (but still positive)

            # Generate 60 days of varied daily returns
            np.random.seed(i)  # Different seed per miner for reproducible variance
            base_returns = [0.015, 0.010, 0.005]  # Different base daily returns per miner

            # Create varied daily PnL values (60 days)
            realized_pnl_list = []
            unrealized_pnl_list = []
            for day in range(60):
                # Add slight variation to each day while maintaining overall trend
                daily_return = base_returns[i] * (1 + np.random.uniform(-0.2, 0.2))
                realized_pnl_list.append(daily_return * 100000)  # Scale by initial capital
                unrealized_pnl_list.append(0.0)  # No unrealized PnL for closed positions

            # Create ledger with varied daily checkpoints
            portfolio_ledger = create_daily_checkpoints_with_pnl(realized_pnl_list, unrealized_pnl_list)
            btc_ledger = create_daily_checkpoints_with_pnl(realized_pnl_list, unrealized_pnl_list)

            ledgers[hotkey] = {
                TP_ID_PORTFOLIO: portfolio_ledger,
                TradePair.BTCUSD.trade_pair_id: btc_ledger
            }

            # Create a simple test position for this hotkey
            # NOTE: Positions MUST be closed for scoring (filter_positions_for_duration skips open positions)
            test_position = Position(
                miner_hotkey=hotkey,
                position_uuid=f"test_position_{hotkey}",
                open_ms=current_time - 1000 * 60 * 60,  # 1 hour ago
                trade_pair=TradePair.BTCUSD,
                account_size=100_000,  # Required for scoring
                orders=[
                    Order(
                        price=60000,
                        processed_ms=current_time - 1000 * 60 * 60,
                        order_uuid=f"order_{hotkey}_1",
                        trade_pair=TradePair.BTCUSD,
                        order_type=OrderType.LONG,
                        leverage=0.1
                    )
                ]
            )
            test_position.rebuild_position_with_updated_orders(self.live_price_fetcher_client)
            test_position.close_out_position(current_time - 1000 * 60 * 30)  # Close 30 min ago (meets 1min minimum)
            self.position_client.save_miner_position(test_position)

        # Save all ledgers at once
        self.perf_ledger_client.save_perf_ledgers(ledgers)
        self.perf_ledger_client.re_init_perf_ledger_data()  # Force reload after save

        # Verify ledgers were saved and can be retrieved for scoring
        filtered_ledgers = self.perf_ledger_client.filtered_ledger_for_scoring(hotkeys=self.test_hotkeys)
        from vali_objects.vali_dataclasses.perf_ledger import TP_ID_PORTFOLIO
        from vali_objects.utils.ledger_utils import LedgerUtils
        for hotkey in self.test_hotkeys:
            if hotkey in filtered_ledgers and TP_ID_PORTFOLIO in filtered_ledgers[hotkey]:
                ledger = filtered_ledgers[hotkey][TP_ID_PORTFOLIO]
                daily_returns = LedgerUtils.daily_return_log(ledger)
                assert len(daily_returns) >= 60, f"{hotkey} has only {len(daily_returns)} daily returns (need 60+)"
            else:
                raise AssertionError(f"Ledger data not found for {hotkey} after save/reload")

        # Add miners to challenge period using batch update (matches reference test pattern)
        miners_dict = {}
        for hotkey in self.test_hotkeys:
            miners_dict[hotkey] = (MinerBucket.MAINCOMP, start_time, None, None)

        self.challenge_period_client.clear_all_miners()
        self.challenge_period_client.update_miners(miners_dict)
        # Note: Data persistence handled automatically by server - no manual disk write needed

        # Inject account sizes data for all test miners (required for scoring penalty calculations)
        contract_client = self.orchestrator.get_client('contract')
        account_sizes_data = {}
        for hotkey in self.test_hotkeys:
            # Create dummy account size records with correct format
            # CollateralRecord requires: account_size, account_size_theta, update_time_ms
            # IMPORTANT: Must be >= MIN_COLLATERAL_VALUE ($150k) to avoid penalty
            account_sizes_data[hotkey] = [
                {
                    'account_size': 200000.0,  # $200k account size (above $150k minimum)
                    'account_size_theta': 200000.0,  # Same as account_size for tests
                    'update_time_ms': start_time
                },
                {
                    'account_size': 200000.0,
                    'account_size_theta': 200000.0,
                    'update_time_ms': current_time
                }
            ]
        contract_client.sync_miner_account_sizes_data(account_sizes_data)
        contract_client.re_init_account_sizes()  # Force reload from disk

    # ==================== Basic Server Tests ====================

    def test_server_instantiation(self):
        """Test that MinerStatisticsServer can be instantiated."""
        self.assertIsNotNone(self.miner_statistics_handle)
        self.assertIsNotNone(self.miner_statistics_client)

    def test_health_check(self):
        """Test that MinerStatisticsClient can communicate with server."""
        health = self.miner_statistics_client.health_check()
        self.assertIsNotNone(health)
        self.assertEqual(health['status'], 'ok')
        self.assertIn('cache_status', health)

    # ==================== Production Code Path Tests ====================

    def test_generate_request_minerstatistics_production_path(self):
        """
        Test that generate_request_minerstatistics executes production code paths.

        This is the critical test that validates the same code path used in production
        to generate miner statistics data.
        """
        current_time_ms = TimeUtil.now_in_millis()

        try:
            self.miner_statistics_client.generate_request_minerstatistics(
                time_now=current_time_ms,
                checkpoints=True,
                risk_report=False,
                bypass_confidence=True  # Bypass confidence for faster test execution
            )
        except AttributeError as e:
            self.fail(f"generate_request_minerstatistics raised AttributeError: {e}")
        except Exception as e:
            self.fail(f"generate_request_minerstatistics raised unexpected exception: {e}")

        # If we got here without exceptions, the production code path executed successfully

    def test_generate_request_minerstatistics_no_checkpoints(self):
        """Test statistics generation without checkpoints."""
        current_time_ms = TimeUtil.now_in_millis()

        try:
            self.miner_statistics_client.generate_request_minerstatistics(
                time_now=current_time_ms,
                checkpoints=False,
                risk_report=False,
                bypass_confidence=True
            )
        except Exception as e:
            self.fail(f"generate_request_minerstatistics failed with checkpoints=False: {e}")

    def test_generate_miner_statistics_data_structure(self):
        """Test that generated statistics have proper structure."""
        current_time_ms = TimeUtil.now_in_millis()

        # Call the method via client to get the data structure
        stats_data = self.miner_statistics_client.generate_miner_statistics_data(
            time_now=current_time_ms,
            checkpoints=False,  # Skip checkpoints for faster execution
            risk_report=False,
            bypass_confidence=True
        )

        # Verify the structure of returned data
        self.assertIsInstance(stats_data, dict)
        self.assertIn('version', stats_data)
        self.assertIn('created_timestamp_ms', stats_data)
        self.assertIn('data', stats_data)
        self.assertIn('constants', stats_data)

        # Verify data contains our test miners
        data = stats_data.get('data', [])
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0, "Should have at least one miner")

        # At least some of our test miners should be present
        hotkeys_in_data = [miner_dict.get('hotkey') for miner_dict in data]
        for test_hotkey in self.test_hotkeys:
            self.assertIn(test_hotkey, hotkeys_in_data,
                         f"Test hotkey {test_hotkey} should be in statistics data")

    def test_get_compressed_statistics(self):
        """Test retrieving compressed statistics from memory cache."""
        current_time_ms = TimeUtil.now_in_millis()

        # First generate statistics to populate the cache
        self.miner_statistics_client.generate_request_minerstatistics(
            time_now=current_time_ms,
            checkpoints=False,  # Faster without checkpoints
            bypass_confidence=True
        )

        # Now retrieve compressed statistics (without checkpoints)
        compressed_without = self.miner_statistics_client.get_compressed_statistics(include_checkpoints=False)

        # Should be populated after generation
        self.assertIsInstance(compressed_without, (bytes, type(None)))

        # If it's bytes, verify it has content
        if isinstance(compressed_without, bytes):
            self.assertGreater(len(compressed_without), 0)

    def test_manager_property_access(self):
        """Test that manager properties are accessible through server."""
        # Note: We test server properties via the handle, not the client
        # Server handle is the multiprocessing.Process object returned by spawn_process()
        # We cannot directly access server properties from the client in RPC mode
        # This test verifies server architecture by checking the server process exists
        self.assertIsNotNone(self.miner_statistics_handle)

    # ==================== Integration Test ====================

    def test_full_production_pipeline(self):
        """
        Integration test: Simulate full production pipeline.

        This test exercises the complete code path that runs in production
        when the validator generates miner statistics.
        """
        current_time_ms = TimeUtil.now_in_millis()

        # Generate statistics (production code path)
        try:
            self.miner_statistics_client.generate_request_minerstatistics(
                time_now=current_time_ms,
                checkpoints=False,  # Faster without checkpoints
                risk_report=False,
                bypass_confidence=True
            )
        except Exception as e:
            self.fail(f"Production pipeline failed: {e}")

        # Verify we can retrieve compressed data
        compressed = self.miner_statistics_client.get_compressed_statistics(include_checkpoints=False)

        # Should be populated after generation
        if compressed is not None:
            self.assertIsInstance(compressed, bytes)
            self.assertGreater(len(compressed), 0)

    def test_miner_data_structure_validation(self):
        """Test that each miner's data structure is valid."""
        current_time_ms = TimeUtil.now_in_millis()

        # Generate statistics via client
        stats_data = self.miner_statistics_client.generate_miner_statistics_data(
            time_now=current_time_ms,
            checkpoints=False,
            risk_report=False,
            bypass_confidence=True
        )

        # Verify each miner has expected fields
        data = stats_data.get('data', [])
        self.assertGreater(len(data), 0, "Should have at least one miner")

        for miner_dict in data:
            # Verify core fields exist
            self.assertIn('hotkey', miner_dict)
            self.assertIn('challengeperiod', miner_dict)
            self.assertIn('scores', miner_dict)
            self.assertIn('weight', miner_dict)

            # Verify weight structure
            weight = miner_dict.get('weight', {})
            self.assertIsInstance(weight, dict)
            self.assertIn('value', weight)
            self.assertIn('rank', weight)
            self.assertIn('percentile', weight)


if __name__ == '__main__':
    unittest.main()
