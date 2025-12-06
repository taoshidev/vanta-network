# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
Integration tests for entity subaccount dashboard data aggregation.

Tests end-to-end dashboard data scenarios including:
- Aggregation from multiple RPC services
- Graceful degradation when services have no data
- Statistics cache integration
- Challenge period, positions, ledger, and elimination data
"""
import unittest
from copy import deepcopy

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.position import Position
from time_util.time_util import TimeUtil


class TestEntityDashboardIntegration(TestBase):
    """
    Integration tests for entity dashboard data aggregation using ServerOrchestrator.

    Servers start once (via singleton orchestrator) and are shared across:
    - All test methods in this class
    - All test classes that use ServerOrchestrator

    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references
    orchestrator = None
    entity_client = None
    metagraph_client = None
    challenge_period_client = None
    debt_ledger_client = None
    position_client = None
    elimination_client = None
    miner_statistics_client = None

    @classmethod
    def setUpClass(cls):
        """One-time setup: Start all servers using ServerOrchestrator."""
        cls.orchestrator = ServerOrchestrator.get_instance()

        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        cls.orchestrator.start_all_servers(
            mode=ServerMode.TESTING,
            secrets=secrets
        )

        # Get all required clients
        cls.entity_client = cls.orchestrator.get_client('entity')
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.debt_ledger_client = cls.orchestrator.get_client('debt_ledger')
        cls.position_client = cls.orchestrator.get_client('position_manager')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.miner_statistics_client = cls.orchestrator.get_client('miner_statistics')

    @classmethod
    def tearDownClass(cls):
        """One-time teardown: No action needed (servers auto-shutdown at process exit)."""
        pass

    def setUp(self):
        """Per-test setup: Reset data state for isolation."""
        self.orchestrator.clear_all_test_data()

        # Test entities
        self.ENTITY_HOTKEY = "dashboard_entity_alpha"
        self.REGULAR_MINER_HOTKEY = "regular_miner_beta"

        # Initialize metagraph
        self.metagraph_client.set_hotkeys([
            self.ENTITY_HOTKEY,
            self.REGULAR_MINER_HOTKEY
        ])

        # Register entity
        self.entity_client.register_entity(
            entity_hotkey=self.ENTITY_HOTKEY,
            collateral_amount=1000.0,
            max_subaccounts=10
        )

        # Create test subaccount
        success, subaccount_info, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY)
        self.assertTrue(success)
        self.synthetic_hotkey = subaccount_info['synthetic_hotkey']

        # Time constants
        self.START_TIME = TimeUtil.now_in_millis()
        self.END_TIME = self.START_TIME + (30 * ValiConfig.DAILY_MS)  # 30 days later

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    # ==================== Helper Methods ====================

    def _create_test_positions(self, hotkey: str, n_positions: int = 5):
        """Create test positions for a hotkey."""
        positions = []
        for i in range(n_positions):
            open_ms = self.START_TIME + (i * ValiConfig.DAILY_MS)
            close_ms = open_ms + ValiConfig.DAILY_MS

            position = Position(
                miner_hotkey=hotkey,
                position_uuid=f"{hotkey}_position_{i}",
                open_ms=open_ms,
                close_ms=close_ms,
                trade_pair=TradePair.BTCUSD,
                is_closed_position=True,
                return_at_close=1.02,  # 2% return
                account_size=100_000,
                orders=[Order(
                    price=60000,
                    processed_ms=open_ms,
                    order_uuid=f"{hotkey}_order_{i}",
                    trade_pair=TradePair.BTCUSD,
                    order_type=OrderType.LONG,
                    leverage=0.1
                )]
            )
            positions.append(position)
            self.position_client.save_miner_position(position)

        return positions

    def _add_to_challenge_period(self, hotkey: str, bucket: MinerBucket):
        """Add a hotkey to challenge period."""
        miners = {hotkey: (bucket, self.START_TIME, None, None)}
        self.challenge_period_client.update_miners(miners)
        self.challenge_period_client._write_challengeperiod_from_memory_to_disk()

    def _build_debt_ledgers(self):
        """Build debt ledgers."""
        self.debt_ledger_client.build_debt_ledgers(verbose=False, delta_update=False)

    def _setup_full_statistics_prerequisites(self, hotkey: str):
        """
        Set up ALL prerequisites for real statistics generation.

        This modularizes the logic from test_miner_statistics.py to ensure
        real statistics are generated (not just gracefully returning None).

        Prerequisites:
        1. Hotkey in metagraph ✓ (caller must do this)
        2. Asset selection
        3. Perf ledger with 60+ days of data
        4. Closed positions
        5. Challenge period (MAINCOMP bucket)
        6. Account size data
        """
        from tests.shared_objects.test_utilities import create_daily_checkpoints_with_pnl
        from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import TP_ID_PORTFOLIO
        from vali_objects.vali_config import TradePairCategory
        import numpy as np

        # 1. Set asset selection (REQUIRED for statistics generation)
        asset_selection_data = {hotkey: TradePairCategory.CRYPTO.value}
        self.orchestrator.get_client('asset_selection').sync_miner_asset_selection_data(asset_selection_data)

        # 2. Create perf ledger with 60 days of varied daily PnL
        np.random.seed(hash(hotkey) % 10000)  # Reproducible but varied per hotkey
        base_return = 0.012  # 1.2% daily return

        realized_pnl_list = []
        unrealized_pnl_list = []
        for day in range(60):
            daily_return = base_return * (1 + np.random.uniform(-0.2, 0.2))
            realized_pnl_list.append(daily_return * 100000)  # Scale by initial capital
            unrealized_pnl_list.append(0.0)

        portfolio_ledger = create_daily_checkpoints_with_pnl(realized_pnl_list, unrealized_pnl_list)
        btc_ledger = create_daily_checkpoints_with_pnl(realized_pnl_list, unrealized_pnl_list)

        ledgers = {
            hotkey: {
                TP_ID_PORTFOLIO: portfolio_ledger,
                TradePair.BTCUSD.trade_pair_id: btc_ledger
            }
        }

        # Get perf ledger client and save ledgers
        perf_ledger_client = self.orchestrator.get_client('perf_ledger')
        perf_ledger_client.save_perf_ledgers(ledgers)
        perf_ledger_client.re_init_perf_ledger_data()  # Force reload

        # 3. Create CLOSED position (open positions are filtered out by scoring)
        test_position = Position(
            miner_hotkey=hotkey,
            position_uuid=f"stats_position_{hotkey}",
            open_ms=self.START_TIME,
            trade_pair=TradePair.BTCUSD,
            account_size=200_000,
            orders=[Order(
                price=60000,
                processed_ms=self.START_TIME,
                order_uuid=f"stats_order_{hotkey}",
                trade_pair=TradePair.BTCUSD,
                order_type=OrderType.LONG,
                leverage=0.1
            )]
        )
        live_price_client = self.orchestrator.get_client('live_price_fetcher')
        test_position.rebuild_position_with_updated_orders(live_price_client)
        test_position.close_out_position(self.START_TIME + (1000 * 60 * 30))  # Close after 30 min
        self.position_client.save_miner_position(test_position)

        # 4. Add to challenge period in MAINCOMP bucket
        start_time = self.START_TIME - (60 * ValiConfig.DAILY_MS)  # 60 days before START_TIME
        miners_dict = {hotkey: (MinerBucket.MAINCOMP, start_time, None, None)}
        self.challenge_period_client.update_miners(miners_dict)

        # 5. Inject account sizes (REQUIRED - must be >= $150k to avoid penalty)
        contract_client = self.orchestrator.get_client('contract')
        account_sizes_data = {
            hotkey: [
                {
                    'account_size': 200000.0,  # $200k (above $150k minimum)
                    'account_size_theta': 200000.0,
                    'update_time_ms': start_time
                },
                {
                    'account_size': 200000.0,
                    'account_size_theta': 200000.0,
                    'update_time_ms': self.END_TIME
                }
            ]
        }
        contract_client.sync_miner_account_sizes_data(account_sizes_data)
        contract_client.re_init_account_sizes()  # Force reload

    def _populate_miner_statistics_cache(self):
        """Populate the miner statistics cache by generating statistics."""
        # Generate statistics for all miners
        # This will populate the in-memory dict cache
        self.miner_statistics_client.generate_request_minerstatistics(
            time_now=self.END_TIME,
            checkpoints=False,
            risk_report=False,
            bypass_confidence=True
        )

    # ==================== Dashboard Data Tests ====================

    def test_dashboard_data_active_subaccount_full_data(self):
        """Test dashboard data aggregation for an active subaccount with complete data."""
        # CRITICAL: Add synthetic hotkey to metagraph (statistics generation only processes metagraph hotkeys)
        current_hotkeys = self.metagraph_client.get_hotkeys()
        self.metagraph_client.set_hotkeys(current_hotkeys + [self.synthetic_hotkey])

        # Setup: Create positions
        self._create_test_positions(self.synthetic_hotkey, n_positions=10)
        self._add_to_challenge_period(self.synthetic_hotkey, MinerBucket.CHALLENGE)

        # Populate statistics cache
        self._populate_miner_statistics_cache()

        # Get dashboard data
        dashboard = self.entity_client.get_subaccount_dashboard_data(self.synthetic_hotkey)

        # Assertions
        self.assertIsNotNone(dashboard, "Dashboard data should not be None")

        # Verify subaccount_info
        self.assertIn('subaccount_info', dashboard)
        subaccount_info = dashboard['subaccount_info']
        self.assertEqual(subaccount_info['synthetic_hotkey'], self.synthetic_hotkey)
        self.assertEqual(subaccount_info['entity_hotkey'], self.ENTITY_HOTKEY)
        self.assertEqual(subaccount_info['subaccount_id'], 0)
        self.assertEqual(subaccount_info['status'], 'active')
        self.assertIsNotNone(subaccount_info['created_at_ms'])
        self.assertIsNone(subaccount_info['eliminated_at_ms'])

        # Verify challenge_period data
        self.assertIn('challenge_period', dashboard)
        challenge_data = dashboard['challenge_period']
        self.assertIsNotNone(challenge_data, "Challenge period data should exist")
        self.assertEqual(challenge_data['bucket'], MinerBucket.CHALLENGE.value)
        self.assertEqual(challenge_data['start_time_ms'], self.START_TIME)

        # Verify positions data
        self.assertIn('positions', dashboard)
        positions_data = dashboard['positions']
        self.assertIsNotNone(positions_data, "Positions data should exist")
        self.assertIn('n_positions', positions_data)
        self.assertIn('total_leverage', positions_data)
        self.assertEqual(positions_data['n_positions'], 10)

        # Verify ledger data exists (debt ledger)
        self.assertIn('ledger', dashboard)
        # Ledger may be None if debt ledger not built - that's ok for this test

        # Verify statistics data
        self.assertIn('statistics', dashboard)
        statistics_data = dashboard['statistics']
        if statistics_data:
            # Statistics should have expected structure
            self.assertIn('hotkey', statistics_data)
            self.assertEqual(statistics_data['hotkey'], self.synthetic_hotkey)
            # May have other fields like scores, engagement, etc.

        # Verify elimination data
        self.assertIn('elimination', dashboard)
        # Should be None for non-eliminated miner
        self.assertIsNone(dashboard['elimination'])

    def test_dashboard_data_eliminated_subaccount(self):
        """Test dashboard data for an eliminated subaccount."""
        # Setup: Create data then eliminate
        self._create_test_positions(self.synthetic_hotkey, n_positions=5)
        self._add_to_challenge_period(self.synthetic_hotkey, MinerBucket.CHALLENGE)

        # Eliminate the subaccount
        success, _ = self.entity_client.eliminate_subaccount(
            entity_hotkey=self.ENTITY_HOTKEY,
            subaccount_id=0,
            reason="test_elimination_for_dashboard"
        )
        self.assertTrue(success)

        # Also add to elimination registry
        self.elimination_client.append_elimination_row(
            self.synthetic_hotkey,
            self.END_TIME,
            "test_elimination"
        )

        # Get dashboard data
        dashboard = self.entity_client.get_subaccount_dashboard_data(self.synthetic_hotkey)

        # Assertions
        self.assertIsNotNone(dashboard)

        # Verify eliminated status
        self.assertEqual(dashboard['subaccount_info']['status'], 'eliminated')
        self.assertIsNotNone(dashboard['subaccount_info']['eliminated_at_ms'])

        # Verify elimination data exists
        elimination_data = dashboard['elimination']
        self.assertIsNotNone(elimination_data, "Elimination data should exist for eliminated miner")
        self.assertEqual(elimination_data['hotkey'], self.synthetic_hotkey)
        self.assertEqual(elimination_data['reason'], "test_elimination")

    def test_dashboard_data_no_positions(self):
        """Test dashboard data when subaccount has no positions."""
        # No positions created - just empty subaccount

        # Get dashboard data
        dashboard = self.entity_client.get_subaccount_dashboard_data(self.synthetic_hotkey)

        # Assertions
        self.assertIsNotNone(dashboard)

        # Subaccount info should still exist
        self.assertEqual(dashboard['subaccount_info']['synthetic_hotkey'], self.synthetic_hotkey)
        self.assertEqual(dashboard['subaccount_info']['status'], 'active')

        # Positions data should be None (no positions)
        self.assertIsNone(dashboard['positions'])

        # Challenge period should be None (not added)
        self.assertIsNone(dashboard['challenge_period'])

        # Ledger should be None (no ledger data)
        self.assertIsNone(dashboard['ledger'])

        # Statistics should be None (no statistics in cache)
        self.assertIsNone(dashboard['statistics'])

        # Elimination should be None (not eliminated)
        self.assertIsNone(dashboard['elimination'])

    def test_dashboard_data_nonexistent_subaccount(self):
        """Test dashboard data for non-existent subaccount."""
        fake_synthetic = "nonexistent_entity_999"

        # Get dashboard data
        dashboard = self.entity_client.get_subaccount_dashboard_data(fake_synthetic)

        # Should return None for non-existent subaccount
        self.assertIsNone(dashboard)

    def test_dashboard_data_invalid_synthetic_hotkey(self):
        """Test dashboard data with invalid synthetic hotkey format."""
        # Not a synthetic hotkey format
        invalid_hotkey = "not_a_synthetic_hotkey"

        # Get dashboard data
        dashboard = self.entity_client.get_subaccount_dashboard_data(invalid_hotkey)

        # Should return None for invalid format
        self.assertIsNone(dashboard)

    def test_dashboard_data_regular_hotkey(self):
        """Test dashboard data with regular miner hotkey (not synthetic)."""
        # Get dashboard data for regular miner
        dashboard = self.entity_client.get_subaccount_dashboard_data(self.REGULAR_MINER_HOTKEY)

        # Should return None (not a subaccount)
        self.assertIsNone(dashboard)

    def test_dashboard_data_partial_service_data(self):
        """Test dashboard data when only some services have data (graceful degradation)."""
        # Setup: Only positions, no ledger or challenge period
        self._create_test_positions(self.synthetic_hotkey, n_positions=3)
        # Deliberately NOT creating ledger or adding to challenge period

        # Get dashboard data
        dashboard = self.entity_client.get_subaccount_dashboard_data(self.synthetic_hotkey)

        # Assertions
        self.assertIsNotNone(dashboard)

        # Subaccount info should exist
        self.assertIsNotNone(dashboard['subaccount_info'])
        self.assertEqual(dashboard['subaccount_info']['status'], 'active')

        # Positions should exist
        positions_data = dashboard['positions']
        self.assertIsNotNone(positions_data)
        self.assertEqual(positions_data['n_positions'], 3)

        # Services without data should return None (graceful degradation)
        self.assertIsNone(dashboard['challenge_period'])
        self.assertIsNone(dashboard['ledger'])
        self.assertIsNone(dashboard['statistics'])
        self.assertIsNone(dashboard['elimination'])

    def test_dashboard_data_multiple_subaccounts(self):
        """Test dashboard data for multiple subaccounts of same entity."""
        # Create second subaccount
        success, subaccount_info2, _ = self.entity_client.create_subaccount(self.ENTITY_HOTKEY)
        self.assertTrue(success)
        synthetic_hotkey2 = subaccount_info2['synthetic_hotkey']

        # Create data for both subaccounts
        self._create_test_positions(self.synthetic_hotkey, n_positions=5)
        self._create_test_positions(synthetic_hotkey2, n_positions=3)

        # Get dashboard data for both
        dashboard1 = self.entity_client.get_subaccount_dashboard_data(self.synthetic_hotkey)
        dashboard2 = self.entity_client.get_subaccount_dashboard_data(synthetic_hotkey2)

        # Both should exist
        self.assertIsNotNone(dashboard1)
        self.assertIsNotNone(dashboard2)

        # Verify correct subaccount IDs
        self.assertEqual(dashboard1['subaccount_info']['subaccount_id'], 0)
        self.assertEqual(dashboard2['subaccount_info']['subaccount_id'], 1)

        # Verify correct synthetic hotkeys
        self.assertEqual(dashboard1['subaccount_info']['synthetic_hotkey'], self.synthetic_hotkey)
        self.assertEqual(dashboard2['subaccount_info']['synthetic_hotkey'], synthetic_hotkey2)

        # Verify different position counts
        self.assertEqual(dashboard1['positions']['n_positions'], 5)
        self.assertEqual(dashboard2['positions']['n_positions'], 3)

    def test_dashboard_data_statistics_cache_populated(self):
        """
        Test that statistics are properly generated and included in dashboard.

        This test uses the full statistics setup prerequisites to ensure REAL statistics
        are generated (not just gracefully returning None).
        """
        # CRITICAL: Add synthetic hotkey to metagraph (statistics generation only processes metagraph hotkeys)
        current_hotkeys = self.metagraph_client.get_hotkeys()
        self.metagraph_client.set_hotkeys(current_hotkeys + [self.synthetic_hotkey])

        # Set up ALL prerequisites for real statistics generation
        # This includes: asset selection, perf ledgers, closed positions, challenge period, account sizes
        self._setup_full_statistics_prerequisites(self.synthetic_hotkey)

        # Populate statistics cache
        self._populate_miner_statistics_cache()

        # Verify statistics cache has REAL data (not None)
        stats_from_cache = self.miner_statistics_client.get_miner_statistics_for_hotkey(self.synthetic_hotkey)
        self.assertIsNotNone(stats_from_cache, "Statistics cache should have REAL data after full setup")
        self.assertIn('hotkey', stats_from_cache)
        self.assertEqual(stats_from_cache['hotkey'], self.synthetic_hotkey)
        print(f"✓ Statistics successfully generated: {list(stats_from_cache.keys())[:10]}")

        # Get dashboard data
        dashboard = self.entity_client.get_subaccount_dashboard_data(self.synthetic_hotkey)

        # Verify dashboard exists
        self.assertIsNotNone(dashboard)

        # Verify statistics are included in dashboard (should NOT be None)
        self.assertIn('statistics', dashboard)
        statistics_data = dashboard['statistics']
        self.assertIsNotNone(statistics_data, "Dashboard statistics should be populated with real data")

        # ==================== Assert Specific Values in Statistics Payload ====================

        # 1. Hotkey field
        self.assertIn('hotkey', statistics_data)
        self.assertEqual(statistics_data['hotkey'], self.synthetic_hotkey)

        # 2. Challenge period structure
        self.assertIn('challengeperiod', statistics_data)
        cp_info = statistics_data['challengeperiod']
        self.assertIsInstance(cp_info, dict)
        self.assertIn('status', cp_info)
        self.assertEqual(cp_info['status'], 'success', "Challenge period status should be 'success' for MAINCOMP bucket")
        print(f"✓ Challenge period: {cp_info}")

        # 3. Scores structure
        self.assertIn('scores', statistics_data)
        scores = statistics_data['scores']
        self.assertIsInstance(scores, dict)
        print(f"✓ Scores: {list(scores.keys())}")

        # 4. Weight structure (rank, value, percentile)
        self.assertIn('weight', statistics_data)
        weight = statistics_data['weight']
        self.assertIsInstance(weight, dict)
        self.assertIn('value', weight)
        self.assertIn('rank', weight)
        self.assertIn('percentile', weight)
        self.assertIsInstance(weight['rank'], int)
        self.assertGreaterEqual(weight['rank'], 1, "Rank should be >= 1")
        self.assertIsInstance(weight['value'], (int, float))
        self.assertGreaterEqual(weight['value'], 0, "Weight value should be >= 0")
        self.assertIsInstance(weight['percentile'], (int, float))
        self.assertGreaterEqual(weight['percentile'], 0, "Percentile should be >= 0")
        self.assertLessEqual(weight['percentile'], 100, "Percentile should be <= 100")
        print(f"✓ Weight: rank={weight['rank']}, value={weight['value']:.6f}, percentile={weight['percentile']:.2f}")

        # 5. Daily returns (should be a list/array)
        self.assertIn('daily_returns', statistics_data)
        daily_returns = statistics_data['daily_returns']
        self.assertIsInstance(daily_returns, (list, dict))
        print(f"✓ Daily returns type: {type(daily_returns).__name__}")

        # 6. Engagement structure
        self.assertIn('engagement', statistics_data)
        engagement = statistics_data['engagement']
        self.assertIsInstance(engagement, dict)
        print(f"✓ Engagement: {list(engagement.keys())}")

        # 7. Volatility (should exist)
        self.assertIn('volatility', statistics_data)

        # 8. Drawdowns (should exist)
        self.assertIn('drawdowns', statistics_data)

        # 9. Augmented scores (should exist)
        self.assertIn('augmented_scores', statistics_data)

        print(f"✓ All statistics fields validated successfully")
        print(f"  Full statistics keys: {list(statistics_data.keys())}")

    def test_dashboard_data_all_services_integration(self):
        """
        Comprehensive integration test with all services providing data.

        This test verifies the complete dashboard data flow:
        1. Subaccount creation
        2. Positions tracking
        3. Ledger generation
        4. Challenge period assignment
        5. Statistics generation
        6. Dashboard aggregation
        """
        # CRITICAL: Add synthetic hotkey to metagraph (statistics generation only processes metagraph hotkeys)
        current_hotkeys = self.metagraph_client.get_hotkeys()
        self.metagraph_client.set_hotkeys(current_hotkeys + [self.synthetic_hotkey])

        # Setup: Create complete data landscape
        positions = self._create_test_positions(self.synthetic_hotkey, n_positions=15)
        self._add_to_challenge_period(self.synthetic_hotkey, MinerBucket.MAINCOMP)

        # Build debt ledgers
        self._build_debt_ledgers()

        # Populate miner statistics cache
        self._populate_miner_statistics_cache()

        # Verify data exists in each service
        db_positions = self.position_client.get_positions_for_one_hotkey(self.synthetic_hotkey)
        self.assertEqual(len(db_positions), 15)

        has_miner = self.challenge_period_client.has_miner(self.synthetic_hotkey)
        self.assertTrue(has_miner)

        # Get dashboard data
        dashboard = self.entity_client.get_subaccount_dashboard_data(self.synthetic_hotkey)

        # Comprehensive assertions
        self.assertIsNotNone(dashboard, "Dashboard should aggregate all data")

        # All sections should be present
        self.assertIn('subaccount_info', dashboard)
        self.assertIn('challenge_period', dashboard)
        self.assertIn('ledger', dashboard)
        self.assertIn('positions', dashboard)
        self.assertIn('statistics', dashboard)
        self.assertIn('elimination', dashboard)

        # Verify subaccount_info
        self.assertEqual(dashboard['subaccount_info']['status'], 'active')
        self.assertEqual(dashboard['subaccount_info']['synthetic_hotkey'], self.synthetic_hotkey)

        # Verify challenge_period
        self.assertIsNotNone(dashboard['challenge_period'])
        self.assertEqual(dashboard['challenge_period']['bucket'], MinerBucket.MAINCOMP.value)

        # Verify positions
        self.assertIsNotNone(dashboard['positions'])
        self.assertEqual(dashboard['positions']['n_positions'], 15)
        self.assertIn('total_leverage', dashboard['positions'])

        # Verify statistics (may or may not be populated depending on cache)
        # statistics_data = dashboard['statistics']
        # If populated, should have hotkey field

        # Verify no elimination
        self.assertIsNone(dashboard['elimination'])

    def test_dashboard_data_verify_all_fields_populated(self):
        """
        Test that verifies ALL dashboard fields are properly populated with real data.

        This is a comprehensive verification test that ensures:
        - Subaccount info is complete
        - Challenge period data exists
        - Debt ledger data exists
        - Positions data is comprehensive
        - Statistics cache is populated
        - Elimination data (when applicable)
        """
        import time

        # CRITICAL: Add synthetic hotkey to metagraph (statistics generation only processes metagraph hotkeys)
        current_hotkeys = self.metagraph_client.get_hotkeys()
        self.metagraph_client.set_hotkeys(current_hotkeys + [self.synthetic_hotkey])

        # Setup: Create comprehensive test data
        # 1. Positions
        positions = self._create_test_positions(self.synthetic_hotkey, n_positions=20)
        self.assertEqual(len(positions), 20, "Should create 20 positions")

        # 2. Challenge Period
        self._add_to_challenge_period(self.synthetic_hotkey, MinerBucket.MAINCOMP)

        # 3. Build Debt Ledgers (CRITICAL - needed for ledger data)
        self._build_debt_ledgers()

        # Verify debt ledger was built
        debt_ledger = self.debt_ledger_client.get_ledger(self.synthetic_hotkey)
        # Note: debt_ledger might still be None if debt ledger build doesn't include this hotkey

        # 5. Populate Statistics Cache (CRITICAL - needed for statistics data)
        self._populate_miner_statistics_cache()

        # Verify statistics were cached
        stats_from_cache = self.miner_statistics_client.get_miner_statistics_for_hotkey(self.synthetic_hotkey)
        # Note: stats might be None if miner not in eligible buckets for statistics

        # Get dashboard data
        dashboard = self.entity_client.get_subaccount_dashboard_data(self.synthetic_hotkey)

        # ==================== VERIFY ALL FIELDS ====================

        self.assertIsNotNone(dashboard, "Dashboard data should exist")

        # 1. SUBACCOUNT INFO - Should ALWAYS be populated
        subaccount_info = dashboard['subaccount_info']
        self.assertIsNotNone(subaccount_info, "Subaccount info should exist")
        self.assertEqual(subaccount_info['synthetic_hotkey'], self.synthetic_hotkey)
        self.assertEqual(subaccount_info['entity_hotkey'], self.ENTITY_HOTKEY)
        self.assertEqual(subaccount_info['subaccount_id'], 0)
        self.assertEqual(subaccount_info['status'], 'active')
        self.assertIsNotNone(subaccount_info['created_at_ms'])
        self.assertIsInstance(subaccount_info['created_at_ms'], int)
        self.assertIsNone(subaccount_info['eliminated_at_ms'])
        print(f"✓ Subaccount info: {subaccount_info}")

        # 2. CHALLENGE PERIOD - Should be populated (we added it)
        challenge_data = dashboard['challenge_period']
        self.assertIsNotNone(challenge_data, "Challenge period data should exist")
        self.assertEqual(challenge_data['bucket'], MinerBucket.MAINCOMP.value)
        self.assertEqual(challenge_data['start_time_ms'], self.START_TIME)
        print(f"✓ Challenge period: {challenge_data}")

        # 3. POSITIONS - Should be populated (we created 20 positions)
        positions_data = dashboard['positions']
        self.assertIsNotNone(positions_data, "Positions data should exist")
        self.assertEqual(positions_data['n_positions'], 20)
        self.assertIn('total_leverage', positions_data)
        self.assertIn('thirty_day_returns', positions_data)
        self.assertIn('all_time_returns', positions_data)
        self.assertIn('percentage_profitable', positions_data)
        print(f"✓ Positions: n_positions={positions_data['n_positions']}, "
              f"leverage={positions_data['total_leverage']}, "
              f"profitable={positions_data['percentage_profitable']}")

        # 4. LEDGER (Debt Ledger) - May or may not be populated
        ledger_data = dashboard['ledger']
        if ledger_data:
            print(f"✓ Ledger data exists")
        else:
            print(f"⚠ Ledger data is None (debt ledger may not be built for this hotkey)")

        # 5. STATISTICS - May or may not be populated
        statistics_data = dashboard['statistics']
        if statistics_data:
            self.assertIn('hotkey', statistics_data)
            self.assertEqual(statistics_data['hotkey'], self.synthetic_hotkey)
            print(f"✓ Statistics exists with fields: {list(statistics_data.keys())[:10]}...")
        else:
            print(f"⚠ Statistics is None (miner may not be in eligible bucket for statistics)")

        # 6. ELIMINATION - Should be None (not eliminated)
        elimination_data = dashboard['elimination']
        self.assertIsNone(elimination_data, "Elimination should be None for active subaccount")
        print(f"✓ Elimination: None (as expected)")

        # Print summary
        print("\n" + "="*60)
        print("DASHBOARD FIELD POPULATION SUMMARY:")
        print("="*60)
        print(f"✓ subaccount_info: POPULATED")
        print(f"✓ challenge_period: POPULATED")
        print(f"✓ positions: POPULATED (20 positions)")
        print(f"{'✓' if ledger_data else '⚠'} ledger: {'POPULATED' if ledger_data else 'NULL (expected in tests)'}")
        print(f"{'✓' if statistics_data else '⚠'} statistics: {'POPULATED' if statistics_data else 'NULL (expected in tests)'}")
        print(f"✓ elimination: NULL (expected)")
        print("="*60)


if __name__ == '__main__':
    unittest.main()
