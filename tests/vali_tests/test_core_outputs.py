# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Test CoreOutputsServer and CoreOutputsClient production code paths.

This test ensures that CoreOutputsServer can:
- Generate checkpoint data via generate_request_core
- Properly expose RPC methods
- Execute the same code paths used in production

Uses RPC mode with ServerOrchestrator for shared server infrastructure.
"""
import unittest

from shared_objects.rpc.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from time_util.time_util import TimeUtil
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.vali_dataclasses.position import Position
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair
from vali_objects.vali_dataclasses.order import Order


class TestCoreOutputs(TestBase):
    """
    Test CoreOutputsServer and CoreOutputsClient functionality using RPC mode.
    Uses class-level server setup for efficiency - servers start once and are shared.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    live_price_fetcher_client = None
    metagraph_client = None
    position_client = None
    perf_ledger_client = None
    challenge_period_client = None
    elimination_client = None
    plagiarism_client = None
    core_outputs_client = None

    # Test constants
    test_hotkeys = [
        "test_hotkey_1_abc123",
        "test_hotkey_2_def456",
        "test_hotkey_3_ghi789"
    ]

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
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.elimination_client = cls.orchestrator.get_client('elimination')
        cls.plagiarism_client = cls.orchestrator.get_client('plagiarism')
        cls.core_outputs_client = cls.orchestrator.get_client('core_outputs')

        # Initialize metagraph with test hotkeys
        cls.metagraph_client.set_hotkeys(cls.test_hotkeys)

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

        # Set up metagraph with test hotkeys
        self.metagraph_client.set_hotkeys(self.test_hotkeys)

        # Create some test positions for miners
        self._create_test_positions()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def _create_test_positions(self):
        """Create some test positions for miners to avoid empty data errors."""
        current_time = TimeUtil.now_in_millis()

        for hotkey in self.test_hotkeys:
            # Add to challenge period
            self.challenge_period_client.set_miner_bucket(
                hotkey,
                MinerBucket.CHALLENGE,
                current_time - 1000 * 60 * 60 * 24  # 1 day ago
            )

            # Create a simple test position
            test_position = Position(
                miner_hotkey=hotkey,
                position_uuid=f"test_position_{hotkey}",
                open_ms=current_time - 1000 * 60 * 60,  # 1 hour ago
                trade_pair=TradePair.BTCUSD,
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
            test_position.close_out_position(current_time - 1000 * 60 * 30)  # 30 min ago
            self.position_client.save_miner_position(test_position)

    # ==================== Basic Server Tests ====================

    def test_client_instantiation(self):
        """Test that CoreOutputsClient is available."""
        self.assertIsNotNone(self.core_outputs_client)

    def test_health_check(self):
        """Test that CoreOutputsClient can communicate with server."""
        health = self.core_outputs_client.health_check()
        self.assertIsNotNone(health)
        self.assertEqual(health['status'], 'ok')
        self.assertIn('cache_status', health)

    # ==================== Production Code Path Tests ====================

    def test_generate_request_core_production_path(self):
        """
        Test that generate_request_core executes production code paths.

        This is the critical test that validates the same code path used in production
        to generate checkpoint data for API consumption.
        """
        try:
            checkpoint_dict = self.core_outputs_client.generate_request_core(
                create_production_files=True,  # Create the dicts
                save_production_files=False,   # Don't write to disk
                upload_production_files=False  # Don't upload to gcloud
            )
        except AttributeError as e:
            self.fail(f"generate_request_core raised AttributeError (likely missing RPC method): {e}")
        except Exception as e:
            self.fail(f"generate_request_core raised unexpected exception: {e}")

        # Verify the checkpoint dict has expected keys
        self.assertIn('challengeperiod', checkpoint_dict)
        self.assertIn('miner_account_sizes', checkpoint_dict)
        self.assertIn('positions', checkpoint_dict)

        # Verify challengeperiod dict is not empty (we added test miners)
        self.assertIsInstance(checkpoint_dict['challengeperiod'], dict)

        # Verify our test miners are present
        challengeperiod = checkpoint_dict['challengeperiod']
        for hotkey in self.test_hotkeys:
            self.assertIn(hotkey, challengeperiod, f"Test hotkey {hotkey} should be in challengeperiod dict")

    def test_checkpoint_dict_structure(self):
        """Test that checkpoint dict has proper structure and data."""
        checkpoint_dict = self.core_outputs_client.generate_request_core(
            create_production_files=True,
            save_production_files=False,
            upload_production_files=False
        )

        # Verify all test miners are in challengeperiod dict
        challengeperiod = checkpoint_dict.get('challengeperiod', {})
        for hotkey in self.test_hotkeys:
            self.assertIn(hotkey, challengeperiod)
            miner_data = challengeperiod[hotkey]
            self.assertIn('bucket', miner_data)
            self.assertIn('bucket_start_time', miner_data)
            self.assertEqual(miner_data['bucket'], 'CHALLENGE')

        # Verify positions data structure
        positions = checkpoint_dict.get('positions', {})
        self.assertIsInstance(positions, dict)

    def test_to_checkpoint_dict_rpc_method(self):
        """
        Test that ChallengePeriodManager has to_checkpoint_dict method.

        This is a regression test for production errors where RPC methods were missing.
        """
        self.assertTrue(
            hasattr(self.challenge_period_client, 'to_checkpoint_dict'),
            "ChallengePeriodManager missing to_checkpoint_dict method"
        )

        # Verify it's callable and returns correct structure
        checkpoint_dict = self.challenge_period_client.to_checkpoint_dict()
        self.assertIsInstance(checkpoint_dict, dict)

        # Verify our test miners are in the dict
        for hotkey in self.test_hotkeys:
            self.assertIn(hotkey, checkpoint_dict)

    def test_generate_request_core_skip_file_creation(self):
        """Test generate_request_core with create_production_files=False."""
        checkpoint_dict = self.core_outputs_client.generate_request_core(
            create_production_files=False,
            save_production_files=False,
            upload_production_files=False
        )

        # Should still return a dict
        self.assertIsNotNone(checkpoint_dict)
        self.assertIsInstance(checkpoint_dict, dict)

    def test_get_compressed_checkpoint_from_memory(self):
        """Test retrieving compressed checkpoint from memory cache."""
        # First generate a checkpoint to potentially populate the cache
        self.core_outputs_client.generate_request_core(
            create_production_files=True,
            save_production_files=False,
            upload_production_files=False
        )

        # Try to retrieve compressed checkpoint
        compressed = self.core_outputs_client.get_compressed_checkpoint_from_memory()

        # May be None if cache not populated (which is OK for tests)
        # The important thing is it doesn't raise an error
        self.assertIsInstance(compressed, (bytes, type(None)))

    # ==================== Integration Test ====================

    def test_full_production_pipeline(self):
        """
        Integration test: Simulate full production pipeline.

        This test exercises the complete code path that runs in production
        when the validator generates checkpoint data.
        """
        current_time_ms = TimeUtil.now_in_millis()

        # Step 1: Generate checkpoint (production code path)
        try:
            checkpoint_dict = self.core_outputs_client.generate_request_core(
                create_production_files=True,
                save_production_files=False,
                upload_production_files=False
            )
        except Exception as e:
            self.fail(f"Production pipeline failed at checkpoint generation: {e}")

        # Verify checkpoint was created successfully
        self.assertIsNotNone(checkpoint_dict)
        self.assertIn('challengeperiod', checkpoint_dict)
        self.assertIn('positions', checkpoint_dict)
        self.assertIn('miner_account_sizes', checkpoint_dict)

        # Verify data integrity
        challengeperiod = checkpoint_dict.get('challengeperiod', {})
        self.assertGreater(len(challengeperiod), 0, "Challengeperiod should contain test miners")

        # Verify all our test miners made it through the pipeline
        for hotkey in self.test_hotkeys:
            self.assertIn(hotkey, challengeperiod,
                         f"Test miner {hotkey} should be in production output")


if __name__ == '__main__':
    unittest.main()
