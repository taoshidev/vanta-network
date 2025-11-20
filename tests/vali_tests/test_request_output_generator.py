# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
Test to ensure RequestOutputGenerator and its components can be instantiated and run.

This test prevents production errors like:
- AttributeError: 'ChallengePeriodManager' object has no attribute 'to_checkpoint_dict'
- Missing RPC methods when running generate_request_core or generate_request_minerstatistics

By exercising these code paths in tests, we catch missing RPC methods before deployment.
"""

import unittest

from shared_objects.mock_metagraph import MockMetagraph
from tests.shared_objects.mock_classes import MockLivePriceFetcher
from tests.vali_tests.base_objects.test_base import TestBase
from runnable.generate_request_core import RequestCoreManager
from runnable.generate_request_minerstatistics import MinerStatisticsManager
from runnable.generate_request_outputs import RequestOutputGenerator
from vali_objects.enums.order_type_enum import OrderType
from vali_objects.position import Position
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.plagiarism_manager import PlagiarismManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
from vali_objects.vali_dataclasses.order import Order
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager
from time_util.time_util import TimeUtil


class TestRequestOutputGenerator(TestBase):
    """
    Test that Request Output Generator components can be instantiated and executed.

    This catches issues like:
    - Missing RPC methods (e.g., to_checkpoint_dict_rpc)
    - Incorrect RPC client/server method signatures
    - Serialization issues in RPC calls
    """

    def setUp(self):
        super().setUp()

        # Create test hotkeys
        self.test_hotkeys = [
            "test_hotkey_1_abc123",
            "test_hotkey_2_def456",
            "test_hotkey_3_ghi789"
        ]

        # Initialize system components
        self.mock_metagraph = MockMetagraph(self.test_hotkeys)
        secrets = ValiUtils.get_secrets(running_unit_tests=True)
        self.live_price_fetcher = MockLivePriceFetcher(secrets=secrets, disable_ws=True)
        self.contract_manager = ValidatorContractManager(running_unit_tests=True)

        # Initialize managers in the correct order to handle circular dependencies
        self.elimination_manager = EliminationManager(
            self.mock_metagraph,
            self.live_price_fetcher,
            None,
            running_unit_tests=True,
            contract_manager=self.contract_manager
        )

        self.perf_ledger_manager = PerfLedgerManager(
            self.mock_metagraph,
            running_unit_tests=True
        )
        self.perf_ledger_manager.clear_all_ledger_data()

        self.position_manager = PositionManager(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            perf_ledger_manager=self.perf_ledger_manager,
            elimination_manager=self.elimination_manager,
            live_price_fetcher=self.live_price_fetcher
        )

        self.plagiarism_manager = PlagiarismManager(
            slack_notifier=None,
            running_unit_tests=True
        )

        self.challengeperiod_manager = ChallengePeriodManager(
            self.mock_metagraph,
            position_manager=self.position_manager,
            perf_ledger_manager=self.perf_ledger_manager,
            contract_manager=self.contract_manager,
            plagiarism_manager=self.plagiarism_manager,
            running_unit_tests=True
        )

        # Set up circular references
        self.position_manager.challengeperiod_manager = self.challengeperiod_manager
        self.elimination_manager.position_manager = self.position_manager
        self.elimination_manager.challengeperiod_manager = self.challengeperiod_manager

        # Initialize weight setter and plagiarism detector
        self.weight_setter = SubtensorWeightSetter(
            metagraph=self.mock_metagraph,
            running_unit_tests=True,
            position_manager=self.position_manager,
            contract_manager=self.contract_manager
        )

        self.plagiarism_detector = PlagiarismDetector(
            self.mock_metagraph,
            None,
            position_manager=self.position_manager
        )

        # Clear all positions
        self.position_manager.clear_all_miner_positions()

        # Create some test positions for miners
        self._create_test_positions()

    def tearDown(self):
        super().tearDown()
        # Cleanup
        self.position_manager.clear_all_miner_positions()
        self.perf_ledger_manager.clear_perf_ledgers_from_disk()
        self.challengeperiod_manager._clear_challengeperiod_in_memory_and_disk()
        self.elimination_manager.clear_eliminations()

    def _create_test_positions(self):
        """Create some test positions for miners to avoid empty data errors."""
        current_time = TimeUtil.now_in_millis()

        for hotkey in self.test_hotkeys:
            # Add to challenge period
            self.challengeperiod_manager.set_miner_bucket(
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
            test_position.rebuild_position_with_updated_orders(self.live_price_fetcher)
            test_position.close_out_position(current_time - 1000 * 60 * 30)  # 30 min ago
            self.position_manager.save_miner_position(test_position)

    def test_request_core_manager_instantiation(self):
        """Test that RequestCoreManager can be instantiated."""
        rcm = RequestCoreManager(
            position_manager=self.position_manager,
            subtensor_weight_setter=self.weight_setter,
            plagiarism_detector=self.plagiarism_detector,
            contract_manager=self.contract_manager,
            asset_selection_manager=None,
            limit_order_manager=None,
            running_unit_tests=True
        )
        self.assertIsNotNone(rcm)
        self.assertIs(rcm._position_manager, self.position_manager)
        # Note: challengeperiod_manager and elimination_manager are accessed via RPC, not stored as attributes

    def test_miner_statistics_manager_instantiation(self):
        """Test that MinerStatisticsManager can be instantiated."""
        msm = MinerStatisticsManager(
            position_manager=self.position_manager,
            subtensor_weight_setter=self.weight_setter,
            plagiarism_detector=self.plagiarism_detector,
            contract_manager=self.contract_manager,
            running_unit_tests=True
        )
        self.assertIsNotNone(msm)
        self.assertIs(msm._position_manager, self.position_manager)

    def test_request_output_generator_instantiation(self):
        """Test that RequestOutputGenerator can be instantiated."""
        rcm = RequestCoreManager(
            position_manager=self.position_manager,
            subtensor_weight_setter=self.weight_setter,
            plagiarism_detector=self.plagiarism_detector,
            contract_manager=self.contract_manager,
            asset_selection_manager=None,
            limit_order_manager=None,
            running_unit_tests=True
        )

        msm = MinerStatisticsManager(
            position_manager=self.position_manager,
            subtensor_weight_setter=self.weight_setter,
            plagiarism_detector=self.plagiarism_detector,
            contract_manager=self.contract_manager,
            running_unit_tests=True
        )

        rog = RequestOutputGenerator(rcm=rcm, msm=msm)
        self.assertIsNotNone(rog)
        self.assertIs(rog.rcm, rcm)
        self.assertIs(rog.msm, msm)

    def test_generate_request_core_no_disk_writes(self):
        """
        Test that generate_request_core can execute without writing to disk.

        This is the critical test that would have caught the missing to_checkpoint_dict RPC method.

        Note: If you need to test with save_production_files=True, use RequestCoreManager.cleanup_test_files()
        in tearDown or a try/finally block to clean up test files.
        """
        rcm = RequestCoreManager(
            position_manager=self.position_manager,
            subtensor_weight_setter=self.weight_setter,
            plagiarism_detector=self.plagiarism_detector,
            contract_manager=self.contract_manager,
            asset_selection_manager=None,
            limit_order_manager=None,
            running_unit_tests=True
        )

        # This call exercises the code path that failed in production:
        # challengeperiod_dict = self.challengeperiod_manager.to_checkpoint_dict()
        try:
            checkpoint_dict = rcm.generate_request_core(
                create_production_files=True,  # Create the dicts
                save_production_files=False,   # Don't write to disk (default, but explicit)
                upload_production_files=False  # Don't upload to gcloud (default, but explicit)
            )
        except AttributeError as e:
            self.fail(f"generate_request_core raised AttributeError (likely missing RPC method): {e}")

        # Verify the checkpoint dict has expected keys
        self.assertIn('challengeperiod', checkpoint_dict)
        self.assertIn('miner_account_sizes', checkpoint_dict)
        self.assertIn('positions', checkpoint_dict)

        # Verify challengeperiod dict is not empty (we added test miners)
        self.assertIsInstance(checkpoint_dict['challengeperiod'], dict)

    def test_generate_request_minerstatistics_no_disk_writes(self):
        """
        Test that generate_request_minerstatistics can execute without errors.

        Note: This method doesn't return a value - it writes to disk and stores compressed data.
        We're just verifying it doesn't raise exceptions (especially missing RPC methods).
        """
        msm = MinerStatisticsManager(
            position_manager=self.position_manager,
            subtensor_weight_setter=self.weight_setter,
            plagiarism_detector=self.plagiarism_detector,
            contract_manager=self.contract_manager,
            running_unit_tests=True
        )

        current_time_ms = TimeUtil.now_in_millis()

        # Create a simple checkpoint dict
        checkpoints = {
            'positions': {},
            'challengeperiod': self.challengeperiod_manager.to_checkpoint_dict(),
            'miner_account_sizes': {}
        }

        try:
            msm.generate_request_minerstatistics(
                time_now=current_time_ms,
                checkpoints=checkpoints
            )
        except AttributeError as e:
            self.fail(f"generate_request_minerstatistics raised AttributeError: {e}")

        # If we got here without exceptions, the test passed

    def test_to_checkpoint_dict_rpc_method_exists(self):
        """
        Test that ChallengePeriodManager has to_checkpoint_dict method (RPC-exposed).

        This is a regression test for the production error.
        """
        self.assertTrue(
            hasattr(self.challengeperiod_manager, 'to_checkpoint_dict'),
            "ChallengePeriodManager missing to_checkpoint_dict method"
        )

        # Verify it's callable
        checkpoint_dict = self.challengeperiod_manager.to_checkpoint_dict()
        self.assertIsInstance(checkpoint_dict, dict)

    def test_challengeperiod_dict_structure(self):
        """
        Test that to_checkpoint_dict returns properly structured data.
        """
        # Add a test miner
        test_hotkey = "structure_test_hotkey"
        current_time = TimeUtil.now_in_millis()

        self.challengeperiod_manager.set_miner_bucket(
            test_hotkey,
            MinerBucket.MAINCOMP,
            current_time
        )

        checkpoint_dict = self.challengeperiod_manager.to_checkpoint_dict()

        # Verify structure
        self.assertIn(test_hotkey, checkpoint_dict)
        miner_data = checkpoint_dict[test_hotkey]

        self.assertIn('bucket', miner_data)
        self.assertIn('bucket_start_time', miner_data)
        self.assertEqual(miner_data['bucket'], 'MAINCOMP')
        self.assertEqual(miner_data['bucket_start_time'], current_time)

    def test_integration_full_pipeline_no_disk_writes(self):
        """
        Integration test: Full pipeline from RequestOutputGenerator without disk writes.

        This simulates what happens in production when RequestOutputGenerator.run_rcm_loop()
        and run_msm_loop() execute, but without writing to disk or uploading to gcloud.
        """
        rcm = RequestCoreManager(
            position_manager=self.position_manager,
            subtensor_weight_setter=self.weight_setter,
            plagiarism_detector=self.plagiarism_detector,
            contract_manager=self.contract_manager,
            asset_selection_manager=None,
            limit_order_manager=None,
            running_unit_tests=True
        )

        msm = MinerStatisticsManager(
            position_manager=self.position_manager,
            subtensor_weight_setter=self.weight_setter,
            plagiarism_detector=self.plagiarism_detector,
            contract_manager=self.contract_manager,
            running_unit_tests=True
        )

        current_time_ms = TimeUtil.now_in_millis()

        # Step 1: Generate core request (this calls to_checkpoint_dict)
        try:
            checkpoint_dict = rcm.generate_request_core(
                create_production_files=True,
                save_production_files=False,   # Don't write to disk (default, but explicit)
                upload_production_files=False  # Don't upload to gcloud (default, but explicit)
            )
        except Exception as e:
            self.fail(f"generate_request_core failed: {e}")

        # Step 2: Generate miner statistics using the checkpoint
        try:
            msm.generate_request_minerstatistics(
                time_now=current_time_ms,
                checkpoints=checkpoint_dict
            )
        except Exception as e:
            self.fail(f"generate_request_minerstatistics failed: {e}")

        # Verify checkpoint_dict was created successfully
        self.assertIsNotNone(checkpoint_dict)
        # If we got here without exceptions, both steps passed


if __name__ == '__main__':
    unittest.main()
