# Copyright © 2024 Taoshi Inc
import unittest
from unittest.mock import Mock, patch, MagicMock

from shared_objects.server_orchestrator import ServerOrchestrator, ServerMode
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.elimination_manager import EliminationReason
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig
from time_util.time_util import TimeUtil


class TestPlagiarism(TestBase):
    """
    Plagiarism tests using ServerOrchestrator for shared server infrastructure.

    Servers start once (via singleton orchestrator) and are shared across all test classes.
    Per-test isolation is achieved by clearing data state (not restarting servers).
    """

    # Class-level references (set in setUpClass via ServerOrchestrator)
    orchestrator = None
    metagraph_client = None
    challenge_period_client = None
    plagiarism_client = None
    elimination_client = None

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
        cls.metagraph_client = cls.orchestrator.get_client('metagraph')
        cls.challenge_period_client = cls.orchestrator.get_client('challenge_period')
        cls.plagiarism_client = cls.orchestrator.get_client('plagiarism')
        cls.elimination_client = cls.orchestrator.get_client('elimination')

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

        # Test miner hotkeys
        self.MINER_HOTKEY1 = "test_miner1"
        self.MINER_HOTKEY2 = "test_miner2"
        self.MINER_HOTKEY3 = "test_miner3"
        self.PLAGIARISM_HOTKEY = "plagiarism_miner"
        self.current_time = TimeUtil.now_in_millis()

        # Set up metagraph with test miners
        self.metagraph_client.set_hotkeys([
            self.MINER_HOTKEY1,
            self.MINER_HOTKEY2,
            self.MINER_HOTKEY3,
            self.PLAGIARISM_HOTKEY
        ])

        # Initialize active miners using update_miners API
        self.challenge_period_client.clear_all_miners()
        self.challenge_period_client.update_miners({
            self.MINER_HOTKEY1: (MinerBucket.MAINCOMP, self.current_time, None, None),
            self.MINER_HOTKEY2: (MinerBucket.PROBATION, self.current_time, None, None),
            self.MINER_HOTKEY3: (MinerBucket.CHALLENGE, self.current_time, None, None),
            self.PLAGIARISM_HOTKEY: (MinerBucket.PLAGIARISM, self.current_time, MinerBucket.PROBATION, self.current_time - ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS)
        })
        self.challenge_period_client._write_challengeperiod_from_memory_to_disk()

    def tearDown(self):
        """Per-test teardown: Clear data for next test."""
        self.orchestrator.clear_all_test_data()

    def test_update_plagiarism_miners_new_plagiarists(self):
        """Test demotion of miners to plagiarism bucket when new plagiarists are detected"""
        # Inject plagiarism data via client - mark miners as plagiarists
        plagiarism_data = {
            self.MINER_HOTKEY1: {"time": self.current_time},
            self.MINER_HOTKEY2: {"time": self.current_time}
        }
        self.plagiarism_client.set_plagiarism_miners_for_test(plagiarism_data, self.current_time)

        initial_bucket = self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY1)
        self.assertEqual(initial_bucket, MinerBucket.MAINCOMP)

        # Call update_plagiarism_miners via client
        self.challenge_period_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={}
        )

        # Verify miners were demoted to plagiarism
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY1), MinerBucket.PLAGIARISM)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY2), MinerBucket.PLAGIARISM)

    def test_update_plagiarism_miners_whitelisted_promotion(self):
        """Test promotion of miners from plagiarism to probation when whitelisted"""
        # Clear plagiarism data (empty = whitelisted)
        self.plagiarism_client.set_plagiarism_miners_for_test({}, self.current_time)

        initial_bucket = self.challenge_period_client.get_miner_bucket(self.PLAGIARISM_HOTKEY)
        self.assertEqual(initial_bucket, MinerBucket.PLAGIARISM)

        # Call update_plagiarism_miners via client
        self.challenge_period_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={self.PLAGIARISM_HOTKEY: self.current_time}
        )

        # Verify miner was promoted from plagiarism to probation
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PROBATION)

    def test_prepare_plagiarism_elimination_miners(self):
        """Test elimination of plagiarism miners who exceed review period"""
        # Inject plagiarism data that should trigger elimination (old timestamp)
        old_time = self.current_time - ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS - 1000
        plagiarism_data = {self.PLAGIARISM_HOTKEY: {"time": old_time}}
        self.plagiarism_client.set_plagiarism_miners_for_test(plagiarism_data, old_time)

        # Call prepare_plagiarism_elimination_miners via client
        result = self.challenge_period_client.prepare_plagiarism_elimination_miners(
            current_time=self.current_time
        )

        # Verify the result contains the miner with correct elimination reason
        expected_result = {
            self.PLAGIARISM_HOTKEY: (EliminationReason.PLAGIARISM.value, -1)
        }
        self.assertEqual(result, expected_result)

    def test_prepare_plagiarism_elimination_miners_not_in_active(self):
        """Test that miners not in active_miners are not included in elimination"""
        non_active_miner = "non_active_miner"

        # Inject plagiarism data for a miner that's not in active miners
        old_time = self.current_time - ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS - 1000
        plagiarism_data = {non_active_miner: {"time": old_time}}
        self.plagiarism_client.set_plagiarism_miners_for_test(plagiarism_data, old_time)

        # Call prepare_plagiarism_elimination_miners via client
        result = self.challenge_period_client.prepare_plagiarism_elimination_miners(
            current_time=self.current_time
        )

        # Verify the result is empty since the miner is not in active_miners
        self.assertEqual(result, {})

    def test_demote_plagiarism_in_memory(self):
        """Test demotion behavior via public update_plagiarism_miners API"""
        hotkeys_to_demote = [self.MINER_HOTKEY1, self.MINER_HOTKEY2]

        # Verify initial states
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY1), MinerBucket.MAINCOMP)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY2), MinerBucket.PROBATION)

        # Inject plagiarism data for these miners
        plagiarism_data = {
            self.MINER_HOTKEY1: {"time": self.current_time},
            self.MINER_HOTKEY2: {"time": self.current_time}
        }
        self.plagiarism_client.set_plagiarism_miners_for_test(plagiarism_data, self.current_time)

        # Call update_plagiarism_miners which internally calls _demote_plagiarism_in_memory
        self.challenge_period_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={}
        )

        # Verify miners were demoted to plagiarism
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY1), MinerBucket.PLAGIARISM)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY2), MinerBucket.PLAGIARISM)

        # Verify timestamps were updated
        timestamp1 = self.challenge_period_client.get_miner_start_time(self.MINER_HOTKEY1)
        timestamp2 = self.challenge_period_client.get_miner_start_time(self.MINER_HOTKEY2)
        self.assertEqual(timestamp1, self.current_time)
        self.assertEqual(timestamp2, self.current_time)

    def test_promote_plagiarism_to_previous_bucket_in_memory(self):
        """Test promotion behavior via public update_plagiarism_miners API"""
        hotkeys_to_promote = [self.PLAGIARISM_HOTKEY]

        # Verify initial state
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PLAGIARISM)

        # Clear plagiarism data to trigger promotion (miner no longer flagged as plagiarist)
        self.plagiarism_client.set_plagiarism_miners_for_test({}, self.current_time)

        # Call update_plagiarism_miners which internally calls _promote_plagiarism_to_previous_bucket_in_memory
        self.challenge_period_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={self.PLAGIARISM_HOTKEY: self.current_time}
        )

        # Verify miner was promoted to probation
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PROBATION)

        # Verify timestamp was updated
        timestamp = self.challenge_period_client.get_miner_start_time(self.PLAGIARISM_HOTKEY)
        self.assertEqual(timestamp, self.current_time - ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS)

    def test_update_plagiarism_miners_whitelisted_promotion_non_existant(self):
        """Test promotion of miners from plagiarism to probation when whitelisted and non_existant"""
        # This could occur if a miner that has already been eliminated is removed from the
        # eliminated miner list on the Plagiarism service for some reason. Ensure that errors don't occur
        # on PTN if this happens.

        # Clear plagiarism data (empty = whitelisted)
        self.plagiarism_client.set_plagiarism_miners_for_test({}, self.current_time)

        initial_bucket = self.challenge_period_client.get_miner_bucket("non_existant")
        self.assertEqual(initial_bucket, None)

        # Call update_plagiarism_miners via client
        self.challenge_period_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={self.PLAGIARISM_HOTKEY: self.current_time}
        )

        # Verify miner still doesn't have a bucket (i.e., not in active miners)
        self.assertEqual(self.challenge_period_client.get_miner_bucket("non_existant"), None)


    def test_demote_plagiarism_empty_list(self):
        """Test demoting with empty list of hotkeys (no plagiarists detected)"""
        # Don't inject any plagiarism data (empty = no plagiarists)
        self.plagiarism_client.set_plagiarism_miners_for_test({}, self.current_time)

        # Call update_plagiarism_miners
        self.challenge_period_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={}
        )

        # Verify all miners remain in their original buckets
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY1), MinerBucket.MAINCOMP)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY2), MinerBucket.PROBATION)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY3), MinerBucket.CHALLENGE)

    def test_promote_plagiarism_empty_list(self):
        """Test promoting with empty list of hotkeys (no miners to promote)"""
        # Inject plagiarism data (miner remains flagged, so no promotion)
        plagiarism_data = {self.PLAGIARISM_HOTKEY: {"time": self.current_time}}
        self.plagiarism_client.set_plagiarism_miners_for_test(plagiarism_data, self.current_time)

        # Call update_plagiarism_miners
        self.challenge_period_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={self.PLAGIARISM_HOTKEY: self.current_time}
        )

        # Verify plagiarism miner remains in plagiarism bucket
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PLAGIARISM)

    def test_slack_notifications_disabled_during_tests(self):
        """Test that slack notifications are disabled during unit tests"""
        # Note: With ServerOrchestrator, we can't directly test slack notifications
        # as the plagiarism manager runs in a separate process via RPC.
        # This test verifies that running_unit_tests=True is respected in the server.

        # Trigger demotion (which would send notification if not in test mode)
        plagiarism_data = {self.MINER_HOTKEY1: {"time": self.current_time}}
        self.plagiarism_client.set_plagiarism_miners_for_test(plagiarism_data, self.current_time)
        self.challenge_period_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={}
        )

        # If this completes without errors, slack notifications are properly disabled
        # (No way to verify mock calls across RPC boundary, but test ensures no crashes)

    def test_get_bucket_methods(self):
        """Test helper methods for getting miners by bucket"""
        # Test getting plagiarism miners
        plagiarism_miners = self.challenge_period_client.get_plagiarism_miners()
        expected_plagiarism = {self.PLAGIARISM_HOTKEY: self.current_time}
        self.assertEqual(plagiarism_miners, expected_plagiarism)

        # Test getting maincomp miners
        maincomp_miners = self.challenge_period_client.get_success_miners()
        expected_maincomp = {self.MINER_HOTKEY1: self.current_time}
        self.assertEqual(maincomp_miners, expected_maincomp)

        # Test getting probation miners
        probation_miners = self.challenge_period_client.get_probation_miners()
        expected_probation = {self.MINER_HOTKEY2: self.current_time}
        self.assertEqual(probation_miners, expected_probation)

    def test_integration_full_plagiarism_flow(self):
        """Integration test for the complete plagiarism flow: demotion -> promotion -> elimination"""
        # Step 1: Test demotion (new plagiarist detected)
        plagiarism_data = {self.MINER_HOTKEY3: {"time": self.current_time}}
        self.plagiarism_client.set_plagiarism_miners_for_test(plagiarism_data, self.current_time)

        # Update plagiarism miners (demotion) via client
        self.challenge_period_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={}
        )

        # Verify demotion
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY3), MinerBucket.PLAGIARISM)

        # Step 2: Test promotion (plagiarist is whitelisted)
        # Clear plagiarism data (empty = whitelisted)
        self.plagiarism_client.set_plagiarism_miners_for_test({}, self.current_time)

        # Update plagiarism miners (promotion) via client
        self.challenge_period_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={self.MINER_HOTKEY3: self.current_time}
        )

        # Verify promotion to original bucket (CHALLENGE)
        self.assertEqual(self.challenge_period_client.get_miner_bucket(self.MINER_HOTKEY3), MinerBucket.CHALLENGE)

        # Step 3: Demote back to plagiarism for elimination test
        plagiarism_data = {self.MINER_HOTKEY3: {"time": self.current_time}}
        self.plagiarism_client.set_plagiarism_miners_for_test(plagiarism_data, self.current_time)
        self.challenge_period_client.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={}
        )

        # Step 4: Test elimination (plagiarist exceeds review period)
        # Inject plagiarism data with old timestamp to trigger elimination
        old_time = self.current_time - ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS - 1000
        plagiarism_data = {self.MINER_HOTKEY3: {"time": old_time}}
        self.plagiarism_client.set_plagiarism_miners_for_test(plagiarism_data, old_time)

        elimination_result = self.challenge_period_client.prepare_plagiarism_elimination_miners(
            current_time=self.current_time
        )

        # Verify elimination preparation
        expected_elimination = {
            self.MINER_HOTKEY3: (EliminationReason.PLAGIARISM.value, -1)
        }
        self.assertEqual(elimination_result, expected_elimination)

        # Apply elimination via elimination client
        for hotkey, (reason, timestamp) in elimination_result.items():
            self.elimination_client.append_elimination_row(hotkey, timestamp, reason)

        # Remove from challenge period (pass None to fetch from elimination_manager)
        self.challenge_period_client.remove_eliminated(eliminations=None)

        # Verify miner was eliminated
        self.assertFalse(self.challenge_period_client.has_miner(self.MINER_HOTKEY3))


if __name__ == '__main__':
    unittest.main()