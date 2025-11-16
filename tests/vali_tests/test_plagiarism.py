# Copyright © 2024 Taoshi Inc
import unittest
from unittest.mock import Mock, patch, MagicMock

from miner_objects.slack_notifier import SlackNotifier
from shared_objects.mock_metagraph import MockMetagraph
from tests.vali_tests.base_objects.test_base import TestBase
from vali_objects.utils.challengeperiod_manager import ChallengePeriodManager
from vali_objects.utils.elimination_manager import EliminationManager, EliminationReason
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.utils.plagiarism_manager import PlagiarismManager
from vali_objects.utils.position_manager import PositionManager
from vali_objects.vali_config import ValiConfig
from time_util.time_util import TimeUtil


class TestPlagiarism(TestBase):

    def setUp(self):
        super().setUp()
        self.MINER_HOTKEY1 = "test_miner1"
        self.MINER_HOTKEY2 = "test_miner2"
        self.MINER_HOTKEY3 = "test_miner3"
        self.PLAGIARISM_HOTKEY = "plagiarism_miner"
        self.current_time = TimeUtil.now_in_millis()

        self.mock_metagraph = MockMetagraph([
            self.MINER_HOTKEY1,
            self.MINER_HOTKEY2,
            self.MINER_HOTKEY3,
            self.PLAGIARISM_HOTKEY
        ])

        # Mock SlackNotifier
        self.mock_slack_notifier = Mock(spec=SlackNotifier)

        # Create PlagiarismManager
        self.plagiarism_manager = PlagiarismManager(
            slack_notifier=self.mock_slack_notifier,
            running_unit_tests=True
        )

        # Mock dependencies for ChallengePeriodManager
        self.mock_position_manager = Mock(spec=PositionManager)
        self.mock_elimination_manager = Mock(spec=EliminationManager)
        self.mock_position_manager.elimination_manager = self.mock_elimination_manager

        # Create ChallengePeriodManager with mocked dependencies
        self.challenge_manager = ChallengePeriodManager(
            metagraph=self.mock_metagraph,
            position_manager=self.mock_position_manager,
            plagiarism_manager=self.plagiarism_manager,
            running_unit_tests=True
        )

        # Initialize active miners using update_miners API
        self.challenge_manager.clear_all_miners()
        self.challenge_manager.update_miners({
            self.MINER_HOTKEY1: (MinerBucket.MAINCOMP, self.current_time, None, None),
            self.MINER_HOTKEY2: (MinerBucket.PROBATION, self.current_time, None, None),
            self.MINER_HOTKEY3: (MinerBucket.CHALLENGE, self.current_time, None, None),
            self.PLAGIARISM_HOTKEY: (MinerBucket.PLAGIARISM, self.current_time, MinerBucket.PROBATION, self.current_time - ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS)
        })

    def test_update_plagiarism_miners_new_plagiarists(self):
        """Test demotion of miners to plagiarism bucket when new plagiarists are detected"""
        # Mock the plagiarism manager to return new plagiarists
        mock_new_plagiarists = [self.MINER_HOTKEY1, self.MINER_HOTKEY2]
        mock_whitelisted = []

        self.plagiarism_manager.update_plagiarism_miners = Mock(
            return_value=(mock_new_plagiarists, mock_whitelisted)
        )

        initial_bucket = self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY1)
        self.assertEqual(initial_bucket, MinerBucket.MAINCOMP)

        # Call update_plagiarism_miners
        self.challenge_manager.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={}
        )

        # Verify miners were demoted to plagiarism
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY1), MinerBucket.PLAGIARISM)
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY2), MinerBucket.PLAGIARISM)

    def test_update_plagiarism_miners_whitelisted_promotion(self):
        """Test promotion of miners from plagiarism to probation when whitelisted"""
        # Mock the plagiarism manager to return whitelisted miners
        mock_new_plagiarists = []
        mock_whitelisted = [self.PLAGIARISM_HOTKEY]

        self.plagiarism_manager.update_plagiarism_miners = Mock(
            return_value=(mock_new_plagiarists, mock_whitelisted)
        )

        initial_bucket = self.challenge_manager.get_miner_bucket(self.PLAGIARISM_HOTKEY)
        self.assertEqual(initial_bucket, MinerBucket.PLAGIARISM)

        # Call update_plagiarism_miners
        self.challenge_manager.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={self.PLAGIARISM_HOTKEY: self.current_time}
        )

        # Verify miner was promoted from plagiarism to probation
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PROBATION)

    def test_prepare_plagiarism_elimination_miners(self):
        """Test elimination of plagiarism miners who exceed review period"""
        # Set up plagiarism manager to return miners for elimination
        elimination_time = self.current_time
        miners_to_eliminate = {self.PLAGIARISM_HOTKEY: elimination_time}

        self.plagiarism_manager.plagiarism_miners_to_eliminate = Mock(
            return_value=miners_to_eliminate
        )

        # Call prepare_plagiarism_elimination_miners
        result = self.challenge_manager.prepare_plagiarism_elimination_miners(
            current_time=self.current_time
        )

        # Verify the result contains the miner with correct elimination reason
        expected_result = {
            self.PLAGIARISM_HOTKEY: (EliminationReason.PLAGIARISM.value, -1)
        }
        self.assertEqual(result, expected_result)

        # Verify plagiarism manager was called with correct time
        self.plagiarism_manager.plagiarism_miners_to_eliminate.assert_called_once_with(self.current_time)

    def test_prepare_plagiarism_elimination_miners_not_in_active(self):
        """Test that miners not in active_miners are not included in elimination"""
        non_active_miner = "non_active_miner"

        # Set up plagiarism manager to return a miner that's not in active_miners
        miners_to_eliminate = {non_active_miner: self.current_time}

        self.plagiarism_manager.plagiarism_miners_to_eliminate = Mock(
            return_value=miners_to_eliminate
        )

        # Call prepare_plagiarism_elimination_miners
        result = self.challenge_manager.prepare_plagiarism_elimination_miners(
            current_time=self.current_time
        )

        # Verify the result is empty since the miner is not in active_miners
        self.assertEqual(result, {})

    def test_demote_plagiarism_in_memory(self):
        """Test _demote_plagiarism_in_memory method directly"""
        hotkeys_to_demote = [self.MINER_HOTKEY1, self.MINER_HOTKEY2]

        # Verify initial states
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY1), MinerBucket.MAINCOMP)
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY2), MinerBucket.PROBATION)

        # Call the method
        self.challenge_manager._demote_plagiarism_in_memory(hotkeys_to_demote, self.current_time)

        # Verify miners were demoted to plagiarism
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY1), MinerBucket.PLAGIARISM)
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY2), MinerBucket.PLAGIARISM)

        # Verify timestamps were updated
        timestamp1 = self.challenge_manager.get_miner_start_time(self.MINER_HOTKEY1)
        timestamp2 = self.challenge_manager.get_miner_start_time(self.MINER_HOTKEY2)
        self.assertEqual(timestamp1, self.current_time)
        self.assertEqual(timestamp2, self.current_time)

    def test_promote_plagiarism_to_previous_bucket_in_memory(self):
        """Test _promote_plagiarism_to_previous_bucket_in_memory method directly"""
        hotkeys_to_promote = [self.PLAGIARISM_HOTKEY]

        # Verify initial state
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PLAGIARISM)

        # Call the method
        self.challenge_manager._promote_plagiarism_to_previous_bucket_in_memory(hotkeys_to_promote, self.current_time)

        # Verify miner was promoted to probation
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PROBATION)

        # Verify timestamp was updated
        timestamp = self.challenge_manager.get_miner_start_time(self.PLAGIARISM_HOTKEY)
        self.assertEqual(timestamp, self.current_time - ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS)

    def test_update_plagiarism_miners_whitelisted_promotion_non_existant(self):
        """Test promotion of miners from plagiarism to probation when whitelisted and non_existant"""
        # This could occur if a miner that has already been eliminated is removed from the
        # eliminated miner list on the Plagiarism service for some reason. Ensure that errors don't occur
        # on PTN if this happens.

        # Mock the plagiarism manager to return whitelisted miners
        mock_new_plagiarists = []
        mock_whitelisted = ["non_existant"]

        self.plagiarism_manager.update_plagiarism_miners = Mock(
            return_value=(mock_new_plagiarists, mock_whitelisted)
        )

        initial_bucket = self.challenge_manager.get_miner_bucket("non_existant")
        self.assertEqual(initial_bucket, None)

        # Call update_plagiarism_miners
        self.challenge_manager.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={self.PLAGIARISM_HOTKEY: self.current_time}
        )

        # Verify miner still doesn't have a bucket (i.e., not in active miners)
        self.assertEqual(self.challenge_manager.get_miner_bucket("non_existant"), None)


    def test_demote_plagiarism_empty_list(self):
        """Test demoting with empty list of hotkeys"""
        # Call with empty list
        self.challenge_manager._demote_plagiarism_in_memory([], self.current_time)

        # Verify all miners remain in their original buckets
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY1), MinerBucket.MAINCOMP)
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY2), MinerBucket.PROBATION)
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY3), MinerBucket.CHALLENGE)

    def test_promote_plagiarism_empty_list(self):
        """Test promoting with empty list of hotkeys"""
        # Call with empty list
        self.challenge_manager._promote_plagiarism_to_previous_bucket_in_memory([], self.current_time)

        # Verify plagiarism miner remains in plagiarism bucket
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.PLAGIARISM_HOTKEY), MinerBucket.PLAGIARISM)

    def test_slack_notifications_disabled_during_tests(self):
        """Test that slack notifications are disabled during unit tests"""
        # Call notification methods directly on plagiarism manager
        self.plagiarism_manager.send_plagiarism_demotion_notification(self.MINER_HOTKEY1)
        self.plagiarism_manager.send_plagiarism_promotion_notification(self.MINER_HOTKEY1)
        self.plagiarism_manager.send_plagiarism_elimination_notification(self.MINER_HOTKEY1)

        # Verify slack notifier methods were not called since running_unit_tests=True
        self.mock_slack_notifier.send_plagiarism_demotion_notification.assert_not_called()
        self.mock_slack_notifier.send_plagiarism_promotion_notification.assert_not_called()

    def test_get_bucket_methods(self):
        """Test helper methods for getting miners by bucket"""
        # Test getting plagiarism miners
        plagiarism_miners = self.challenge_manager.get_plagiarism_miners()
        expected_plagiarism = {self.PLAGIARISM_HOTKEY: self.current_time}
        self.assertEqual(plagiarism_miners, expected_plagiarism)

        # Test getting maincomp miners
        maincomp_miners = self.challenge_manager.get_success_miners()
        expected_maincomp = {self.MINER_HOTKEY1: self.current_time}
        self.assertEqual(maincomp_miners, expected_maincomp)

        # Test getting probation miners
        probation_miners = self.challenge_manager.get_probation_miners()
        expected_probation = {self.MINER_HOTKEY2: self.current_time}
        self.assertEqual(probation_miners, expected_probation)

    def test_integration_full_plagiarism_flow(self):
        """Integration test for the complete plagiarism flow: demotion -> promotion -> elimination"""
        # Step 1: Test demotion (new plagiarist detected)
        mock_new_plagiarists = [self.MINER_HOTKEY3]  # Challenge miner becomes plagiarist
        mock_whitelisted = []

        self.plagiarism_manager.update_plagiarism_miners = Mock(
            return_value=(mock_new_plagiarists, mock_whitelisted)
        )

        # Update plagiarism miners (demotion)
        self.challenge_manager.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={}
        )

        # Verify demotion
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY3), MinerBucket.PLAGIARISM)

        # Step 2: Test promotion (plagiarist is whitelisted)
        mock_new_plagiarists = []
        mock_whitelisted = [self.MINER_HOTKEY3]

        self.plagiarism_manager.update_plagiarism_miners = Mock(
            return_value=(mock_new_plagiarists, mock_whitelisted)
        )

        # Update plagiarism miners (promotion)
        self.challenge_manager.update_plagiarism_miners(
            current_time=self.current_time,
            plagiarism_miners={self.MINER_HOTKEY3: self.current_time}
        )

        # Verify promotion to probation
        self.assertEqual(self.challenge_manager.get_miner_bucket(self.MINER_HOTKEY3), MinerBucket.CHALLENGE)

        # Step 3: Demote back to plagiarism for elimination test
        self.challenge_manager._demote_plagiarism_in_memory([self.MINER_HOTKEY3], self.current_time)

        # Step 4: Test elimination (plagiarist exceeds review period)
        miners_to_eliminate = {self.MINER_HOTKEY3: self.current_time}
        self.plagiarism_manager.plagiarism_miners_to_eliminate = Mock(
            return_value=miners_to_eliminate
        )

        elimination_result = self.challenge_manager.prepare_plagiarism_elimination_miners(
            current_time=self.current_time
        )

        # Verify elimination preparation
        expected_elimination = {
            self.MINER_HOTKEY3: (EliminationReason.PLAGIARISM.value, -1)
        }
        self.assertEqual(elimination_result, expected_elimination)

        # Apply elimination
        self.challenge_manager._eliminate_challengeperiod_in_memory(elimination_result)

        # Verify miner was eliminated
        self.assertFalse(self.challenge_manager.has_miner(self.MINER_HOTKEY3))


if __name__ == '__main__':
    unittest.main()