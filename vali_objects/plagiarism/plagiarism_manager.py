from typing import Dict

import requests

from shared_objects.slack_notifier import SlackNotifier
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig
import bittensor as bt


class PlagiarismManager:

    def __init__(self, slack_notifier: SlackNotifier, ipc_manager=None, running_unit_tests=False):
        self.refreshed_plagiarism_time_ms = 0
        self.plagiarism_miners = {} # hotkey -> elimination_time_ms
        self.slack_notifier = slack_notifier
        self.plagiarism_url = ValiConfig.PLAGIARISM_URL
        if ipc_manager:
            self.plagiarism_miners = ipc_manager.dict()
        else:
            self.plagiarism_miners = {}
        self.running_unit_tests = running_unit_tests

    def _check_plagiarism_refresh(self, current_time):
        return current_time - self.refreshed_plagiarism_time_ms > ValiConfig.PLAGIARISM_UPDATE_FREQUENCY_MS

    def plagiarism_miners_to_eliminate(self, current_time):
        """Returns a dict of miners that should be eliminated."""
        current_plagiarism_miners = self.get_plagiarism_elimination_scores(current_time)

        # If API call failed, return empty dict to maintain current state
        if current_plagiarism_miners is None:
            bt.logging.error("API call failed - cannot determine plagiarism eliminations")
            return {}

        miners_to_eliminate = {}
        for hotkey, plagiarism_data in current_plagiarism_miners.items():
            plagiarism_time = plagiarism_data["time"]
            if current_time - plagiarism_time > ValiConfig.PLAGIARISM_REVIEW_PERIOD_MS:
                miners_to_eliminate[hotkey] = current_time
        return miners_to_eliminate

    def update_plagiarism_miners(self, current_time: int, plagiarism_miners: Dict[str, MinerBucket]):

        # Get updated elimination miners from microservice
        current_plagiarism_miners = self.get_plagiarism_elimination_scores(current_time)

        # If API call failed, return empty lists to maintain current state
        if current_plagiarism_miners is None:
            bt.logging.error("API call failed - maintaining current plagiarism state")
            return [], []

        # The api is the source of truth
        # If a miner is no longer listed as a plagiarist, put them back in probation
        whitelisted_miners = []
        for miner in plagiarism_miners:
            if miner not in current_plagiarism_miners:
                whitelisted_miners.append(miner)

        # Miners that are now listed as plagiarists need to be updated
        new_plagiarism_miners = []
        for miner in current_plagiarism_miners:
            if miner not in plagiarism_miners:
                new_plagiarism_miners.append(miner)
        return new_plagiarism_miners, whitelisted_miners

    def _update_plagiarism_in_memory(self, current_time, plagiarism_miners):
        self.plagiarism_miners = plagiarism_miners
        self.refreshed_plagiarism_time_ms = current_time

    def get_plagiarism_elimination_scores(self, current_time, api_base_url=None):
        """
        Get elimination scores from the plagiarism API

        Args:
            api_base_url (str): Base URL of the API server

        Returns:
            dict: dict of elimination scores, or None if API error occurred
        """
        if api_base_url is None:
            api_base_url = self.plagiarism_url

        if self._check_plagiarism_refresh(current_time):
            try:
                response = requests.get(f"{api_base_url}/elimination_scores")
                response.raise_for_status()
                new_miners = response.json()

                if not isinstance(new_miners, dict):
                    raise ValueError(f"API returned invalid data type: expected dict, got: {new_miners} with type: {type(new_miners)}")

                bt.logging.info(f"Updating plagiarism api miners from {self.plagiarism_miners} to {new_miners}")
                self._update_plagiarism_in_memory(current_time, new_miners)
                return self.plagiarism_miners
            except Exception as e:
                print(f"Error fetching plagiarism elimination scores: {e}")
                return None
        else:
            return self.plagiarism_miners

    def send_plagiarism_demotion_notification(self, hotkey: str):
        """Send notification when a miner is demoted due to plagiarism"""
        if self.running_unit_tests:
            return
        self.slack_notifier.send_plagiarism_demotion_notification(hotkey)

    def send_plagiarism_promotion_notification(self, hotkey: str):
        """Send notification when a miner is promoted from plagiarism back to probation"""
        if self.running_unit_tests:
            return
        self.slack_notifier.send_plagiarism_promotion_notification(hotkey)

    def send_plagiarism_elimination_notification(self, hotkey: str):
        """Send notification when a miner is eliminated from plagiarism"""
        if self.running_unit_tests:
            return
        self.slack_notifier.send_plagiarism_elimination_notification(hotkey)