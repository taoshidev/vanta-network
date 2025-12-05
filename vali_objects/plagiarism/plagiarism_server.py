# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
PlagiarismServer - RPC server for plagiarism management.

This server runs in its own process and exposes plagiarism management via RPC.
Clients connect using PlagiarismClient.

Usage:
    # Validator spawns the server at startup
    from vali_objects.plagiarism.plagiarism_server import PlagiarismServer

    server = PlagiarismServer(
        slack_notifier=slack_notifier,
        start_server=True,
        start_daemon=False
    )

    # Other processes connect via PlagiarismClient
    from vali_objects.plagiarism.plagiarism_server import PlagiarismClient
    client = PlagiarismClient()  # Uses ValiConfig.RPC_PLAGIARISM_PORT
"""
from typing import Dict, Optional

import requests
import bittensor as bt

from shared_objects.slack_notifier import SlackNotifier
from shared_objects.rpc.rpc_server_base import RPCServerBase
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig, RPCConnectionMode


# ==================== Server Implementation ====================

class PlagiarismServer(RPCServerBase):
    """
    RPC server for plagiarism management.

    Inherits from RPCServerBase for unified RPC server and daemon infrastructure.

    All public methods ending in _rpc are exposed via RPC to PlagiarismClient.
    Internal state (plagiarism_miners) is kept local to this process.

    Architecture:
    - Runs in its own process (or thread in test mode)
    - Ports are obtained from ValiConfig
    """
    service_name = ValiConfig.RPC_PLAGIARISM_SERVICE_NAME
    service_port = ValiConfig.RPC_PLAGIARISM_PORT
    def __init__(
        self,
        slack_notifier: SlackNotifier = None,
        running_unit_tests: bool = False,
        start_server: bool = True,
        start_daemon: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize PlagiarismServer.

        Args:
            slack_notifier: SlackNotifier for alerts
            running_unit_tests: Whether running in test mode
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon (not used currently)
        """
        # Initialize RPCServerBase (handles RPC server lifecycle)
        # daemon_interval_s: 1 hour (plagiarism update frequency)
        # hang_timeout_s: Dynamically set to 2x interval to prevent false alarms during normal sleep
        daemon_interval_s = ValiConfig.PLAGIARISM_UPDATE_FREQUENCY_MS / 1000.0  # 1 hour (3600s)
        hang_timeout_s = daemon_interval_s * 2.0  # 2 hours (2x interval)

        super().__init__(
            service_name=ValiConfig.RPC_PLAGIARISM_SERVICE_NAME,
            port=ValiConfig.RPC_PLAGIARISM_PORT,
            connection_mode=connection_mode,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=start_daemon,
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=hang_timeout_s
        )
        self.running_unit_tests = running_unit_tests
        self.slack_notifier = slack_notifier
        self.plagiarism_url = ValiConfig.PLAGIARISM_URL

        # Local state (no IPC)
        self.refreshed_plagiarism_time_ms = 0
        self.plagiarism_miners: Dict[str, dict] = {}

        bt.logging.success(f"PlagiarismServer initialized on port {ValiConfig.RPC_PLAGIARISM_PORT}")

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work. Called by RPCServerBase daemon loop.
        Currently not used - plagiarism refresh happens on-demand.
        """
        pass

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        return {
            "num_plagiarism_miners": len(self.plagiarism_miners),
            "refreshed_plagiarism_time_ms": self.refreshed_plagiarism_time_ms
        }

    def get_plagiarism_miners_rpc(self) -> Dict[str, dict]:
        """Get current plagiarism miners dict."""
        return dict(self.plagiarism_miners)

    def _check_plagiarism_refresh_rpc(self, current_time: int) -> bool:
        """Check if plagiarism data needs refresh."""
        return current_time - self.refreshed_plagiarism_time_ms > ValiConfig.PLAGIARISM_UPDATE_FREQUENCY_MS

    def plagiarism_miners_to_eliminate_rpc(self, current_time: int) -> Dict[str, int]:
        """
        Returns a dict of miners that should be eliminated.

        Args:
            current_time: Current timestamp in milliseconds

        Returns:
            Dict of hotkey -> elimination_time_ms for miners to eliminate
        """
        current_plagiarism_miners = self.get_plagiarism_elimination_scores_rpc(current_time)

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

    def update_plagiarism_miners_rpc(self, current_time: int, plagiarism_miners: Dict[str, MinerBucket]) -> tuple:
        """
        Update plagiarism miners based on current data.

        Args:
            current_time: Current timestamp in milliseconds
            plagiarism_miners: Current dict of plagiarism miners

        Returns:
            Tuple of (new_plagiarism_miners list, whitelisted_miners list)
        """
        # Get updated elimination miners from microservice
        current_plagiarism_miners = self.get_plagiarism_elimination_scores_rpc(current_time)

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

    def _update_plagiarism_in_memory_rpc(self, current_time: int, plagiarism_miners: dict) -> None:
        """Update plagiarism data in memory."""
        self.plagiarism_miners = plagiarism_miners
        self.refreshed_plagiarism_time_ms = current_time

    def clear_plagiarism_data_rpc(self) -> None:
        """Clear all plagiarism data (for testing)."""
        self.plagiarism_miners.clear()
        self.refreshed_plagiarism_time_ms = 0

    def set_plagiarism_miners_for_test_rpc(self, plagiarism_miners: dict, current_time: int) -> None:
        """
        Set plagiarism miners directly for testing (bypasses API).

        Args:
            plagiarism_miners: Dict of {hotkey: {"time": timestamp_ms}}
            current_time: Current timestamp to set as refresh time
        """
        self._update_plagiarism_in_memory_rpc(current_time, plagiarism_miners)

    def get_plagiarism_elimination_scores_rpc(self, current_time: int, api_base_url: str = None) -> Optional[dict]:
        """
        Get elimination scores from the plagiarism API.

        Args:
            current_time: Current timestamp in milliseconds
            api_base_url: Base URL of the API server (optional override)

        Returns:
            Dict of elimination scores, or None if API error occurred
        """

        if api_base_url is None:
            api_base_url = self.plagiarism_url

        # During unit tests, skip API calls and just return in-memory data
        # Tests use set_plagiarism_miners_for_test() to inject test data
        if self.running_unit_tests:
            return self.plagiarism_miners

        if self._check_plagiarism_refresh_rpc(current_time):
            try:
                response = requests.get(f"{api_base_url}/elimination_scores")
                response.raise_for_status()
                new_miners = response.json()

                if not isinstance(new_miners, dict):
                    raise ValueError(f"API returned invalid data type: expected dict, got: {new_miners} with type: {type(new_miners)}")

                bt.logging.info(f"Updating plagiarism api miners from {self.plagiarism_miners} to {new_miners}")
                self._update_plagiarism_in_memory_rpc(current_time, new_miners)
                return self.plagiarism_miners
            except Exception as e:
                print(f"Error fetching plagiarism elimination scores: {e}")
                return None
        else:
            return self.plagiarism_miners

    def send_plagiarism_demotion_notification_rpc(self, hotkey: str) -> None:
        """Send notification when a miner is demoted due to plagiarism."""
        if self.running_unit_tests:
            return
        if self.slack_notifier:
            self.slack_notifier.send_plagiarism_demotion_notification(hotkey)

    def send_plagiarism_promotion_notification_rpc(self, hotkey: str) -> None:
        """Send notification when a miner is promoted from plagiarism back to probation."""
        if self.running_unit_tests:
            return
        if self.slack_notifier:
            self.slack_notifier.send_plagiarism_promotion_notification(hotkey)

    def send_plagiarism_elimination_notification_rpc(self, hotkey: str) -> None:
        """Send notification when a miner is eliminated from plagiarism."""
        if self.running_unit_tests:
            return
        if self.slack_notifier:
            self.slack_notifier.send_plagiarism_elimination_notification(hotkey)

    # ==================== Forward-Compatible Aliases (without _rpc suffix) ====================
    # These allow direct use of the server in tests without RPC

    def get_plagiarism_miners(self) -> Dict[str, dict]:
        """Get current plagiarism miners dict."""
        return self.get_plagiarism_miners_rpc()

    def plagiarism_miners_to_eliminate(self, current_time: int) -> Dict[str, int]:
        """Returns a dict of miners that should be eliminated."""
        return self.plagiarism_miners_to_eliminate_rpc(current_time)

    def update_plagiarism_miners(self, current_time: int, plagiarism_miners: Dict[str, MinerBucket]) -> tuple:
        """Update plagiarism miners based on current data."""
        return self.update_plagiarism_miners_rpc(current_time, plagiarism_miners)

    def get_plagiarism_elimination_scores(self, current_time: int, api_base_url: str = None) -> Optional[dict]:
        """Get elimination scores from the plagiarism API."""
        return self.get_plagiarism_elimination_scores_rpc(current_time, api_base_url)

    def send_plagiarism_demotion_notification(self, hotkey: str) -> None:
        """Send notification when a miner is demoted due to plagiarism."""
        self.send_plagiarism_demotion_notification_rpc(hotkey)

    def send_plagiarism_promotion_notification(self, hotkey: str) -> None:
        """Send notification when a miner is promoted from plagiarism back to probation."""
        self.send_plagiarism_promotion_notification_rpc(hotkey)

    def send_plagiarism_elimination_notification(self, hotkey: str) -> None:
        """Send notification when a miner is eliminated from plagiarism."""
        self.send_plagiarism_elimination_notification_rpc(hotkey)

    def clear_plagiarism_data(self) -> None:
        """Clear all plagiarism data (for testing)."""
        self.clear_plagiarism_data_rpc()


# ==================== Server Entry Point ====================

def start_plagiarism_server(
    slack_notifier=None,
    running_unit_tests: bool = False,
    shutdown_dict=None,
    server_ready=None
):
    """
    Entry point for server process.

    Args:
        slack_notifier: SlackNotifier for alerts
        running_unit_tests: Whether running in test mode
        shutdown_dict: Shared shutdown flag
        server_ready: Event to signal when server is ready
    """
    from setproctitle import setproctitle
    import time

    setproctitle("vali_PlagiarismServerProcess")

    # Create server with auto-start of RPC server
    server_instance = PlagiarismServer(
        slack_notifier=slack_notifier,
        running_unit_tests=running_unit_tests,
        shutdown_dict=shutdown_dict,
        start_server=True,
        start_daemon=False
    )

    bt.logging.success(f"PlagiarismServer ready on port {ValiConfig.RPC_PLAGIARISM_PORT}")

    if server_ready:
        server_ready.set()

    # Block until shutdown (RPCServerBase runs server in background thread)
    while not shutdown_dict:
        time.sleep(1)

    # Graceful shutdown
    server_instance.shutdown()
    bt.logging.info("PlagiarismServer process exiting")
