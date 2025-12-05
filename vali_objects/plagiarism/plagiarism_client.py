from typing import Dict, Optional

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import RPCConnectionMode, ValiConfig


class PlagiarismClient(RPCClientBase):
    """
    Lightweight RPC client for PlagiarismServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_PLAGIARISM_PORT.

    In test mode (running_unit_tests=True), the client won't connect via RPC.
    Instead, use set_direct_server() to provide a direct PlagiarismServer instance.
    """

    def __init__(self, port: int = None, running_unit_tests: bool = False,
                 connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
                 connect_immediately: bool = False):
        """
        Initialize plagiarism client.

        Args:
            port: Port number of the plagiarism server (default: ValiConfig.RPC_PLAGIARISM_PORT)
            running_unit_tests: If True, don't connect via RPC (use set_direct_server() instead)
            connection_mode: RPC connection mode (LOCAL or RPC)
            connect_immediately: Whether to connect to server immediately
        """
        self.running_unit_tests = running_unit_tests

        # In test mode, don't connect via RPC - tests will set direct server
        super().__init__(
            service_name=ValiConfig.RPC_PLAGIARISM_SERVICE_NAME,
            port=port or ValiConfig.RPC_PLAGIARISM_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode
        )

    # ==================== Query Methods ====================

    def get_plagiarism_miners(self) -> Dict[str, dict]:
        """Get current plagiarism miners dict."""
        return self._server.get_plagiarism_miners_rpc()

    def plagiarism_miners_to_eliminate(self, current_time: int) -> Dict[str, int]:
        """
        Returns a dict of miners that should be eliminated.

        Args:
            current_time: Current timestamp in milliseconds

        Returns:
            Dict of hotkey -> elimination_time_ms for miners to eliminate
        """
        return self._server.plagiarism_miners_to_eliminate_rpc(current_time)

    def update_plagiarism_miners(self, current_time: int, plagiarism_miners: Dict[str, MinerBucket]) -> tuple:
        """
        Update plagiarism miners based on current data.

        Args:
            current_time: Current timestamp in milliseconds
            plagiarism_miners: Current dict of plagiarism miners

        Returns:
            Tuple of (new_plagiarism_miners list, whitelisted_miners list)
        """
        return self._server.update_plagiarism_miners_rpc(current_time, plagiarism_miners)

    def get_plagiarism_elimination_scores(self, current_time: int, api_base_url: str = None) -> Optional[dict]:
        """
        Get elimination scores from the plagiarism API.

        Args:
            current_time: Current timestamp in milliseconds
            api_base_url: Base URL of the API server (optional override)

        Returns:
            Dict of elimination scores, or None if API error occurred
        """
        return self._server.get_plagiarism_elimination_scores_rpc(current_time, api_base_url)

    # ==================== Notification Methods ====================

    def send_plagiarism_demotion_notification(self, hotkey: str) -> None:
        """Send notification when a miner is demoted due to plagiarism."""
        self._server.send_plagiarism_demotion_notification_rpc(hotkey)

    def send_plagiarism_promotion_notification(self, hotkey: str) -> None:
        """Send notification when a miner is promoted from plagiarism back to probation."""
        self._server.send_plagiarism_promotion_notification_rpc(hotkey)

    def send_plagiarism_elimination_notification(self, hotkey: str) -> None:
        """Send notification when a miner is eliminated from plagiarism."""
        self._server.send_plagiarism_elimination_notification_rpc(hotkey)

    # ==================== Data Management ====================

    def clear_plagiarism_data(self) -> None:
        """Clear all plagiarism data (for testing)."""
        self._server.clear_plagiarism_data_rpc()

    def set_plagiarism_miners_for_test(self, plagiarism_miners: dict, current_time: int) -> None:
        """
        Set plagiarism miners directly for testing (bypasses API).

        Args:
            plagiarism_miners: Dict of {hotkey: {"time": timestamp_ms}}
            current_time: Current timestamp to set as refresh time
        """
        self._server.set_plagiarism_miners_for_test_rpc(plagiarism_miners, current_time)

    # ==================== Health Check ====================

    def health_check(self) -> dict:
        """Health check endpoint."""
        return self._server.health_check_rpc()
