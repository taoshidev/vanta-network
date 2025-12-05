from typing import Dict, List

from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import RPCConnectionMode, ValiConfig


class PlagiarismDetectorClient(RPCClientBase):
    """
    Lightweight RPC client for PlagiarismDetectorServer.

    Can be created in ANY process. No server ownership.
    Port is obtained from ValiConfig.RPC_PLAGIARISM_DETECTOR_PORT.

    In test mode (running_unit_tests=True), the client won't connect via RPC.
    Instead, use set_direct_server() to provide a direct PlagiarismDetectorServer instance.
    """

    def __init__(
        self,
        port: int = None,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize plagiarism detector client.

        Args:
            port: Port number of the server (default: ValiConfig.RPC_PLAGIARISM_DETECTOR_PORT)
            running_unit_tests: If True, don't connect via RPC (use set_direct_server() instead)
            connect_immediately: Whether to connect to server immediately
        """
        self.running_unit_tests = running_unit_tests
        self._direct_server = None

        # In test mode, don't connect via RPC - tests will set direct server
        super().__init__(
            service_name=ValiConfig.RPC_PLAGIARISM_DETECTOR_SERVICE_NAME,
            port=port or ValiConfig.RPC_PLAGIARISM_DETECTOR_PORT,
            max_retries=5,
            retry_delay_s=1.0,
            connection_mode=connection_mode
        )

    # ==================== Query Methods ====================

    def get_plagiarism_scores_from_disk(self) -> Dict[str, float]:
        """
        Get plagiarism scores from disk.

        Returns:
            Dict mapping hotkeys to their plagiarism scores
        """
        return self._server.get_plagiarism_scores_from_disk_rpc()

    def get_plagiarism_data_from_disk(self) -> Dict[str, dict]:
        """
        Get detailed plagiarism data from disk.

        Returns:
            Dict mapping hotkeys to their full plagiarism data
        """
        return self._server.get_plagiarism_data_from_disk_rpc()

    def get_miner_plagiarism_data_from_disk(self, hotkey: str) -> dict:
        """
        Get plagiarism data for a specific miner from disk.

        Args:
            hotkey: Miner hotkey to look up

        Returns:
            Dict of plagiarism data for the miner, or empty dict if not found
        """
        return self._server.get_miner_plagiarism_data_from_disk_rpc(hotkey)

    def detect(self, hotkeys: List[str] = None, hotkey_positions: dict = None) -> None:
        """
        Run plagiarism detection.

        Args:
            hotkeys: List of hotkeys to analyze (optional)
            hotkey_positions: Pre-fetched positions (optional, for testing)
        """
        self._server.detect_rpc(hotkeys=hotkeys, hotkey_positions=hotkey_positions)

    def clear_plagiarism_from_disk(self, target_hotkey: str = None) -> None:
        """
        Clear plagiarism data from disk.

        Args:
            target_hotkey: Specific hotkey to clear, or None to clear all
        """
        self._server.clear_plagiarism_from_disk_rpc(target_hotkey=target_hotkey)

    # ==================== Health Check ====================

    def health_check(self) -> dict:
        """Health check endpoint."""
        return self._server.health_check_rpc()
