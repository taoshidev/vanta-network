from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import RPCConnectionMode, ValiConfig


class MinerStatisticsClient(RPCClientBase):
    """
    Lightweight RPC client for accessing MinerStatisticsServer.

    Creates no dependencies - just connects to existing server.
    Can be created in any process that needs statistics data.

    Forward compatibility - consumers create their own client instance.

    Example:
        client = MinerStatisticsClient()
        compressed = client.get_compressed_statistics(include_checkpoints=True)
        client.generate_request_minerstatistics(time_now=...)
    """

    def __init__(
        self,
        port: int = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        connect_immediately: bool = True,
        running_unit_tests: bool = False
    ):
        """
        Initialize MinerStatisticsClient.

        Args:
            port: Port number of the MinerStatistics server (default: ValiConfig.RPC_MINERSTATS_PORT)
            connection_mode: RPCConnectionMode enum specifying connection behavior:
                - LOCAL (0): Direct mode - bypass RPC, use set_direct_server()
                - RPC (1): Normal RPC mode - connect via network
            connect_immediately: If True, connect in __init__. If False, call connect() later.
            running_unit_tests: Whether running in unit test mode (used by orchestrator)
        """
        super().__init__(
            service_name=ValiConfig.RPC_MINERSTATS_SERVICE_NAME,
            port=port or ValiConfig.RPC_MINERSTATS_PORT,
            max_retries=60,
            retry_delay_s=1.0,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode
        )

    # ==================== Client Methods ====================

    def generate_request_minerstatistics(
        self,
        time_now: int,
        checkpoints: bool = True,
        risk_report: bool = False,
        bypass_confidence: bool = False,
        custom_output_path: str = None
    ) -> None:
        """
        Generate miner statistics and update the pre-compressed cache.

        Args:
            time_now: Current timestamp in milliseconds
            checkpoints: Whether to include checkpoints in the output
            risk_report: Whether to include risk report
            bypass_confidence: Whether to bypass confidence checks
            custom_output_path: Optional custom output path for the file
        """
        return self._server.generate_request_minerstatistics_rpc(
            time_now=time_now,
            checkpoints=checkpoints,
            risk_report=risk_report,
            bypass_confidence=bypass_confidence,
            custom_output_path=custom_output_path
        )

    def get_compressed_statistics(self, include_checkpoints: bool = True) -> bytes | None:
        """
        Get pre-compressed statistics payload for immediate API response.

        Args:
            include_checkpoints: If True, returns stats with checkpoints; otherwise without

        Returns:
            Cached compressed gzip bytes of statistics JSON (None if cache not built yet)
        """
        return self._server.get_compressed_statistics_rpc(include_checkpoints)

    def generate_miner_statistics_data(
        self,
        time_now: int = None,
        checkpoints: bool = True,
        risk_report: bool = False,
        selected_miner_hotkeys: list = None,
        final_results_weighting: bool = True,
        bypass_confidence: bool = False
    ) -> dict:
        """
        Generate miner statistics data structure (used for testing and advanced access).

        Args:
            time_now: Current timestamp in milliseconds (optional, defaults to current time)
            checkpoints: Whether to include checkpoints in the output
            risk_report: Whether to include risk report
            selected_miner_hotkeys: Optional list of specific hotkeys to process
            final_results_weighting: Whether to apply final results weighting
            bypass_confidence: Whether to bypass confidence checks

        Returns:
            dict: Miner statistics data structure
        """
        return self._server.generate_miner_statistics_data_rpc(
            time_now=time_now,
            checkpoints=checkpoints,
            risk_report=risk_report,
            selected_miner_hotkeys=selected_miner_hotkeys,
            final_results_weighting=final_results_weighting,
            bypass_confidence=bypass_confidence
        )

    def get_miner_statistics_for_hotkeys(self, hotkeys: list) -> dict:
        """
        Get statistics for a batch of hotkeys from in-memory cache (fast lookup).

        This is much faster than get_compressed_statistics() + decompression + filtering
        for querying a small number of miners. Statistics are refreshed every 5 minutes
        by the daemon.

        Args:
            hotkeys: List of miner hotkeys to fetch statistics for

        Returns:
            Dict mapping hotkey -> miner statistics dict
        """
        return self._server.get_miner_statistics_for_hotkeys_rpc(hotkeys)

    def get_miner_statistics_for_hotkey(self, hotkey: str) -> dict | None:
        """
        Get statistics for a single hotkey from in-memory cache (fast O(1) lookup).

        Statistics are refreshed every 5 minutes by the daemon.

        Args:
            hotkey: Miner hotkey to fetch statistics for

        Returns:
            Miner statistics dict or None if not found/cache not built yet
        """
        return self._server.get_miner_statistics_for_hotkey_rpc(hotkey)

    def health_check(self) -> dict:
        """Check server health."""
        return self._server.health_check_rpc()
