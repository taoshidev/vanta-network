"""
MinerStatisticsManager RPC Client

This module provides a lightweight RPC client for accessing pre-computed miner statistics.

The actual statistics generation happens in MinerStatisticsManagerServer (see generate_request_minerstatistics_server.py).
This client only provides read-only access to the pre-compressed statistics cache via RPC.

Architecture:
- MinerStatisticsManagerServer: Heavy server class that generates and caches statistics
- MinerStatisticsManager (this file): Lightweight RPC client for API/UI access

Usage:
    # Create client (automatically starts server process)
    manager = MinerStatisticsManager(
        position_manager=position_manager,
        subtensor_weight_setter=subtensor_weight_setter,
        plagiarism_detector=plagiarism_detector,
        contract_manager=contract_manager
    )

    # Get compressed statistics (instant RPC call, no computation)
    compressed_data = manager.get_compressed_statistics(include_checkpoints=True)
"""

from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.vali_config import ValiConfig
from shared_objects.rpc_service_base import RPCServiceBase
from typing import Dict


class MinerStatisticsManager(RPCServiceBase):
    """
    Lightweight RPC client for accessing miner statistics data.

    The actual statistics generation happens in MinerStatisticsManagerServer.
    This client only provides read-only access to pre-compressed statistics via RPC.

    Inherits from RPCServiceBase for common RPC infrastructure (connection management,
    process lifecycle, stale server cleanup, health checks).

    Public API remains the same for backward compatibility.
    """

    def __init__(
        self,
        position_manager: PositionManager,
        subtensor_weight_setter: SubtensorWeightSetter,
        plagiarism_detector: PlagiarismDetector,
        contract_manager: ValidatorContractManager,
        metrics: Dict = None,
        start_server: bool = True,
        running_unit_tests: bool = False
    ):
        """
        Initialize client and optionally start server process.

        Args:
            position_manager: Position manager instance (passed to server)
            subtensor_weight_setter: Subtensor weight setter instance (passed to server)
            plagiarism_detector: Plagiarism detector instance (passed to server)
            contract_manager: Contract manager instance (passed to server)
            metrics: Metrics configuration dict (passed to server)
            start_server: Whether to start the server process
            running_unit_tests: Whether running in unit test mode
        """
        # Store dependencies for server creation
        self._position_manager = position_manager
        self._subtensor_weight_setter = subtensor_weight_setter
        self._plagiarism_detector = plagiarism_detector
        self._contract_manager = contract_manager
        self._metrics = metrics

        # Initialize RPCServiceBase (handles connection, process lifecycle, etc.)
        super().__init__(
            service_name=ValiConfig.RPC_MINERSTATISTICS_SERVICE_NAME,
            port=ValiConfig.RPC_MINERSTATISTICS_PORT,
            running_unit_tests=running_unit_tests,
            enable_health_check=False,  # Can be enabled later if needed
            slack_notifier=None
        )

        # Initialize the service (RPC mode or direct mode for tests)
        if start_server:
            self._initialize_service()

    # ============================================================================
    # RPCServiceBase IMPLEMENTATION (required abstract methods)
    # ============================================================================

    def _create_direct_server(self):
        """
        Create a direct in-memory server instance for unit tests.

        Returns:
            MinerStatisticsManagerServer instance (not proxied, direct Python object)
        """
        from runnable.generate_request_minerstatistics_server import MinerStatisticsManagerServer

        return MinerStatisticsManagerServer(
            position_manager=self._position_manager,
            subtensor_weight_setter=self._subtensor_weight_setter,
            plagiarism_detector=self._plagiarism_detector,
            contract_manager=self._contract_manager,
            metrics=self._metrics
        )

    def _start_server_process(self, address, authkey, server_ready):
        """
        Start the RPC server in a separate process.

        Args:
            address: (host, port) tuple for RPC server
            authkey: Authentication key for RPC connection
            server_ready: Event to signal when server is ready

        Returns:
            Process object for the server process
        """
        from multiprocessing import Process
        from runnable.generate_request_minerstatistics_server import MinerStatisticsManagerServer

        def server_main():
            from setproctitle import setproctitle
            setproctitle(f"vali_{self.service_name}")

            # Create server instance
            server = MinerStatisticsManagerServer(
                position_manager=self._position_manager,
                subtensor_weight_setter=self._subtensor_weight_setter,
                plagiarism_detector=self._plagiarism_detector,
                contract_manager=self._contract_manager,
                metrics=self._metrics
            )

            # Serve via RPC (uses RPCServiceBase helper)
            self._serve_rpc(server, address, authkey, server_ready)

        process = Process(target=server_main, daemon=True)
        process.start()
        return process

    # ============================================================================
    # PUBLIC CLIENT METHODS (call server methods - RPC or direct depending on mode)
    # ============================================================================

    def generate_request_minerstatistics(
        self,
        time_now: int,
        checkpoints: bool = True,
        risk_report: bool = False,
        bypass_confidence: bool = False,
        custom_output_path=None
    ):
        """
        Generate miner statistics and update the pre-compressed cache.

        This method generates the statistics data, writes it to disk for backup,
        and updates the in-memory compressed cache for instant RPC access.

        Note: In direct mode (unit tests), this calls the server directly.
              In RPC mode, this is NOT an RPC method - it's only accessible in direct mode.

        Args:
            time_now: Current timestamp in milliseconds
            checkpoints: Whether to include checkpoints in the output
            risk_report: Whether to include risk report
            bypass_confidence: Whether to bypass confidence checks
            custom_output_path: Optional custom output path for the file
        """
        self._server_proxy.generate_request_minerstatistics(
            time_now=time_now,
            checkpoints=checkpoints,
            risk_report=risk_report,
            bypass_confidence=bypass_confidence,
            custom_output_path=custom_output_path
        )

    def get_compressed_statistics(self, include_checkpoints: bool = True) -> bytes | None:
        """
        Get pre-compressed statistics payload for immediate API response.

        This method returns pre-compressed data that was cached during the last
        statistics generation, providing instant RPC access without compression overhead.

        Args:
            include_checkpoints: If True, returns stats with checkpoints; otherwise without

        Returns:
            Cached compressed gzip bytes of statistics JSON (None if cache not built yet)
        """
        return self._server_proxy.get_compressed_statistics_rpc(include_checkpoints)

    def health_check(self) -> bool:
        """Check server health."""
        return self._server_proxy.health_check_rpc()
