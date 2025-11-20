"""
RequestCoreManager RPC Client

This module provides a lightweight RPC client for accessing validator checkpoint generation.

The actual checkpoint generation happens in RequestCoreManagerServer (see generate_request_core_server.py).
This client provides access to checkpoint generation and retrieval via RPC.

Architecture:
- RequestCoreManagerServer: Heavy server class that generates and caches checkpoints
- RequestCoreManager (this file): Lightweight RPC client for validator access

Usage:
    # Create client (automatically starts server process)
    manager = RequestCoreManager(
        position_manager=position_manager,
        subtensor_weight_setter=subtensor_weight_setter,
        plagiarism_detector=plagiarism_detector,
        contract_manager=contract_manager
    )

    # Generate checkpoint (triggers server-side generation)
    checkpoint = manager.generate_request_core()

    # Get compressed checkpoint from memory cache
    compressed_data = manager.get_compressed_checkpoint_from_memory()
"""

from vali_objects.utils.position_manager import PositionManager
from vali_objects.utils.subtensor_weight_setter import SubtensorWeightSetter
from vali_objects.utils.plagiarism_detector import PlagiarismDetector
from vali_objects.utils.validator_contract_manager import ValidatorContractManager
from vali_objects.utils.asset_selection_manager import AssetSelectionManager
from vali_objects.utils.limit_order_manager import LimitOrderManager
from vali_objects.vali_config import ValiConfig
from shared_objects.rpc_service_base import RPCServiceBase


class RequestCoreManager(RPCServiceBase):
    """
    Lightweight RPC client for validator checkpoint generation and retrieval.

    The actual checkpoint generation happens in RequestCoreManagerServer.
    This client provides access to checkpoint generation and cached data via RPC.

    Inherits from RPCServiceBase for common RPC infrastructure (connection management,
    process lifecycle, stale server cleanup, health checks).

    Public API remains the same for backward compatibility.
    """

    def __init__(
        self,
        position_manager: PositionManager,
        subtensor_weight_setter: SubtensorWeightSetter,
        plagiarism_detector: PlagiarismDetector,
        contract_manager: ValidatorContractManager = None,
        asset_selection_manager: AssetSelectionManager = None,
        limit_order_manager: LimitOrderManager = None,
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
            asset_selection_manager: Asset selection manager instance (passed to server)
            limit_order_manager: Limit order manager instance (passed to server)
            start_server: Whether to start the server process
            running_unit_tests: Whether running in unit test mode
        """
        # Store dependencies for server creation
        self._position_manager = position_manager
        self._subtensor_weight_setter = subtensor_weight_setter
        self._plagiarism_detector = plagiarism_detector
        self._contract_manager = contract_manager
        self._asset_selection_manager = asset_selection_manager
        self._limit_order_manager = limit_order_manager

        # Initialize RPCServiceBase (handles connection, process lifecycle, etc.)
        super().__init__(
            service_name=ValiConfig.RPC_REQUESTCORE_SERVICE_NAME,
            port=ValiConfig.RPC_REQUESTCORE_PORT,
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
            RequestCoreManagerServer instance (not proxied, direct Python object)
        """
        from runnable.generate_request_core_server import RequestCoreManagerServer

        return RequestCoreManagerServer(
            position_manager=self._position_manager,
            subtensor_weight_setter=self._subtensor_weight_setter,
            plagiarism_detector=self._plagiarism_detector,
            contract_manager=self._contract_manager,
            asset_selection_manager=self._asset_selection_manager,
            limit_order_manager=self._limit_order_manager
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
        from runnable.generate_request_core_server import RequestCoreManagerServer

        def server_main():
            from setproctitle import setproctitle
            setproctitle(f"vali_{self.service_name}")

            # Create server instance
            server = RequestCoreManagerServer(
                position_manager=self._position_manager,
                subtensor_weight_setter=self._subtensor_weight_setter,
                plagiarism_detector=self._plagiarism_detector,
                contract_manager=self._contract_manager,
                asset_selection_manager=self._asset_selection_manager,
                limit_order_manager=self._limit_order_manager
            )

            # Serve via RPC (uses RPCServiceBase helper)
            self._serve_rpc(server, address, authkey, server_ready)

        process = Process(target=server_main, daemon=True)
        process.start()
        return process

    # ============================================================================
    # PUBLIC CLIENT METHODS (call server methods - RPC or direct depending on mode)
    # ============================================================================

    def generate_request_core(
        self,
        get_dash_data_hotkey: str | None = None,
        write_and_upload_production_files=False,
        create_production_files=True,
        save_production_files=False,
        upload_production_files=False
    ) -> dict:
        """
        Generate request core data and optionally create/save/upload production files.

        This method generates checkpoint data containing positions, challengeperiod info, etc.

        Note: In direct mode (unit tests), this calls the server directly.
              In RPC mode, this calls the server via RPC.

        Args:
            get_dash_data_hotkey: Optional specific hotkey to query (for dashboard)
            write_and_upload_production_files: Legacy parameter - if True, creates/saves/uploads files
            create_production_files: If False, skips creating production file dicts
            save_production_files: If False, skips writing files to disk
            upload_production_files: If False, skips uploading to gcloud

        Returns:
            dict: Checkpoint data containing positions, challengeperiod, etc.
        """
        return self._server_proxy.generate_request_core(
            get_dash_data_hotkey=get_dash_data_hotkey,
            write_and_upload_production_files=write_and_upload_production_files,
            create_production_files=create_production_files,
            save_production_files=save_production_files,
            upload_production_files=upload_production_files
        )

    def get_compressed_checkpoint_from_memory(self) -> bytes | None:
        """
        Get pre-compressed checkpoint data from memory cache for immediate API response.

        This method returns pre-compressed data that was cached during the last
        checkpoint generation, providing instant RPC access without compression overhead.

        Returns:
            Cached compressed gzip bytes of checkpoint JSON (None if cache not built yet)
        """
        return self._server_proxy.get_compressed_checkpoint_from_memory_rpc()

    def health_check(self) -> bool:
        """Check server health."""
        return self._server_proxy.health_check_rpc()
