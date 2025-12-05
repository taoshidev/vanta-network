from shared_objects.rpc.rpc_client_base import RPCClientBase
from vali_objects.vali_config import RPCConnectionMode, ValiConfig


class CoreOutputsClient(RPCClientBase):
    """
    Lightweight RPC client for accessing CoreOutputsServer.

    Creates no dependencies - just connects to existing server.
    Can be created in any process that needs checkpoint data.

    Forward compatibility - consumers create their own client instance.

    Example:
        client = CoreOutputsClient()
        checkpoint = client.generate_request_core()
        compressed = client.get_compressed_checkpoint_from_memory()
    """

    def __init__(
        self,
        port: int = None,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        connect_immediately: bool = True,
        running_unit_tests: bool = False
    ):
        """
        Initialize CoreOutputsClient.

        Args:
            port: Port number of the CoreOutputs server (default: ValiConfig.RPC_COREOUTPUTS_PORT)
            connection_mode: RPCConnectionMode enum specifying connection behavior:
                - LOCAL (0): Direct mode - bypass RPC, use set_direct_server()
                - RPC (1): Normal RPC mode - connect via network
            connect_immediately: Whether to connect immediately (default: True)
        """
        self.running_unit_tests = running_unit_tests
        super().__init__(
            service_name=ValiConfig.RPC_COREOUTPUTS_SERVICE_NAME,
            port=port or ValiConfig.RPC_COREOUTPUTS_PORT,
            max_retries=60,
            retry_delay_s=1.0,
            connect_immediately=connect_immediately,
            connection_mode=connection_mode
        )

    def generate_request_core(
        self,
        get_dash_data_hotkey: str | None = None,
        write_and_upload_production_files: bool = False,
        create_production_files: bool = True,
        save_production_files: bool = False,
        upload_production_files: bool = False
    ) -> dict:
        """
        Generate request core data and optionally create/save/upload production files.

        Args:
            get_dash_data_hotkey: Optional specific hotkey to query (for dashboard)
            write_and_upload_production_files: Legacy parameter - if True, creates/saves/uploads files
            create_production_files: If False, skips creating production file dicts
            save_production_files: If False, skips writing files to disk
            upload_production_files: If False, skips uploading to gcloud

        Returns:
            dict: Checkpoint data containing positions, challengeperiod, etc.
        """
        return self._server.generate_request_core_rpc(
            get_dash_data_hotkey=get_dash_data_hotkey,
            write_and_upload_production_files=write_and_upload_production_files,
            create_production_files=create_production_files,
            save_production_files=save_production_files,
            upload_production_files=upload_production_files
        )

    def get_compressed_checkpoint_from_memory(self) -> bytes | None:
        """
        Get pre-compressed checkpoint data from memory cache.

        Returns:
            Cached compressed gzip bytes of checkpoint JSON (None if cache not built yet)
        """
        return self._server.get_compressed_checkpoint_from_memory_rpc()

    def health_check(self) -> bool:
        """Check server health."""
        return self._server.health_check_rpc()
