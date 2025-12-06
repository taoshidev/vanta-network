# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
MinerStatisticsServer and MinerStatisticsClient - RPC-based miner statistics service.

This module provides:
- MinerStatisticsServer: Wraps MinerStatisticsManager and exposes statistics generation via RPC
- MinerStatisticsClient: Lightweight RPC client for accessing statistics data

Architecture:
- MinerStatisticsManager (in miner_statistics_manager.py): Contains all heavy business logic
- MinerStatisticsServer: Wraps manager and exposes methods via RPC (inherits from RPCServerBase)
- MinerStatisticsClient: Lightweight RPC client (inherits from RPCClientBase)
- Forward-compatible: Consumers create their own MinerStatisticsClient instances

This follows the same pattern as PerfLedgerServer/PerfLedgerManager and
EliminationServer/EliminationManager.

Usage:
    # In validator.py - create server with daemon for periodic cache refresh
    miner_statistics_server = MinerStatisticsServer(
        start_server=True,
        start_daemon=True  # Daemon refreshes statistics cache every 5 minutes
    )

    # In consumers - create client
    client = MinerStatisticsClient()
    compressed = client.get_compressed_statistics(include_checkpoints=True)
    client.generate_request_minerstatistics(time_now=...)
"""

import bittensor as bt

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.statistics.miner_statistics_manager import MinerStatisticsManager

from shared_objects.rpc.rpc_server_base import RPCServerBase


class MinerStatisticsServer(RPCServerBase):
    """
    RPC server for miner statistics generation and management.

    Wraps MinerStatisticsManager and exposes its methods via RPC.
    All public methods ending in _rpc are exposed via RPC to clients.

    This follows the same pattern as PerfLedgerServer and EliminationServer.
    """
    service_name = ValiConfig.RPC_MINERSTATISTICS_PORT
    service_port = ValiConfig.RPC_MINERSTATS_PORT

    def __init__(
        self,
        metrics: dict = None,
        running_unit_tests: bool = False,
        slack_notifier=None,
        start_server: bool = True,
        start_daemon: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize MinerStatisticsServer.

        The server creates its own MinerStatisticsManager internally (forward compatibility pattern).

        Args:
            metrics: Metrics configuration dict (optional, uses defaults if None)
            running_unit_tests: Whether running in unit test mode
            slack_notifier: Optional SlackNotifier for alerts
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon (refreshes statistics cache every 5 minutes)
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        self.running_unit_tests = running_unit_tests

        # Initialize RPCServerBase (handles RPC server lifecycle, daemon, watchdog)
        super().__init__(
            service_name=ValiConfig.RPC_MINERSTATS_SERVICE_NAME,
            port=ValiConfig.RPC_MINERSTATS_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # We'll start daemon after manager is initialized
            daemon_interval_s=300.0,  # Refresh statistics cache every 5 minutes (expensive operation)
            hang_timeout_s=600.0,  # 10 minute timeout for statistics generation
            connection_mode=connection_mode,
            daemon_stagger_s=60
        )

        # Create the actual MinerStatisticsManager (contains all business logic)
        self._manager = MinerStatisticsManager(
            metrics=metrics,
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode
        )

        bt.logging.info(f"[MINERSTATS_SERVER] MinerStatisticsManager initialized")

        # Start daemon if requested (deferred until all initialization complete)
        if start_daemon:
            self.start_daemon()

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work - delegates to manager's statistics generation.

        MinerStatisticsServer daemon periodically generates miner statistics to keep
        the in-memory cache fresh for API requests. This pre-warms the cache so
        API responses are instant rather than requiring on-demand generation.

        Runs every ~5 minutes (controlled by daemon_interval_s in __init__).
        Statistics generation is expensive, so we use a longer interval.
        """
        try:
            time_now = TimeUtil.now_in_millis()
            bt.logging.debug(f"MinerStatisticsServer daemon: generating statistics cache...")

            # Delegate to manager for statistics generation
            self._manager.generate_request_minerstatistics(
                time_now=time_now,
                checkpoints=True,
                risk_report=False,
                bypass_confidence=False
            )

            elapsed_ms = TimeUtil.now_in_millis() - time_now
            bt.logging.info(f"MinerStatisticsServer daemon: statistics cache refreshed in {elapsed_ms}ms")

        except Exception as e:
            bt.logging.error(f"MinerStatisticsServer daemon error: {e}")
            # Don't re-raise - let daemon continue on next iteration

    # ==================== Properties (Forward Compatibility) ====================

    @property
    def position_manager(self):
        """Get position manager client (via manager)."""
        return self._manager.position_manager

    @property
    def elimination_manager(self):
        """Get elimination manager client (via manager)."""
        return self._manager.elimination_manager

    @property
    def challengeperiod_manager(self):
        """Get challenge period client (via manager)."""
        return self._manager.challengeperiod_manager

    @property
    def contract_manager(self):
        """Get contract client (via manager - forward compatibility)."""
        return self._manager.contract_manager

    @property
    def perf_ledger_manager(self):
        """Get perf ledger client (via manager)."""
        return self._manager.perf_ledger_manager

    @property
    def metrics_calculator(self):
        """Get metrics calculator (via manager)."""
        return self._manager.metrics_calculator

    # ==================== RPC Methods (exposed to clients) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        cache_with_checkpoints = self._manager.get_compressed_statistics(include_checkpoints=True)
        cache_without_checkpoints = self._manager.get_compressed_statistics(include_checkpoints=False)

        cache_status = 'both_cached' if (cache_with_checkpoints and cache_without_checkpoints) else \
                      'partial' if (cache_with_checkpoints or cache_without_checkpoints) else 'empty'

        return {
            "cache_status": cache_status
        }

    def generate_request_minerstatistics_rpc(
        self,
        time_now: int,
        checkpoints: bool = True,
        risk_report: bool = False,
        bypass_confidence: bool = False,
        custom_output_path: str = None
    ) -> None:
        """
        Generate miner statistics and update the pre-compressed cache via RPC.

        Delegates to manager for actual statistics generation.
        """
        return self._manager.generate_request_minerstatistics(
            time_now=time_now,
            checkpoints=checkpoints,
            risk_report=risk_report,
            bypass_confidence=bypass_confidence,
            custom_output_path=custom_output_path
        )

    def get_compressed_statistics_rpc(self, include_checkpoints: bool = True) -> bytes | None:
        """
        Retrieve compressed miner statistics data directly from memory cache via RPC.

        Delegates to manager for cache retrieval.
        """
        return self._manager.get_compressed_statistics(include_checkpoints)

    def generate_miner_statistics_data_rpc(
        self,
        time_now: int = None,
        checkpoints: bool = True,
        risk_report: bool = False,
        selected_miner_hotkeys: list = None,
        final_results_weighting: bool = True,
        bypass_confidence: bool = False
    ) -> dict:
        """
        Generate miner statistics data structure via RPC.

        Delegates to manager for statistics generation.
        """
        return self._manager.generate_miner_statistics_data(
            time_now=time_now,
            checkpoints=checkpoints,
            risk_report=risk_report,
            selected_miner_hotkeys=selected_miner_hotkeys,
            final_results_weighting=final_results_weighting,
            bypass_confidence=bypass_confidence
        )

    def get_miner_statistics_for_hotkeys_rpc(self, hotkeys: list) -> dict:
        """
        Get statistics for a batch of hotkeys from in-memory cache via RPC.

        Delegates to manager for fast O(1) lookup per hotkey.

        Args:
            hotkeys: List of miner hotkeys to fetch statistics for

        Returns:
            Dict mapping hotkey -> miner statistics dict
        """
        return self._manager.get_miner_statistics_for_hotkeys(hotkeys)

    def get_miner_statistics_for_hotkey_rpc(self, hotkey: str) -> dict | None:
        """
        Get statistics for a single hotkey from in-memory cache via RPC.

        Delegates to manager for fast O(1) lookup.

        Args:
            hotkey: Miner hotkey to fetch statistics for

        Returns:
            Miner statistics dict or None if not found
        """
        return self._manager.get_miner_statistics_for_hotkey(hotkey)

    # ==================== Forward-Compatible Aliases (without _rpc suffix) ====================
    # These allow direct use of the server in tests without RPC

    def generate_request_minerstatistics(
        self,
        time_now: int,
        checkpoints: bool = True,
        risk_report: bool = False,
        bypass_confidence: bool = False,
        custom_output_path: str = None
    ) -> None:
        """
        Generate miner statistics - delegates to manager.

        This is a forward-compatible alias for direct server access (tests).
        """
        return self._manager.generate_request_minerstatistics(
            time_now=time_now,
            checkpoints=checkpoints,
            risk_report=risk_report,
            bypass_confidence=bypass_confidence,
            custom_output_path=custom_output_path
        )

    def get_compressed_statistics(self, include_checkpoints: bool = True) -> bytes | None:
        """Get compressed statistics from memory - delegates to manager."""
        return self._manager.get_compressed_statistics(include_checkpoints)

    def generate_miner_statistics_data(
        self,
        time_now: int = None,
        checkpoints: bool = True,
        risk_report: bool = False,
        selected_miner_hotkeys: list = None,
        final_results_weighting: bool = True,
        bypass_confidence: bool = False
    ) -> dict:
        """Generate miner statistics data - delegates to manager."""
        return self._manager.generate_miner_statistics_data(
            time_now=time_now,
            checkpoints=checkpoints,
            risk_report=risk_report,
            selected_miner_hotkeys=selected_miner_hotkeys,
            final_results_weighting=final_results_weighting,
            bypass_confidence=bypass_confidence
        )


if __name__ == "__main__":
    # NOTE: This standalone test script needs the RPC servers running
    # In production, MinerStatisticsServer creates its own clients

    import os
    from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

    bt.logging.enable_info()
    all_hotkeys = ValiBkpUtils.get_directories_in_dir(ValiBkpUtils.get_miner_dir())
    print('N hotkeys:', len(all_hotkeys))

    # MinerStatisticsServer creates its own RPC clients
    server = MinerStatisticsServer(
        running_unit_tests=False,
        start_server=True,
        start_daemon=False
    )

    pwd = os.getcwd()
    custom_output_path = os.path.join(pwd, 'debug_miner_statistics.json')
    server.generate_request_minerstatistics(TimeUtil.now_in_millis(), True, custom_output_path=custom_output_path)

    # Confirm output path and ability to read file
    if os.path.exists(custom_output_path):
        import json
        with open(custom_output_path, 'r') as f:
            data = json.load(f)
            print('Generated miner statistics:', custom_output_path)
    else:
        print(f"Output file not found at {custom_output_path}")
