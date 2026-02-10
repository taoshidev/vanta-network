# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
WeightCalculatorServer - RPC server for weight calculation and setting.

This server runs in its own process and handles:
- Computing miner weights using debt-based scoring
- Sending weight setting requests to SubtensorOpsManager via RPC

All business logic is delegated to WeightCalculatorManager.

Usage:
    # Validator spawns the server at startup
    from vali_objects.scoring.weight_calculator_server import start_weight_calculator_server

    process = Process(target=start_weight_calculator_server, args=(...))
    process.start()

    # Other processes connect via WeightCalculatorClient
    from vali_objects.scoring.weight_calculator_client import WeightCalculatorClient
    client = WeightCalculatorClient()
"""
import time
import traceback
from typing import List, Tuple

from setproctitle import setproctitle

import bittensor as bt

from shared_objects.cache_controller import CacheController
from shared_objects.error_utils import ErrorUtils
from shared_objects.rpc.rpc_server_base import RPCServerBase
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.scoring.weight_calculator_manager import WeightCalculatorManager
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator


class WeightCalculatorServer(RPCServerBase, CacheController):
    """
    RPC server for weight calculation and setting.

    Wraps WeightCalculatorManager and exposes its methods via RPC.
    All public methods ending in _rpc are exposed via RPC to WeightCalculatorClient.

    This follows the same pattern as ChallengePeriodServer.

    Inherits from:
    - RPCServerBase: Provides RPC server lifecycle, daemon management, watchdog
    - CacheController: Provides cache file management utilities
    """
    service_name = ValiConfig.RPC_WEIGHT_CALCULATOR_SERVICE_NAME
    service_port = ValiConfig.RPC_WEIGHT_CALCULATOR_PORT

    def __init__(
        self,
        running_unit_tests=False,
        is_backtesting=False,
        slack_notifier=None,
        config=None,
        hotkey=None,
        is_mainnet=True,
        start_server=True,
        start_daemon=True,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize WeightCalculatorServer.

        Args:
            running_unit_tests: Whether running in test mode
            is_backtesting: Whether running in backtesting mode
            slack_notifier: Slack notifier for error reporting
            config: Validator config (for slack webhook URLs)
            hotkey: Validator hotkey
            is_mainnet: Whether running on mainnet
            start_server: Whether to start RPC server immediately
            start_daemon: Whether to start daemon immediately
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        self.running_unit_tests = running_unit_tests
        self.connection_mode = connection_mode

        # Create mock config/hotkey if running tests and not provided
        if running_unit_tests:
            from shared_objects.rpc.test_mock_factory import TestMockFactory
            config = TestMockFactory.create_mock_config_if_needed(config, netuid=116, network="test")
            hotkey = TestMockFactory.create_mock_hotkey_if_needed(hotkey, default_hotkey="test_validator_hotkey")
            if is_mainnet is None:
                is_mainnet = False

        # Always create in-process - constructor NEVER spawns
        bt.logging.info("[WC_SERVER] Creating WeightCalculatorServer in-process")

        # Initialize CacheController first (for cache file setup)
        CacheController.__init__(
            self,
            running_unit_tests=running_unit_tests,
            is_backtesting=is_backtesting,
            connection_mode=connection_mode
        )

        # Create the actual WeightCalculatorManager FIRST, before RPCServerBase.__init__
        # This ensures _manager exists before RPC server starts accepting calls (if start_server=True)
        # CRITICAL: Prevents race condition where RPC calls fail with AttributeError during initialization
        self._manager = WeightCalculatorManager(
            is_backtesting=is_backtesting,
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode,
            config=config,
            hotkey=hotkey,
            is_mainnet=is_mainnet,
            slack_notifier=slack_notifier
        )

        bt.logging.info("[WC_SERVER] WeightCalculatorManager initialized")

        # Initialize RPCServerBase (may start RPC server immediately if start_server=True)
        # At this point, self._manager exists, so RPC calls won't fail
        # daemon_interval_s: 5 minutes (weight calculation frequency)
        # hang_timeout_s: 10 minutes (accounts for 5min sleep in retry logic + processing time)
        daemon_interval_s = ValiConfig.SET_WEIGHT_REFRESH_TIME_MS / 1000.0  # 5 minutes (300s)
        hang_timeout_s = 600.0  # 10 minutes (accounts for time.sleep(300) in retry logic + processing)

        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_WEIGHT_CALCULATOR_SERVICE_NAME,
            port=ValiConfig.RPC_WEIGHT_CALCULATOR_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # We'll start daemon after full initialization
            daemon_interval_s=daemon_interval_s,
            hang_timeout_s=hang_timeout_s,
            connection_mode=connection_mode
        )

        # Start daemon if requested (deferred until all initialization complete)
        if start_daemon:
            self.start_daemon()

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work. Called by RPCServerBase daemon loop.

        Computes weights and sends to SubtensorOpsManager.
        """
        if not self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
            return

        bt.logging.info("Running weight calculator daemon iteration")
        current_time = TimeUtil.now_in_millis()

        try:
            # Delegate to manager - it handles everything
            checkpoint_results, transformed_list = self._manager.compute_weights(current_time)

            if transformed_list:
                self.set_last_update_time()
            else:
                # No weights computed - likely debt ledgers not ready yet
                bt.logging.warning(
                    "No weights computed (debt ledgers may still be initializing). "
                    "Waiting 5 minutes before retry..."
                )
                time.sleep(300)

        except Exception as e:
            bt.logging.error(f"Error in weight calculator daemon: {e}")
            bt.logging.error(traceback.format_exc())

            # Send error notification (manager also sends errors, but daemon errors are critical)
            if self._manager.slack_notifier:
                compact_trace = ErrorUtils.get_compact_stacktrace(e)
                self._manager.slack_notifier.send_message(
                    f"Weight calculator daemon error!\n"
                    f"Error: {str(e)}\n"
                    f"Trace: {compact_trace}",
                    level="error"
                )
            time.sleep(30)

    # ==================== Properties ====================

    @property
    def slack_notifier(self):
        """Get slack notifier from manager."""
        return self._manager.slack_notifier

    @slack_notifier.setter
    def slack_notifier(self, value):
        """Set slack notifier (used by RPCServerBase during initialization)."""
        self._manager._external_slack_notifier = value

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        n_results = len(self._manager.checkpoint_results)
        n_weights = len(self._manager.transformed_list)
        return {
            "num_checkpoint_results": n_results,
            "num_weights": n_weights
        }

    def get_checkpoint_results_rpc(self) -> List[Tuple[str, float]]:
        """Get latest checkpoint results."""
        return self._manager.get_checkpoint_results()

    def get_transformed_list_rpc(self) -> List[Tuple[int, float]]:
        """Get latest transformed weight list."""
        return self._manager.get_transformed_list()

    def compute_weights_rpc(self, current_time: int = None) -> Tuple[List[Tuple[str, float]], List[Tuple[int, float]]]:
        """
        Compute weights for all miners (exposed for testing/manual triggering).

        Args:
            current_time: Current time in milliseconds. If None, uses TimeUtil.now_in_millis().

        Returns:
            Tuple of (checkpoint_results, transformed_list)
        """
        return self._manager.compute_weights(current_time)


# ==================== Server Entry Point ====================

def start_weight_calculator_server(
    slack_notifier=None,
    config=None,
    hotkey=None,
    is_mainnet=True,
    server_ready=None
):
    """
    Entry point for server process.

    The server creates its own manager which creates its own clients internally:
    - CommonDataClient (for shutdown coordination)
    - MetagraphClient (for hotkey/UID mapping)
    - PositionManagerClient (for position data)
    - ChallengePeriodClient (for miner buckets)
    - ContractClient (for contract state)
    - DebtLedgerClient (for debt-based scoring)
    - SubtensorOpsClient (for weight setting RPC to SubtensorOpsManager)

    Args:
        slack_notifier: Slack notifier for error reporting
        config: Validator config (for slack webhook URLs)
        hotkey: Validator hotkey
        is_mainnet: Whether running on mainnet
        server_ready: Event to signal when server is ready
    """
    setproctitle("vali_WeightCalculatorServerProcess")

    # Create server with auto-start of RPC server and daemon
    server_instance = WeightCalculatorServer(
        running_unit_tests=False,
        is_backtesting=False,
        slack_notifier=slack_notifier,
        config=config,
        hotkey=hotkey,
        is_mainnet=is_mainnet,
        start_server=True,
        start_daemon=True,
        connection_mode=RPCConnectionMode.RPC
    )

    bt.logging.success(f"WeightCalculatorServer ready on port {ValiConfig.RPC_WEIGHT_CALCULATOR_PORT}")

    if server_ready:
        server_ready.set()

    # Block until shutdown (uses ShutdownCoordinator)
    while not ShutdownCoordinator.is_shutdown():
        time.sleep(1)

    # Graceful shutdown
    server_instance.shutdown()
    bt.logging.info("WeightCalculatorServer process exiting")
