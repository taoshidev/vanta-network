# developer: jbonilla
# Copyright © 2024 Taoshi Inc
"""
WeightCalculatorServer - RPC server for weight calculation and setting.

This server runs in its own process and handles:
- Computing miner weights using debt-based scoring
- Sending weight setting requests to MetagraphUpdater via RPC

Usage:
    # Validator spawns the server at startup
    from vali_objects.utils.weight_calculator_server import start_weight_calculator_server

    process = Process(target=start_weight_calculator_server, args=(...))
    process.start()

    # Other processes connect via WeightCalculatorClient
    from vali_objects.utils.weight_calculator_server import WeightCalculatorClient
    client = WeightCalculatorClient()
"""
import time
import traceback
import threading
from typing import List, Tuple

from setproctitle import setproctitle

import bittensor as bt

from shared_objects.cache_controller import CacheController
from shared_objects.error_utils import ErrorUtils
from shared_objects.rpc.rpc_server_base import RPCServerBase
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.enums.miner_bucket_enum import MinerBucket
from shared_objects.slack_notifier import SlackNotifier
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator




class WeightCalculatorServer(RPCServerBase, CacheController):
    """
    RPC server for weight calculation and setting.

    Inherits from:
    - RPCServerBase: Provides RPC server lifecycle, daemon management, watchdog
    - CacheController: Provides cache file management utilities

    Architecture:
    - Runs in its own process
    - Creates RPC clients to communicate with other services
    - Computes weights using debt-based scoring
    - Sends weight setting requests to MetagraphUpdater via RPC
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
        start_daemon=True
    ):
        # Initialize CacheController first (for cache file setup)
        CacheController.__init__(self, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)

        # Store config for slack notifier creation
        self.config = config
        self.hotkey = hotkey
        self.is_mainnet = is_mainnet
        self.subnet_version = 200

        # Create own CommonDataClient (forward compatibility - no parameter passing)
        from shared_objects.rpc.common_data_server import CommonDataClient
        self._common_data_client = CommonDataClient(
            running_unit_tests=running_unit_tests
        )

        # Initialize RPCServerBase (handles RPC server and daemon lifecycle)
        # daemon_interval_s: 5 minutes (weight calculation frequency)
        # hang_timeout_s: 10 minutes (accounts for 5min sleep in retry logic + processing time)
        RPCServerBase.__init__(
            self,
            service_name=ValiConfig.RPC_WEIGHT_CALCULATOR_SERVICE_NAME,
            port=ValiConfig.RPC_WEIGHT_CALCULATOR_PORT,
            slack_notifier=slack_notifier,
            start_server=start_server,
            start_daemon=False,  # We'll start daemon after full initialization
            daemon_interval_s=ValiConfig.SET_WEIGHT_REFRESH_TIME_MS / 1000.0,  # 5 minutes (300s)
            hang_timeout_s=600.0  # 10 minutes (accounts for time.sleep(300) in retry logic + processing)
        )

        # Create own PositionManagerClient (forward compatibility - no parameter passing)
        from vali_objects.position_management.position_manager_client import PositionManagerClient
        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT, running_unit_tests=running_unit_tests
        )

        # Create own ChallengePeriodClient (forward compatibility - no parameter passing)
        from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
        self._challengeperiod_client = ChallengePeriodClient(running_unit_tests=running_unit_tests
        )

        # Create own ContractClient (forward compatibility - no parameter passing)
        from vali_objects.contract.contract_server import ContractClient
        self._contract_client = ContractClient(running_unit_tests=running_unit_tests)

        # Create own DebtLedgerClient (forward compatibility - no parameter passing)
        from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
        self._debt_ledger_client = DebtLedgerClient(running_unit_tests=running_unit_tests
        )

        # Create MetagraphUpdaterClient for weight setting RPC
        from shared_objects.metagraph.metagraph_updater import MetagraphUpdaterClient
        self._metagraph_updater_client = MetagraphUpdaterClient(
            running_unit_tests=running_unit_tests
        )

        # Slack notifier (lazy initialization)
        self._external_slack_notifier = slack_notifier
        self._slack_notifier = None

        # Store results for external access
        self.checkpoint_results: List[Tuple[str, float]] = []
        self.transformed_list: List[Tuple[int, float]] = []
        self._results_lock = threading.Lock()

        # Start daemon if requested (deferred until all initialization complete)
        if start_daemon:
            self.start_daemon()

    # ==================== RPCServerBase Abstract Methods ====================

    def run_daemon_iteration(self) -> None:
        """
        Single iteration of daemon work. Called by RPCServerBase daemon loop.

        Computes weights and sends to MetagraphUpdater.
        """
        if not self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
            return

        bt.logging.info("Computing weights for RPC request")
        current_time = TimeUtil.now_in_millis()

        try:
            # Compute weights
            checkpoint_results, transformed_list = self.compute_weights_default(current_time)

            # Store results (thread-safe)
            with self._results_lock:
                self.checkpoint_results = checkpoint_results
                self.transformed_list = transformed_list

            if transformed_list:
                # Send weight setting request via RPC
                self._send_weight_request(transformed_list)
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

            # Send error notification
            if self.slack_notifier:
                compact_trace = ErrorUtils.get_compact_stacktrace(e)
                self.slack_notifier.send_message(
                    f"Weight calculator error!\n"
                    f"Error: {str(e)}\n"
                    f"Trace: {compact_trace}",
                    level="error"
                )
            time.sleep(30)

    # ==================== Properties ====================

    @property
    def metagraph(self):
        """Get metagraph client (forward compatibility - created internally)."""
        return self._metagraph_client

    @property
    def position_manager(self):
        """Get position manager client (forward compatibility - created internally)."""
        return self._position_client

    @property
    def contract_manager(self):
        """Get contract manager client (forward compatibility - created internally)."""
        return self._contract_client

    @property
    def slack_notifier(self):
        """Get slack notifier (lazy initialization)."""
        if self._external_slack_notifier:
            return self._external_slack_notifier

        if self._slack_notifier is None and self.config and self.hotkey:
            self._slack_notifier = SlackNotifier(
                hotkey=self.hotkey,
                webhook_url=getattr(self.config, 'slack_webhook_url', None),
                error_webhook_url=getattr(self.config, 'slack_error_webhook_url', None),
                is_miner=False
            )
        return self._slack_notifier

    @slack_notifier.setter
    def slack_notifier(self, value):
        """Set slack notifier (used by RPCServerBase during initialization)."""
        self._external_slack_notifier = value

    # ==================== RPC Methods (exposed to client) ====================

    def get_health_check_details(self) -> dict:
        """Add service-specific health check details."""
        with self._results_lock:
            n_results = len(self.checkpoint_results)
            n_weights = len(self.transformed_list)
        return {
            "num_checkpoint_results": n_results,
            "num_weights": n_weights
        }

    def get_checkpoint_results_rpc(self) -> list:
        """Get latest checkpoint results."""
        with self._results_lock:
            return list(self.checkpoint_results)

    def get_transformed_list_rpc(self) -> list:
        """Get latest transformed weight list."""
        with self._results_lock:
            return list(self.transformed_list)

    # ==================== Weight Calculation Logic ====================

    def compute_weights_default(self, current_time: int) -> Tuple[List[Tuple[str, float]], List[Tuple[int, float]]]:
        """
        Compute weights for all miners using debt-based scoring.

        Args:
            current_time: Current time in milliseconds

        Returns:
            Tuple of (checkpoint_results, transformed_list)
            - checkpoint_results: List of (hotkey, score) tuples
            - transformed_list: List of (uid, weight) tuples
        """
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        # Collect metagraph hotkeys to ensure we are only setting weights for miners in the metagraph
        metagraph_hotkeys = list(self.metagraph.get_hotkeys())
        metagraph_hotkeys_set = set(metagraph_hotkeys)
        hotkey_to_idx = {hotkey: idx for idx, hotkey in enumerate(metagraph_hotkeys)}

        # Get all miners from all buckets
        challenge_hotkeys = list(self._challengeperiod_client.get_hotkeys_by_bucket(MinerBucket.CHALLENGE))
        probation_hotkeys = list(self._challengeperiod_client.get_hotkeys_by_bucket(MinerBucket.PROBATION))
        plagiarism_hotkeys = list(self._challengeperiod_client.get_hotkeys_by_bucket(MinerBucket.PLAGIARISM))
        success_hotkeys = list(self._challengeperiod_client.get_hotkeys_by_bucket(MinerBucket.MAINCOMP))

        all_hotkeys = challenge_hotkeys + probation_hotkeys + plagiarism_hotkeys + success_hotkeys

        # Filter out zombie miners (miners in buckets but not in metagraph)
        all_hotkeys_before_filter = len(all_hotkeys)
        all_hotkeys = [hk for hk in all_hotkeys if hk in metagraph_hotkeys_set]
        zombies_filtered = all_hotkeys_before_filter - len(all_hotkeys)

        if zombies_filtered > 0:
            bt.logging.info(f"Filtered out {zombies_filtered} zombie miners (not in metagraph)")

        bt.logging.info(
            f"Computing weights for {len(all_hotkeys)} miners: "
            f"{len(success_hotkeys)} MAINCOMP, {len(probation_hotkeys)} PROBATION, "
            f"{len(challenge_hotkeys)} CHALLENGE, {len(plagiarism_hotkeys)} PLAGIARISM "
            f"({zombies_filtered} zombies filtered)"
        )

        # Compute weights for all miners using debt-based scoring
        checkpoint_netuid_weights, checkpoint_results = self._compute_miner_weights(
            all_hotkeys, hotkey_to_idx, current_time
        )

        if checkpoint_netuid_weights is None or len(checkpoint_netuid_weights) == 0:
            bt.logging.info("No weights computed. Do nothing for now.")
            return [], []

        transformed_list = checkpoint_netuid_weights
        bt.logging.info(f"transformed list: {transformed_list}")

        return checkpoint_results, transformed_list

    def _compute_miner_weights(
        self,
        hotkeys_to_compute_weights_for: List[str],
        hotkey_to_idx: dict,
        current_time: int
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[str, float]]]:
        """
        Compute weights for specified miners using debt-based scoring.

        Args:
            hotkeys_to_compute_weights_for: List of miner hotkeys
            hotkey_to_idx: Mapping of hotkey to metagraph index
            current_time: Current time in milliseconds

        Returns:
            Tuple of (netuid_weights, checkpoint_results)
        """
        if len(hotkeys_to_compute_weights_for) == 0:
            return [], []

        bt.logging.info("Calculating new subtensor weights using debt-based scoring...")

        # Get debt ledgers for the specified miners via RPC
        all_debt_ledgers = self._debt_ledger_client.get_all_debt_ledgers()
        filtered_debt_ledgers = {
            hotkey: ledger
            for hotkey, ledger in all_debt_ledgers.items()
            if hotkey in hotkeys_to_compute_weights_for
        }

        if len(filtered_debt_ledgers) == 0:
            total_ledgers = len(all_debt_ledgers)
            if total_ledgers == 0:
                bt.logging.info(
                    f"No debt ledgers loaded yet. "
                    f"Requested {len(hotkeys_to_compute_weights_for)} hotkeys. "
                    f"Debt ledger daemon likely still building initial data (120s delay + build time). "
                    f"Will retry in 5 minutes."
                )
            else:
                bt.logging.warning(
                    f"No debt ledgers found. "
                    f"Requested {len(hotkeys_to_compute_weights_for)} hotkeys, "
                    f"debt_ledger_client has {total_ledgers} ledgers loaded."
                )
            return [], []

        # Use debt-based scoring with shared metagraph
        checkpoint_results = DebtBasedScoring.compute_results(
            ledger_dict=filtered_debt_ledgers,
            metagraph=self.metagraph,
            challengeperiod_client=self._challengeperiod_client,
            contract_client=self._contract_client,
            current_time_ms=current_time,
            verbose=True,
            is_testnet=not self.is_mainnet
        )

        bt.logging.info(f"Debt-based scoring results: [{checkpoint_results}]")

        checkpoint_netuid_weights = []
        for miner, score in checkpoint_results:
            if miner in hotkey_to_idx:
                checkpoint_netuid_weights.append((
                    hotkey_to_idx[miner],
                    score
                ))
            else:
                bt.logging.error(f"Miner {miner} not found in the metagraph.")

        return checkpoint_netuid_weights, checkpoint_results

    def _send_weight_request(self, transformed_list: List[Tuple[int, float]]):
        """
        Send weight setting request to MetagraphUpdater via RPC.

        Args:
            transformed_list: List of (uid, weight) tuples
        """
        try:
            uids = [x[0] for x in transformed_list]
            weights = [x[1] for x in transformed_list]

            # Send request via RPC (synchronous - get success/failure feedback)
            result = self._metagraph_updater_client.set_weights_rpc(
                uids=uids,
                weights=weights,
                version_key=self.subnet_version
            )

            if result.get('success'):
                bt.logging.info(f"Weight request succeeded: {len(uids)} UIDs via RPC")
            else:
                error = result.get('error', 'Unknown error')
                bt.logging.error(f"Weight request failed: {error}")

                # NOTE: Don't send Slack alert here - MetagraphUpdater handles alerting
                # with proper benign error filtering (e.g., "too soon to commit weights").

        except Exception as e:
            bt.logging.error(f"Error sending weight request via RPC: {e}")
            bt.logging.error(traceback.format_exc())

            if self.slack_notifier:
                compact_trace = ErrorUtils.get_compact_stacktrace(e)
                self.slack_notifier.send_message(
                    f"Weight request RPC error!\n"
                    f"Error: {str(e)}\n"
                    f"Trace: {compact_trace}",
                    level="error"
                )


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

    The server creates its own clients internally (forward compatibility pattern):
    - CommonDataClient (for shutdown_dict)
    - MetagraphClient
    - PositionManagerClient
    - ChallengePeriodClient
    - ContractClient
    - DebtLedgerClient
    - MetagraphUpdaterClient (for weight setting RPC)

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
        start_daemon=True
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
