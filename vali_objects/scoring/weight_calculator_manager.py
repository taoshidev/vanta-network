# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
WeightCalculatorManager - Core business logic for weight calculation and setting.

This manager handles all heavy logic for weight calculation operations.
WeightCalculatorServer wraps this and exposes methods via RPC.

This follows the same pattern as ChallengePeriodManager.
"""
import time
import traceback
import threading
from typing import List, Tuple

import bittensor as bt

from shared_objects.cache_controller import CacheController
from shared_objects.error_utils import ErrorUtils
from shared_objects.slack_notifier import SlackNotifier
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.enums.miner_bucket_enum import MinerBucket


class WeightCalculatorManager(CacheController):
    """
    Weight Calculator Manager - Contains all business logic for weight calculation.

    This manager is wrapped by WeightCalculatorServer which exposes methods via RPC.
    All heavy logic resides here - server delegates to this manager.

    Pattern:
    - Server holds a `self._manager` instance
    - Server delegates all RPC methods to manager methods
    - Manager creates its own clients internally (forward compatibility)
    """

    def __init__(
        self,
        *,
        is_backtesting=False,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC,
        config=None,
        hotkey=None,
        is_mainnet=True,
        slack_notifier=None
    ):
        """
        Initialize WeightCalculatorManager.

        Args:
            is_backtesting: Whether running in backtesting mode
            running_unit_tests: Whether running in test mode
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
            config: Validator config (for slack webhook URLs)
            hotkey: Validator hotkey
            is_mainnet: Whether running on mainnet
            slack_notifier: Optional external slack notifier
        """
        super().__init__(
            running_unit_tests=running_unit_tests,
            is_backtesting=is_backtesting,
            connection_mode=connection_mode
        )

        self.running_unit_tests = running_unit_tests
        self.connection_mode = connection_mode
        self.config = config
        self.hotkey = hotkey
        self.is_mainnet = is_mainnet
        self.subnet_version = 200

        # Create clients internally (forward compatibility - no parameter passing)
        from shared_objects.rpc.common_data_client import CommonDataClient
        self._common_data_client = CommonDataClient(
            running_unit_tests=running_unit_tests,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        from shared_objects.rpc.metagraph_client import MetagraphClient
        self._metagraph_client = MetagraphClient(
            running_unit_tests=running_unit_tests,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        from vali_objects.position_management.position_manager_client import PositionManagerClient
        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            running_unit_tests=running_unit_tests,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
        self._challengeperiod_client = ChallengePeriodClient(
            running_unit_tests=running_unit_tests,
            connection_mode=connection_mode
        )

        from vali_objects.miner_account.miner_account_client import MinerAccountClient
        self._miner_account_client = MinerAccountClient()

        from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
        self._debt_ledger_client = DebtLedgerClient(
            running_unit_tests=running_unit_tests,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        from shared_objects.subtensor_ops.subtensor_ops_client import SubtensorOpsClient
        self._subtensor_ops_client = SubtensorOpsClient(
            running_unit_tests=running_unit_tests,
            connect_immediately=False
        )

        # Slack notifier (lazy initialization)
        self._external_slack_notifier = slack_notifier
        self._slack_notifier = None

        # Store results for external access
        self.checkpoint_results: List[Tuple[str, float]] = []
        self.transformed_list: List[Tuple[int, float]] = []
        self._results_lock = threading.Lock()

        bt.logging.info("[WC_MANAGER] WeightCalculatorManager initialized")

    # ==================== Properties ====================

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

    # ==================== Core Business Logic ====================

    def compute_weights(self, current_time: int = None) -> Tuple[List[Tuple[str, float]], List[Tuple[int, float]]]:
        """
        Compute weights for all miners using debt-based scoring.

        This is the main entry point for weight calculation.

        Args:
            current_time: Current time in milliseconds. If None, uses TimeUtil.now_in_millis().

        Returns:
            Tuple of (checkpoint_results, transformed_list)
            - checkpoint_results: List of (hotkey, score) tuples
            - transformed_list: List of (uid, weight) tuples
        """
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        bt.logging.info("Computing weights for all miners")

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
            else:
                bt.logging.warning(
                    "No weights computed (debt ledgers may still be initializing). "
                    "Will retry later..."
                )

            return checkpoint_results, transformed_list

        except Exception as e:
            bt.logging.error(f"Error computing weights: {e}")
            bt.logging.error(traceback.format_exc())

            if self.slack_notifier:
                compact_trace = ErrorUtils.get_compact_stacktrace(e)
                self.slack_notifier.send_message(
                    f"Weight computation error!\n"
                    f"Error: {str(e)}\n"
                    f"Trace: {compact_trace}",
                    level="error"
                )
            raise

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
        metagraph_hotkeys = list(self._metagraph_client.get_hotkeys())
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
            metagraph_client=self._metagraph_client,
            challengeperiod_client=self._challengeperiod_client,
            miner_account_client=self._miner_account_client,
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
        Send weight setting request to SubtensorOpsManager via RPC.

        Args:
            transformed_list: List of (uid, weight) tuples
        """
        try:
            uids = [x[0] for x in transformed_list]
            weights = [x[1] for x in transformed_list]

            # Send request via RPC (synchronous - get success/failure feedback)
            result = self._subtensor_ops_client.set_weights_rpc(
                uids=uids,
                weights=weights,
                version_key=self.subnet_version
            )

            if result.get('success'):
                bt.logging.info(f"Weight request succeeded: {len(uids)} UIDs via RPC")
            else:
                error = result.get('error', 'Unknown error')
                bt.logging.error(f"Weight request failed: {error}")

                # NOTE: Don't send Slack alert here - SubtensorOpsManager handles alerting
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

    # ==================== Getter Methods ====================

    def get_checkpoint_results(self) -> List[Tuple[str, float]]:
        """Get latest checkpoint results (thread-safe)."""
        with self._results_lock:
            return list(self.checkpoint_results)

    def get_transformed_list(self) -> List[Tuple[int, float]]:
        """Get latest transformed weight list (thread-safe)."""
        with self._results_lock:
            return list(self.transformed_list)
