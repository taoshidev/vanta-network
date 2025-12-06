# developer: jbonilla
import time
import traceback
from setproctitle import setproctitle

import bittensor as bt

from shared_objects.slack_notifier import SlackNotifier
from time_util.time_util import TimeUtil
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from shared_objects.cache_controller import CacheController
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from shared_objects.error_utils import ErrorUtils
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.challenge_period.challengeperiod_client import ChallengePeriodClient
from vali_objects.contract.contract_client import ContractClient
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient


class SubtensorWeightSetter(CacheController):
    def __init__(self, connection_mode: "RPCConnectionMode" = RPCConnectionMode.RPC, is_backtesting=False, use_slack_notifier=False,
                 metagraph_updater_rpc=None, config=None, hotkey=None, is_mainnet=True):
        self.connection_mode = connection_mode
        running_unit_tests = connection_mode == RPCConnectionMode.LOCAL

        super().__init__(running_unit_tests=running_unit_tests, is_backtesting=is_backtesting, connection_mode=connection_mode)

        self._position_client = PositionManagerClient(
            port=ValiConfig.RPC_POSITIONMANAGER_PORT,
            connect_immediately=not running_unit_tests
        )
        self._challenge_period_client = ChallengePeriodClient(
            connection_mode=connection_mode
        )
        # Create own ContractClient (forward compatibility - no parameter passing)
        self._contract_client = ContractClient(running_unit_tests=running_unit_tests)
        # Note: perf_ledger_manager removed - no longer used (debt-based scoring uses debt_ledger_manager)
        self.subnet_version = 200
        # Store weights for use in backtesting
        self.checkpoint_results = []
        self.transformed_list = []
        self.use_slack_notifier = use_slack_notifier
        self._slack_notifier = None
        self.config = config
        self.hotkey = hotkey

        # Debt-based scoring dependencies
        # DebtLedgerClient provides encapsulated access to debt ledgers
        # In backtesting mode, delay connection until first use
        self._debt_ledger_client = DebtLedgerClient(
            connection_mode=connection_mode,
            connect_immediately=not is_backtesting
        )
        self.is_mainnet = is_mainnet

        # RPC client for weight setting (replaces queue)
        self.metagraph_updater_rpc = metagraph_updater_rpc

    @property
    def metagraph(self):
        """Get metagraph client (forward compatibility - created internally)."""
        return self._metagraph_client

    @property
    def slack_notifier(self):
        if self.use_slack_notifier and self._slack_notifier is None:
            self._slack_notifier = SlackNotifier(hotkey=self.hotkey,
                                                webhook_url=getattr(self.config, 'slack_webhook_url', None),
                                                error_webhook_url=getattr(self.config, 'slack_error_webhook_url', None),
                                                is_miner=False)  # This is a validator
        return self._slack_notifier

    @property
    def position_manager(self):
        """Get position manager client."""
        return self._position_client

    @property
    def contract_manager(self):
        """Get contract client (forward compatibility - created internally)."""
        return self._contract_client

    def compute_weights_default(self, current_time: int) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        # Collect metagraph hotkeys to ensure we are only setting weights for miners in the metagraph
        metagraph_hotkeys = list(self.metagraph.get_hotkeys())
        metagraph_hotkeys_set = set(metagraph_hotkeys)
        hotkey_to_idx = {hotkey: idx for idx, hotkey in enumerate(metagraph_hotkeys)}

        # Get all miners from all buckets
        challenge_hotkeys = list(self._challenge_period_client.get_hotkeys_by_bucket(MinerBucket.CHALLENGE))
        probation_hotkeys = list(self._challenge_period_client.get_hotkeys_by_bucket(MinerBucket.PROBATION))
        plagiarism_hotkeys = list(self._challenge_period_client.get_hotkeys_by_bucket(MinerBucket.PLAGIARISM))
        success_hotkeys = list(self._challenge_period_client.get_hotkeys_by_bucket(MinerBucket.MAINCOMP))

        # DebtBasedScoring handles all miners together - it applies:
        # - Debt-based weights for MAINCOMP/PROBATION (earning periods)
        # - Minimum dust weights for CHALLENGE/PLAGIARISM/UNKNOWN
        # - Burn address gets excess weight when sum < 1.0
        if self.is_backtesting:
            all_hotkeys = challenge_hotkeys + probation_hotkeys + plagiarism_hotkeys + success_hotkeys
        else:
            all_hotkeys = challenge_hotkeys + probation_hotkeys + plagiarism_hotkeys + success_hotkeys

        # Filter out zombie miners (miners in buckets but not in metagraph)
        # This can happen when miners deregister but haven't been pruned from active_miners yet
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
        # subcategory_min_days parameter no longer needed for debt-based scoring
        checkpoint_netuid_weights, checkpoint_results = self._compute_miner_weights(
            all_hotkeys, hotkey_to_idx, current_time, asset_class_min_days={}, scoring_challenge=False
        )

        if checkpoint_netuid_weights is None or len(checkpoint_netuid_weights) == 0:
            bt.logging.info("No weights computed. Do nothing for now.")
            return [], []

        transformed_list = checkpoint_netuid_weights
        bt.logging.info(f"transformed list: {transformed_list}")

        return checkpoint_results, transformed_list

    def _compute_miner_weights(self, hotkeys_to_compute_weights_for, hotkey_to_idx, current_time, asset_class_min_days, scoring_challenge: bool = False):
        miner_group = "challenge period" if scoring_challenge else "main competition"

        if len(hotkeys_to_compute_weights_for) == 0:
            return [], []

        bt.logging.info(f"Calculating new subtensor weights for {miner_group} using debt-based scoring...")

        # Filter debt ledgers to only include specified hotkeys
        # Get all debt ledgers via RPC
        all_debt_ledgers = self._debt_ledger_client.get_all_debt_ledgers()
        filtered_debt_ledgers = {
            hotkey: ledger
            for hotkey, ledger in all_debt_ledgers.items()
            if hotkey in hotkeys_to_compute_weights_for
        }

        if len(filtered_debt_ledgers) == 0:
            # Diagnostic logging to understand the mismatch
            total_ledgers = len(all_debt_ledgers)
            if total_ledgers == 0:
                bt.logging.info(
                    f"No debt ledgers loaded yet for {miner_group}. "
                    f"Requested {len(hotkeys_to_compute_weights_for)} hotkeys. "
                    f"Debt ledger daemon likely still building initial data (120s delay + build time). "
                    f"Will retry in 5 minutes."
                )
            else:
                bt.logging.warning(
                    f"No debt ledgers found for {miner_group}. "
                    f"Requested {len(hotkeys_to_compute_weights_for)} hotkeys, "
                    f"debt_ledger_server has {total_ledgers} ledgers loaded."
                )
                if hotkeys_to_compute_weights_for and all_debt_ledgers:
                    bt.logging.debug(
                        f"Sample requested hotkey: {hotkeys_to_compute_weights_for[0][:16]}..."
                    )
                    sample_available = list(all_debt_ledgers.keys())[0]
                    bt.logging.debug(f"Sample available hotkey: {sample_available[:16]}...")
            return [], []

        # Use debt-based scoring with shared metagraph
        # The metagraph contains substrate reserves refreshed by SubtensorOpsManager
        checkpoint_results = DebtBasedScoring.compute_results(
            ledger_dict=filtered_debt_ledgers,
            metagraph_client=self.metagraph,  # Shared metagraph with substrate reserves
            challengeperiod_client=self._challenge_period_client,
            contract_client=self._contract_client,  # For collateral-aware weight assignment
            current_time_ms=current_time,
            verbose=True,
            is_testnet=not self.is_mainnet
        )

        bt.logging.info(f"Debt-based scoring results for {miner_group}: [{checkpoint_results}]")

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

    def _store_weights(self, checkpoint_results: list[tuple[str, float]], transformed_list: list[tuple[str, float]]):
        self.checkpoint_results = checkpoint_results
        self.transformed_list = transformed_list

    def run_update_loop(self):
        """
        Weight setter loop that sends RPC requests to SubtensorOpsManager.
        """
        setproctitle(f"vali_{self.__class__.__name__}")
        bt.logging.enable_info()
        bt.logging.info("Starting weight setter update loop (RPC mode)")

        while not ShutdownCoordinator.is_shutdown():
            try:
                if self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
                    bt.logging.info("Computing weights for RPC request")
                    current_time = TimeUtil.now_in_millis()

                    # Compute weights (existing logic)
                    checkpoint_results, transformed_list = self.compute_weights_default(current_time)
                    self.checkpoint_results = checkpoint_results
                    self.transformed_list = transformed_list

                    if transformed_list and self.metagraph_updater_rpc:
                        # Send weight setting request via RPC (synchronous with feedback)
                        self.metagraph_updater_rpc._send_weight_request(transformed_list)
                        self.set_last_update_time()
                    else:
                        if not transformed_list:
                            bt.logging.warning(
                                "No weights computed (debt ledgers may still be initializing). "
                                "Waiting 5 minutes before retry..."
                            )
                        else:
                            bt.logging.debug("No RPC client available")

                        # Always sleep 5 minutes when weights aren't ready to avoid spam
                        time.sleep(300)

            except Exception as e:
                bt.logging.error(f"Error in weight setter update loop: {e}")
                bt.logging.error(traceback.format_exc())

                # Send error notification
                if self.slack_notifier:
                    # Get compact stack trace using shared utility
                    compact_trace = ErrorUtils.get_compact_stacktrace(e)
                    self.slack_notifier.send_message(
                        f"❌ Weight setter process error!\n"
                        f"Error: {str(e)}\n"
                        f"This occurred in the weight setter update loop\n"
                        f"Trace: {compact_trace}",
                        level="error"
                    )
                time.sleep(30)

            time.sleep(1)

        bt.logging.info("Weight setter update loop shutting down")
    
    def _send_weight_request(self, transformed_list):
        """Send weight setting request to SubtensorOpsManager via RPC (synchronous with feedback)"""
        try:
            uids = [x[0] for x in transformed_list]
            weights = [x[1] for x in transformed_list]

            # Send request via RPC (synchronous - get success/failure feedback)
            # SubtensorOpsManager will use its own config for netuid and wallet
            result = self.metagraph_updater_rpc.set_weights_rpc(
                uids=uids,
                weights=weights,
                version_key=self.subnet_version
            )

            if result.get('success'):
                bt.logging.info(f"✓ Weight request succeeded: {len(uids)} UIDs via RPC")
            else:
                error = result.get('error', 'Unknown error')
                bt.logging.error(f"✗ Weight request failed: {error}")

                # NOTE: Don't send Slack alert here - SubtensorOpsManager handles alerting
                # with proper benign error filtering (e.g., "too soon to commit weights").
                # Alerting here would create duplicate spam for normal/expected failures.

        except Exception as e:
            bt.logging.error(f"Error sending weight request via RPC: {e}")
            bt.logging.error(traceback.format_exc())

            # Send error notification
            if self.slack_notifier:
                # Get compact stack trace using shared utility
                compact_trace = ErrorUtils.get_compact_stacktrace(e)
                self.slack_notifier.send_message(
                    f"❌ Weight request RPC error!\n"
                    f"Error: {str(e)}\n"
                    f"This occurred while sending weight request via RPC\n"
                    f"Trace: {compact_trace}",
                    level="error"
                )

    def set_weights(self, current_time):
        # Compute weights (existing logic)
        checkpoint_results, transformed_list = self.compute_weights_default(current_time)
        self.checkpoint_results = checkpoint_results
        self.transformed_list = transformed_list

