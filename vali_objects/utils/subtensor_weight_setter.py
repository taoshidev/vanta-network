# developer: jbonilla
import time
import traceback
from setproctitle import setproctitle

import bittensor as bt

from miner_objects.slack_notifier import SlackNotifier
from time_util.time_util import TimeUtil
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.miner_bucket_enum import MinerBucket
from vali_objects.vali_config import ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.utils.position_manager import PositionManager
from vali_objects.scoring.scoring import Scoring
from vali_objects.scoring.debt_based_scoring import DebtBasedScoring
from vali_objects.vali_dataclasses.perf_ledger import PerfLedger
from shared_objects.error_utils import ErrorUtils



class SubtensorWeightSetter(CacheController):
    def __init__(self, metagraph, position_manager: PositionManager,
                 running_unit_tests=False, is_backtesting=False, use_slack_notifier=False,
                 shutdown_dict=None, weight_request_queue=None, config=None, hotkey=None, contract_manager=None,
                 debt_ledger_manager=None, is_mainnet=True):
        super().__init__(metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)
        self.position_manager = position_manager
        self.perf_ledger_manager = position_manager.perf_ledger_manager
        self.subnet_version = 200
        # Store weights for use in backtesting
        self.checkpoint_results = []
        self.transformed_list = []
        self.use_slack_notifier = use_slack_notifier
        self._slack_notifier = None
        self.config = config
        self.hotkey = hotkey
        self.contract_manager = contract_manager

        # Debt-based scoring dependencies
        # DebtLedgerManager provides encapsulated access to IPC-shared debt_ledgers dict
        self.debt_ledger_manager = debt_ledger_manager
        self.is_mainnet = is_mainnet

        # IPC setup
        self.shutdown_dict = shutdown_dict if shutdown_dict is not None else {}
        self.weight_request_queue = weight_request_queue

    @property
    def slack_notifier(self):
        if self.use_slack_notifier and self._slack_notifier is None:
            self._slack_notifier = SlackNotifier(hotkey=self.hotkey,
                                                webhook_url=getattr(self.config, 'slack_webhook_url', None),
                                                error_webhook_url=getattr(self.config, 'slack_error_webhook_url', None),
                                                is_miner=False)  # This is a validator
        return self._slack_notifier

    def compute_weights_default(self, current_time: int) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        # Collect metagraph hotkeys to ensure we are only setting weights for miners in the metagraph
        metagraph_hotkeys = list(self.metagraph.get_hotkeys())
        metagraph_hotkeys_set = set(metagraph_hotkeys)
        hotkey_to_idx = {hotkey: idx for idx, hotkey in enumerate(metagraph_hotkeys)}

        # Get all miners from all buckets
        challenge_hotkeys = list(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.CHALLENGE))
        probation_hotkeys = list(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.PROBATION))
        plagiarism_hotkeys = list(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.PLAGIARISM))
        success_hotkeys = list(self.position_manager.challengeperiod_manager.get_hotkeys_by_bucket(MinerBucket.MAINCOMP))

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

        # Get debt ledgers for the specified miners
        # Access IPC-shared debt_ledgers dict through manager for proper encapsulation
        if self.debt_ledger_manager is None:
            bt.logging.warning("debt_ledger_manager not available for scoring")
            return [], []

        # Filter debt ledgers to only include specified hotkeys
        # debt_ledger_manager.debt_ledgers is an IPC-managed dict
        filtered_debt_ledgers = {
            hotkey: ledger
            for hotkey, ledger in self.debt_ledger_manager.debt_ledgers.items()
            if hotkey in hotkeys_to_compute_weights_for
        }

        if len(filtered_debt_ledgers) == 0:
            # Diagnostic logging to understand the mismatch
            total_ledgers = len(self.debt_ledger_manager.debt_ledgers)
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
                    f"debt_ledger_manager has {total_ledgers} ledgers loaded."
                )
                if hotkeys_to_compute_weights_for and self.debt_ledger_manager.debt_ledgers:
                    bt.logging.debug(
                        f"Sample requested hotkey: {hotkeys_to_compute_weights_for[0][:16]}..."
                    )
                    sample_available = list(self.debt_ledger_manager.debt_ledgers.keys())[0]
                    bt.logging.debug(f"Sample available hotkey: {sample_available[:16]}...")
            return [], []

        # Use debt-based scoring with shared metagraph
        # The metagraph contains substrate reserves refreshed by MetagraphUpdater
        checkpoint_results = DebtBasedScoring.compute_results(
            ledger_dict=filtered_debt_ledgers,
            metagraph=self.metagraph,  # Shared metagraph with substrate reserves
            challengeperiod_manager=self.position_manager.challengeperiod_manager,
            contract_manager=self.contract_manager,  # For collateral-aware weight assignment
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
        Weight setter loop that sends fire-and-forget requests to MetagraphUpdater.
        """
        setproctitle(f"vali_{self.__class__.__name__}")
        bt.logging.enable_info()
        bt.logging.info("Starting weight setter update loop (fire-and-forget IPC mode)")

        while not self.shutdown_dict:
            try:
                if self.refresh_allowed(ValiConfig.SET_WEIGHT_REFRESH_TIME_MS):
                    bt.logging.info("Computing weights for IPC request")
                    current_time = TimeUtil.now_in_millis()

                    # Compute weights (existing logic)
                    checkpoint_results, transformed_list = self.compute_weights_default(current_time)
                    self.checkpoint_results = checkpoint_results
                    self.transformed_list = transformed_list

                    if transformed_list and self.weight_request_queue:
                        # Send weight setting request (fire-and-forget)
                        self._send_weight_request(transformed_list)
                        self.set_last_update_time()
                    else:
                        # No weights computed - likely debt_ledger_manager not ready yet
                        # Sleep for 5 minutes to avoid busy looping and log spam
                        if self.debt_ledger_manager is None:
                            bt.logging.warning(
                                "debt_ledger_manager not available. "
                                "Waiting 5 minutes before retry..."
                            )
                        elif not transformed_list:
                            bt.logging.warning(
                                "No weights computed (debt ledgers may still be initializing). "
                                "Waiting 5 minutes before retry..."
                            )
                        else:
                            bt.logging.debug("No IPC queue available")

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
        """Send weight setting request to MetagraphUpdater (fire-and-forget)"""
        try:
            uids = [x[0] for x in transformed_list]
            weights = [x[1] for x in transformed_list]
            
            # Send request (no response expected)
            # MetagraphUpdater will use its own config for netuid and wallet
            request = {
                'uids': uids,
                'weights': weights,
                'version_key': self.subnet_version,
                'timestamp': TimeUtil.now_in_millis()
            }
            
            self.weight_request_queue.put_nowait(request)
            bt.logging.info(f"Weight request sent: {len(uids)} UIDs via IPC")
            
        except Exception as e:
            bt.logging.error(f"Error sending weight request: {e}")
            bt.logging.error(traceback.format_exc())

            # Send error notification
            if self.slack_notifier:
                # Get compact stack trace using shared utility
                compact_trace = ErrorUtils.get_compact_stacktrace(e)
                self.slack_notifier.send_message(
                    f"❌ Weight request IPC error!\n"
                    f"Error: {str(e)}\n"
                    f"This occurred while sending weight request via IPC\n"
                    f"Trace: {compact_trace}",
                    level="error"
                )

    def set_weights(self, current_time):
        # Compute weights (existing logic)
        checkpoint_results, transformed_list = self.compute_weights_default(current_time)
        self.checkpoint_results = checkpoint_results
        self.transformed_list = transformed_list

