# developer: trdougherty, jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ChallengePeriodManager - Core business logic for challenge period management.

This manager handles all heavy logic for challenge period operations.
ChallengePeriodServer wraps this and exposes methods via RPC.

This follows the same pattern as EliminationManager.
"""
import time

import bittensor as bt
import threading
import copy
from typing import Dict, Optional, Tuple
from datetime import datetime

from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePairCategory, ValiConfig, RPCConnectionMode
from vali_objects.utils.asset_selection.asset_selection_manager import ASSET_CLASS_SELECTION_TIME_MS
from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
from shared_objects.cache_controller import CacheController
from vali_objects.scoring.scoring import Scoring
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger, TP_ID_PORTFOLIO
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.elimination.elimination_manager import EliminationReason
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.plagiarism.plagiarism_client import PlagiarismClient
from vali_objects.contract.contract_client import ContractClient
from shared_objects.rpc.common_data_client import CommonDataClient


class ChallengePeriodManager(CacheController):
    """
    Challenge Period Manager - Contains all business logic for challenge period management.

    This manager is wrapped by ChallengePeriodServer which exposes methods via RPC.
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
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize ChallengePeriodManager.

        Args:
            is_backtesting: Whether running in backtesting mode
            running_unit_tests: Whether running in test mode
            connection_mode: RPCConnectionMode.LOCAL for tests, RPCConnectionMode.RPC for production
        """
        super().__init__(running_unit_tests=running_unit_tests, is_backtesting=is_backtesting, connection_mode=connection_mode)

        self.running_unit_tests = running_unit_tests
        self.connection_mode = connection_mode

        # Create clients internally (forward compatibility - no parameter passing)
        self._perf_ledger_client = PerfLedgerClient(
            connection_mode=connection_mode,
            connect_immediately=False,
            running_unit_tests=running_unit_tests
        )

        self._position_client = PositionManagerClient(
            connect_immediately=False,
            connection_mode=connection_mode
        )

        self.elim_client = EliminationClient(
            connection_mode=connection_mode,
            connect_immediately=False
        )

        self._plagiarism_client = PlagiarismClient(
            connection_mode=connection_mode,
            connect_immediately=False
        )

        self._contract_client = ContractClient(
            connection_mode=connection_mode,
            connect_immediately=False
        )

        # Create own CommonDataClient (forward compatibility - no parameter passing)
        self._common_data_client = CommonDataClient(
            connect_immediately=False,
            connection_mode=connection_mode
        )

        # Create AssetSelectionClient for asset class selection support
        self.asset_selection_client = AssetSelectionClient(
            connect_immediately=False,
            connection_mode=connection_mode
        )

        # Local dicts (NOT IPC managerized) - much faster!
        self.eliminations_with_reasons: Dict[str, Tuple[str, float]] = {}
        self.active_miners: Dict[str, Tuple[MinerBucket, int, Optional[MinerBucket], Optional[int]]] = {}

        # Local lock (NOT shared across processes) - RPC methods are auto-serialized
        self.eliminations_lock = threading.Lock()

        self.CHALLENGE_FILE = ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=running_unit_tests)

        # Load initial active_miners from disk
        initial_active_miners = {}
        if not self.is_backtesting:
            disk_data = ValiUtils.get_vali_json_file_dict(self.CHALLENGE_FILE)
            initial_active_miners = self.parse_checkpoint_dict(disk_data)

        self.active_miners = initial_active_miners

        if not self.is_backtesting and len(self.active_miners) == 0:
            self._write_challengeperiod_from_memory_to_disk()

        self.refreshed_challengeperiod_start_time = False

        bt.logging.info("[CP_MANAGER] ChallengePeriodManager initialized with local dicts (no IPC)")

    # ==================== Core Business Logic ====================

    def refresh(self, current_time: int = None, iteration_epoch=None):
        """
        Refresh the challenge period manager.

        Args:
            current_time: Current time in milliseconds. If None, uses TimeUtil.now_in_millis().
            iteration_epoch: Epoch captured at start of iteration. Used to detect stale data.
        """
        if current_time is None:
            current_time = TimeUtil.now_in_millis()

        if not self.refresh_allowed(ValiConfig.CHALLENGE_PERIOD_REFRESH_TIME_MS):
            time.sleep(1)
            return
        bt.logging.info("Refreshing challenge period")

        # Store iteration epoch for this refresh cycle
        self._current_iteration_epoch = iteration_epoch

        # Read current eliminations
        eliminations = self.elim_client.get_eliminations_from_memory()

        self.update_plagiarism_miners(current_time, self.get_plagiarism_miners())

        # Collect challenge period and update with new eliminations criteria
        self.remove_eliminated(eliminations=eliminations)

        hk_to_positions, hk_to_first_order_time = self._position_client.filtered_positions_for_scoring(
            hotkeys=self._metagraph_client.get_hotkeys()
        )

        # Add to testing if not in eliminated, already in the challenge period, or in the new eliminations list
        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self._metagraph_client.get_hotkeys(),
            eliminations=eliminations,
            hk_to_first_order_time=hk_to_first_order_time,
            default_time=current_time
        )

        challengeperiod_success_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)
        challengeperiod_testing_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.CHALLENGE)
        challengeperiod_probation_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.PROBATION)
        all_miners = challengeperiod_success_hotkeys + challengeperiod_testing_hotkeys + challengeperiod_probation_hotkeys

        if not self.refreshed_challengeperiod_start_time:
            self.refreshed_challengeperiod_start_time = True
            self._refresh_challengeperiod_start_time(hk_to_first_order_time)

        ledger = self._perf_ledger_client.filtered_ledger_for_scoring(hotkeys=all_miners, portfolio_only=False)

        inspection_miners = self.get_testing_miners() | self.get_probation_miners()
        challengeperiod_success, challengeperiod_demoted, challengeperiod_eliminations = self.inspect(
            positions=hk_to_positions,
            ledger=ledger,
            success_hotkeys=challengeperiod_success_hotkeys,
            probation_hotkeys=challengeperiod_probation_hotkeys,
            inspection_hotkeys=inspection_miners,
            current_time=current_time,
            hk_to_first_order_time=hk_to_first_order_time
        )

        # Update plagiarism eliminations
        plagiarism_elim_miners = self.prepare_plagiarism_elimination_miners(current_time=current_time)
        challengeperiod_eliminations.update(plagiarism_elim_miners)

        # Update elimination reasons atomically
        self.update_elimination_reasons(challengeperiod_eliminations)

        any_changes = bool(challengeperiod_success) or bool(challengeperiod_eliminations) or bool(challengeperiod_demoted)

        # Moves challenge period testing to challenge period success in memory
        self._promote_challengeperiod_in_memory(challengeperiod_success, current_time)
        self._demote_challengeperiod_in_memory(challengeperiod_demoted, current_time)
        self._eliminate_challengeperiod_in_memory(eliminations_with_reasons=challengeperiod_eliminations)

        # Remove any miners who are no longer in the metagraph
        any_changes |= self._prune_deregistered_metagraph()

        # Sync challenge period with disk
        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

        # Clear iteration epoch after refresh completes
        self._current_iteration_epoch = None

        self.set_last_update_time()

        bt.logging.info(
            "Challenge Period snapshot after refresh "
            f"(MAINCOMP, {len(self.get_success_miners())}) "
            f"(PROBATION, {len(self.get_probation_miners())}) "
            f"(CHALLENGE, {len(self.get_testing_miners())}) "
            f"(PLAGIARISM, {len(self.get_plagiarism_miners())})"
        )

    def _prune_deregistered_metagraph(self, hotkeys=None) -> bool:
        """Prune the challenge period of all miners who are no longer in the metagraph."""
        if not hotkeys:
            hotkeys = self._metagraph_client.get_hotkeys()

        any_changes = False
        for hotkey in self.get_all_miner_hotkeys():
            if hotkey not in hotkeys:
                self.remove_miner(hotkey)
                any_changes = True

        return any_changes

    @staticmethod
    def is_recently_re_registered(ledger, hotkey, hk_to_first_order_time):
        """Check if a miner recently re-registered (edge case detection)."""
        if not hk_to_first_order_time:
            return False
        if ledger:
            time_of_ledger_start = ledger.start_time_ms
        else:
            return False

        first_order_time = hk_to_first_order_time.get(hotkey, None)
        if first_order_time is None:
            msg = f'No positions for hotkey {hotkey} - ledger start time: {time_of_ledger_start}'
            print(msg)
            return True

        # A perf ledger can never begin before the first order
        ans = time_of_ledger_start < first_order_time
        if ans:
            msg = (f'Hotkey {hotkey} has a ledger start time of {TimeUtil.millis_to_formatted_date_str(time_of_ledger_start)},'
                   f' a first order time of {TimeUtil.millis_to_formatted_date_str(first_order_time)}, and an'
                   f' initialization time of {TimeUtil.millis_to_formatted_date_str(ledger.initialization_time_ms)}.')
        return ans

    def inspect(
        self,
        positions: dict[str, list[Position]],
        ledger: dict[str, dict[str, PerfLedger]],
        success_hotkeys: list[str],
        probation_hotkeys: list[str],
        inspection_hotkeys: dict[str, int],
        current_time: int,
        hk_to_first_order_time: dict[str, int] | None = None,
        combined_scores_dict: dict[TradePairCategory, dict] | None = None,
    ) -> tuple[list[str], list[str], dict[str, tuple[str, float]]]:
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period. Does not modify the challenge period in memory.

        Args:
            combined_scores_dict (dict[TradePairCategory, dict] | None) - Optional pre-computed scores dict for testing.
                If provided, skips score calculation. Useful for unit tests.

        Returns:
            hotkeys_to_promote - list of miners that should be promoted from challenge/probation to maincomp
            hotkeys_to_demote - list of miners whose scores were lower than the threshold rank, to be demoted to probation
            miners_to_eliminate - dictionary of hotkey to a tuple of the form (reason failed challenge period, maximum drawdown)
        """
        if len(inspection_hotkeys) == 0:
            return [], [], {}  # no hotkeys to inspect

        if not current_time:
            current_time = TimeUtil.now_in_millis()

        miners_to_eliminate = {}
        miners_not_enough_positions = []

        # Used for checking base cases
        portfolio_only_ledgers = {}
        for hotkey, asset_ledgers in ledger.items():
            if asset_ledgers is not None:
                if isinstance(asset_ledgers, dict):
                    portfolio_only_ledgers[hotkey] = asset_ledgers.get(TP_ID_PORTFOLIO)
                else:
                    raise TypeError(f"Expected asset_ledgers to be dict, got {type(asset_ledgers)}")

        promotion_eligible_hotkeys = []
        rank_eligible_hotkeys = []

        for hotkey, bucket_start_time in inspection_hotkeys.items():
            if not self.running_unit_tests and ChallengePeriodManager.is_recently_re_registered(portfolio_only_ledgers.get(hotkey), hotkey, hk_to_first_order_time):
                bt.logging.warning(f'Found a re-registered hotkey with a perf ledger. Alert the team ASAP {hotkey}')
                continue

            if bucket_start_time is None:
                bt.logging.warning(f'Hotkey {hotkey} has no inspection time. Unexpected.')
                continue

            miner_bucket = self.get_miner_bucket(hotkey)
            before_challenge_end = self.meets_time_criteria(current_time, bucket_start_time, miner_bucket)
            if not before_challenge_end:
                bt.logging.info(f'Hotkey {hotkey} has failed the {miner_bucket.value} period due to time. cp_failed')
                miners_to_eliminate[hotkey] = (EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value, -1)
                continue

            # Get hotkey to ledger dict that only includes the inspection miner
            has_minimum_ledger, inspection_ledger = ChallengePeriodManager.screen_minimum_ledger(portfolio_only_ledgers, hotkey)
            if not has_minimum_ledger:
                continue

            # This step we want to check their drawdown. If they fail, we can move on.
            # inspection_ledger is the PerfLedger object for this hotkey (not a dict)
            exceeds_max_drawdown, recorded_drawdown_percentage = LedgerUtils.is_beyond_max_drawdown(inspection_ledger)
            if exceeds_max_drawdown:
                bt.logging.info(f'Hotkey {hotkey} has failed the {miner_bucket.value} period due to drawdown {recorded_drawdown_percentage}. cp_failed')
                miners_to_eliminate[hotkey] = (EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value, recorded_drawdown_percentage)
                continue

            # Get hotkey to positions dict that only includes the inspection miner
            has_minimum_positions, inspection_positions = ChallengePeriodManager.screen_minimum_positions(positions, hotkey)
            if not has_minimum_positions:
                miners_not_enough_positions.append(hotkey)
                continue

            # Check if miner has selected an asset class (only enforce after selection time)
            if current_time >= ASSET_CLASS_SELECTION_TIME_MS and not self.asset_selection_client.get_asset_selection(hotkey):
                continue

            # Miner passed basic checks - include in ranking for accurate threshold calculation
            rank_eligible_hotkeys.append(hotkey)

            # Additional check for promotion eligibility: minimum trading days
            if self.screen_minimum_interaction(inspection_ledger):
                promotion_eligible_hotkeys.append(hotkey)

        # Calculate dynamic minimum participation days for asset classes
        combined_hotkeys = set(success_hotkeys + probation_hotkeys)
        maincomp_ledger = {hotkey: ledger_data for hotkey, ledger_data in ledger.items() if hotkey in combined_hotkeys}
        asset_classes = list(AssetSegmentation.distill_asset_classes(ValiConfig.ASSET_CLASS_BREAKDOWN))
        asset_class_min_days = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            maincomp_ledger, asset_classes
        )
        bt.logging.info(f"challengeperiod_manager asset class minimum days: {asset_class_min_days}")

        all_miner_account_sizes = self._contract_client.get_all_miner_account_sizes(timestamp_ms=current_time)

        # Use provided scores dict if available (for testing), otherwise compute scores
        if combined_scores_dict is None:
            # Score all rank-eligible miners (including those without minimum days) for accurate threshold
            scoring_hotkeys = success_hotkeys + rank_eligible_hotkeys
            scoring_ledgers = {hotkey: ledger for hotkey, ledger in ledger.items() if hotkey in scoring_hotkeys}
            scoring_positions = {hotkey: pos_list for hotkey, pos_list in positions.items() if hotkey in scoring_hotkeys}

            combined_scores_dict = Scoring.score_miners(
                ledger_dict=scoring_ledgers,
                positions=scoring_positions,
                asset_class_min_days=asset_class_min_days,
                evaluation_time_ms=current_time,
                weighting=True,
                all_miner_account_sizes=all_miner_account_sizes
            )

        hotkeys_to_promote, hotkeys_to_demote = self.evaluate_promotions(
            success_hotkeys,
            promotion_eligible_hotkeys,
            combined_scores_dict
        )

        bt.logging.info(f"Challenge Period: evaluated {len(promotion_eligible_hotkeys)}/{len(inspection_hotkeys)} miners eligible for promotion")
        bt.logging.info(f"Challenge Period: evaluated {len(success_hotkeys)} miners eligible for demotion")
        bt.logging.info(f"Hotkeys to promote: {hotkeys_to_promote}")
        bt.logging.info(f"Hotkeys to demote: {hotkeys_to_demote}")
        bt.logging.info(f"Hotkeys to eliminate: {list(miners_to_eliminate.keys())}")
        bt.logging.info(f"Miners with no positions (skipped): {len(miners_not_enough_positions)}")

        return hotkeys_to_promote, hotkeys_to_demote, miners_to_eliminate

    def evaluate_promotions(
            self,
            success_hotkeys,
            promotion_eligible_hotkeys,
            combined_scores_dict
            ) -> tuple[list[str], list[str]]:

        # score them based on asset class
        asset_combined_scores = Scoring.combine_scores(combined_scores_dict)
        asset_softmaxed_scores = Scoring.softmax_by_asset(asset_combined_scores)

        # Get asset class selections for filtering during threshold calculation
        miner_asset_selections = {}
        all_selections = self.asset_selection_client.get_all_miner_selections()
        for hotkey, selection in all_selections.items():
            if isinstance(selection, str):
                miner_asset_selections[hotkey] = TradePairCategory(selection)
            else:
                miner_asset_selections[hotkey] = selection

        maincomp_hotkeys = set()
        promotion_threshold_rank = ValiConfig.PROMOTION_THRESHOLD_RANK
        for asset_class, asset_scores in asset_softmaxed_scores.items():
            # Filter to only include miners who selected this asset class when calculating threshold
            if miner_asset_selections:
                miner_scores = {
                    hotkey: score for hotkey, score in asset_scores.items()
                    if miner_asset_selections.get(hotkey) == asset_class
                }
            else:
                miner_scores = asset_scores

            # threshold_score = 0
            sorted_scores = sorted(miner_scores.items(), key=lambda item: item[1], reverse=True)

            # Only take miners with positive scores
            top_miners = [(hotkey, score) for hotkey, score in sorted_scores[:promotion_threshold_rank] if score >= 0]
            maincomp_hotkeys.update({hotkey for hotkey, _ in top_miners})

            bt.logging.info(f"{asset_class}: {len(sorted_scores)} miners ranked for evaluation")

            # Logging for missing hotkeys
            for hotkey in success_hotkeys:
                if hotkey not in asset_scores:
                    bt.logging.warning(f"Could not find MAINCOMP hotkey {hotkey} when scoring, miner will not be evaluated")
            for hotkey in promotion_eligible_hotkeys:
                if hotkey not in asset_scores:
                    bt.logging.warning(
                        f"Could not find CHALLENGE/PROBATION hotkey {hotkey} when scoring, miner will not be evaluated")

        # Only promote miners who are in top ranks AND are valid candidates (passed minimum days)
        promote_hotkeys = (maincomp_hotkeys - set(success_hotkeys)) & set(promotion_eligible_hotkeys)
        demote_hotkeys = set(success_hotkeys) - maincomp_hotkeys

        return list(promote_hotkeys), list(demote_hotkeys)

    @staticmethod
    def screen_minimum_interaction(ledger_element) -> bool:
        """Check if miner has minimum number of trading days."""
        if ledger_element is None:
            bt.logging.warning("Ledger element is None. Returning False.")
            return False

        miner_returns = LedgerUtils.daily_return_log(ledger_element)
        return len(miner_returns) >= ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS

    def meets_time_criteria(self, current_time, bucket_start_time, bucket):
        if bucket == MinerBucket.MAINCOMP:
            return False

        if bucket == MinerBucket.CHALLENGE:
            probation_end_time_ms = bucket_start_time + ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS
            return current_time <= probation_end_time_ms

        if bucket == MinerBucket.PROBATION:
            probation_end_time_ms = bucket_start_time + ValiConfig.PROBATION_MAXIMUM_MS
            return current_time <= probation_end_time_ms

    @staticmethod
    def screen_minimum_ledger(
        ledger: dict[str, PerfLedger],
        inspection_hotkey: str
    ) -> tuple[bool, PerfLedger] | tuple[bool, None]:
        """Ensure there is enough ledger data for the specific miner."""
        # Note: Caller should check if ledger dict is empty before calling this in a loop
        if ledger is None or len(ledger) == 0:
            return False, None

        single_ledger = ledger.get(inspection_hotkey, None)
        if single_ledger is None:
            return False, None

        has_minimum_ledger = len(single_ledger.cps) > 0

        if not has_minimum_ledger:
            bt.logging.debug(f"Hotkey: {inspection_hotkey} doesn't have the minimum ledger for challenge period.")

        return has_minimum_ledger, single_ledger

    @staticmethod
    def screen_minimum_positions(
        positions: dict[str, list[Position]],
        inspection_hotkey: str
    ) -> tuple[bool, dict[str, list[Position]]]:
        """Ensure there are enough positions for the specific miner."""
        if positions is None or len(positions) == 0:
            bt.logging.info(f"No positions for any miner to evaluate for challenge period. positions: {positions}")
            return False, {}

        positions_list = positions.get(inspection_hotkey, None)
        has_minimum_positions = positions_list is not None and len(positions_list) > 0

        inspection_positions = {inspection_hotkey: positions_list} if has_minimum_positions else {}

        return has_minimum_positions, inspection_positions

    def sync_challenge_period_data(self, active_miners_sync):
        """Sync challenge period data from another validator."""
        if not active_miners_sync:
            bt.logging.error(f'challenge_period_data {active_miners_sync} appears invalid')

        synced_miners = self.parse_checkpoint_dict(active_miners_sync)

        self.clear_active_miners()
        self.update_active_miners(synced_miners)
        self._write_challengeperiod_from_memory_to_disk()

    def get_hotkeys_by_bucket(self, bucket: MinerBucket) -> list[str]:
        """Get all hotkeys in a specific bucket."""
        return [hotkey for hotkey, (b, _, _, _) in self.active_miners.items() if b == bucket]

    def _remove_eliminated_from_memory(self, eliminations: list[dict] = None) -> bool:
        """Remove eliminated miners from memory."""
        if eliminations:
            eliminations_hotkeys = set([x['hotkey'] for x in eliminations])
        else:
            eliminations_hotkeys = self.elim_client.get_eliminated_hotkeys()

        bt.logging.info(f"[CP_DEBUG] _remove_eliminated_from_memory processing {len(eliminations_hotkeys)} eliminated hotkeys")

        any_changes = False
        for hotkey in eliminations_hotkeys:
            if self.has_miner(hotkey):
                bt.logging.info(f"[CP_DEBUG] Removing already-eliminated hotkey {hotkey} from active_miners")
                self.remove_miner(hotkey)
                any_changes = True

        return any_changes

    def remove_eliminated(self, eliminations=None):
        """Remove eliminated miners and sync to disk."""
        any_changes = self._remove_eliminated_from_memory(eliminations=eliminations)
        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

    def _clear_challengeperiod_in_memory_and_disk(self):
        """Clear all challenge period data."""
        if not self.running_unit_tests:
            raise Exception("Clearing challenge period is only allowed during unit tests.")
        self.clear_active_miners()
        self.clear_elimination_reasons()  # CRITICAL: Also clear elimination reasons for test isolation
        self._write_challengeperiod_from_memory_to_disk()

    def update_plagiarism_miners(self, current_time, plagiarism_miners):
        """Update plagiarism miners status."""
        new_plagiarism_miners, whitelisted_miners = self._plagiarism_client.update_plagiarism_miners(
            current_time, plagiarism_miners
        )
        self._demote_plagiarism_in_memory(new_plagiarism_miners, current_time)
        self._promote_plagiarism_to_previous_bucket_in_memory(whitelisted_miners, current_time)

    def prepare_plagiarism_elimination_miners(self, current_time):
        """Prepare plagiarism miners for elimination."""
        miners_to_eliminate = self._plagiarism_client.plagiarism_miners_to_eliminate(current_time)
        elim_miners_to_return = {}
        for hotkey in miners_to_eliminate:
            if self.has_miner(hotkey):
                bt.logging.info(
                    f'Hotkey {hotkey} is overdue in {MinerBucket.PLAGIARISM} at time {current_time}')
                elim_miners_to_return[hotkey] = (EliminationReason.PLAGIARISM.value, -1)
                self._plagiarism_client.send_plagiarism_elimination_notification(hotkey)

        return elim_miners_to_return

    def _promote_challengeperiod_in_memory(self, hotkeys: list[str], current_time: int):
        """Promote miners to main competition."""
        if len(hotkeys) > 0:
            bt.logging.info(f"Promoting {len(hotkeys)} miners to main competition.")

        for hotkey in hotkeys:
            bucket_value = self.get_miner_bucket(hotkey)
            if bucket_value is None:
                bt.logging.error(f"Hotkey {hotkey} is not an active miner. Skipping promotion")
                continue
            bt.logging.info(f"Promoting {hotkey} from {self.get_miner_bucket(hotkey).value} to MAINCOMP")
            self.set_miner_bucket(hotkey, MinerBucket.MAINCOMP, current_time)

    def _promote_plagiarism_to_previous_bucket_in_memory(self, hotkeys: list[str], current_time):
        """Promote plagiarism miners to their previous bucket."""
        if len(hotkeys) > 0:
            bt.logging.info(f"Promoting {len(hotkeys)} plagiarism miners to probation.")

        for hotkey in hotkeys:
            try:
                bucket_value = self.get_miner_bucket(hotkey)
                if bucket_value is None or bucket_value != MinerBucket.PLAGIARISM:
                    bt.logging.error(f"Hotkey {hotkey} is not an active miner. Skipping promotion")
                    continue

                previous_bucket = self.get_miner_previous_bucket(hotkey)
                previous_time = self.get_miner_previous_time(hotkey)

                bt.logging.info(f"Promoting {hotkey} from {bucket_value.value} to {previous_bucket.value} with time {previous_time}")
                self.set_miner_bucket(hotkey, previous_bucket, previous_time)

                # Send Slack notification
                self._plagiarism_client.send_plagiarism_promotion_notification(hotkey)
            except Exception as e:
                bt.logging.error(f"Failed to promote {hotkey} from plagiarism at time {current_time}: {e}")

    def _eliminate_challengeperiod_in_memory(self, eliminations_with_reasons: dict[str, tuple[str, float]]):
        """Eliminate miners from challenge period."""
        hotkeys = eliminations_with_reasons.keys()
        if hotkeys:
            bt.logging.info(f"[CP_DEBUG] Removing {len(hotkeys)} hotkeys from challenge period: {list(hotkeys)}")
            bt.logging.info(f"[CP_DEBUG] active_miners has {len(self.active_miners)} entries before elimination")

        for hotkey in hotkeys:
            if self.has_miner(hotkey):
                bucket = self.get_miner_bucket(hotkey)
                bt.logging.info(f"[CP_DEBUG] Eliminating {hotkey} from bucket {bucket.value}")
                self.remove_miner(hotkey)

                # Verify deletion
                if not self.has_miner(hotkey):
                    bt.logging.info(f"[CP_DEBUG] ✓ Verified {hotkey} was removed from active_miners")
                else:
                    bt.logging.error(f"[CP_DEBUG] ✗ FAILED to remove {hotkey} from active_miners!")
            else:
                bt.logging.error(f"[CP_DEBUG] Hotkey {hotkey} was not in active_miners but elimination was attempted. active_miners keys: {self.get_all_miner_hotkeys()}")

    def _demote_challengeperiod_in_memory(self, hotkeys: list[str], current_time):
        """Demote miners to probation."""
        if hotkeys:
            bt.logging.info(f"Demoting {len(hotkeys)} miners to probation")

        for hotkey in hotkeys:
            bucket_value = self.get_miner_bucket(hotkey)
            if bucket_value is None:
                bt.logging.error(f"Hotkey {hotkey} is not an active miner. Skipping demotion")
                continue
            bt.logging.info(f"Demoting {hotkey} to PROBATION")
            self.set_miner_bucket(hotkey, MinerBucket.PROBATION, current_time)

    def _demote_plagiarism_in_memory(self, hotkeys: list[str], current_time):
        """Demote miners to plagiarism bucket."""
        for hotkey in hotkeys:
            try:
                prev_bucket_value = self.get_miner_bucket(hotkey)
                if prev_bucket_value is None:
                    continue
                prev_bucket_time = self.get_miner_start_time(hotkey)
                bt.logging.info(f"Demoting {hotkey} to PLAGIARISM from {prev_bucket_value}")
                # Maintain previous state to make reverting easier
                self.set_miner_bucket(hotkey, MinerBucket.PLAGIARISM, current_time, prev_bucket_value, prev_bucket_time)

                # Send Slack notification
                self._plagiarism_client.send_plagiarism_demotion_notification(hotkey)
            except Exception as e:
                bt.logging.error(f"Failed to demote {hotkey} for plagiarism at time {current_time}: {e}")

    def _write_challengeperiod_from_memory_to_disk(self):
        """Write challenge period data from memory to disk."""
        if self.is_backtesting:
            return

        # Epoch-based validation: check if sync occurred during our iteration
        if hasattr(self, '_current_iteration_epoch') and self._current_iteration_epoch is not None:
            current_epoch = self._common_data_client.get_sync_epoch()
            if current_epoch != self._current_iteration_epoch:
                bt.logging.warning(
                    f"Sync occurred during ChallengePeriodManager iteration "
                    f"(epoch {self._current_iteration_epoch} -> {current_epoch}). "
                    f"Skipping save to avoid data corruption"
                )
                return

        challengeperiod_data = self.to_checkpoint_dict()
        ValiBkpUtils.write_file(self.CHALLENGE_FILE, challengeperiod_data)

    def _add_challengeperiod_testing_in_memory_and_disk(
        self,
        new_hotkeys: list[str],
        eliminations: list[dict],
        hk_to_first_order_time: dict[str, int],
        default_time: int
    ):
        """Add miners to challenge period testing."""
        if not eliminations:
            eliminations = self.elim_client.get_eliminations_from_memory()

        elimination_hotkeys = set(x['hotkey'] for x in eliminations)

        # Get local eliminations that haven't been persisted yet
        with self.eliminations_lock:
            local_elimination_hotkeys = set(self.eliminations_with_reasons.keys())

        maincomp_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)
        probation_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.PROBATION)
        plagiarism_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.PLAGIARISM)

        any_changes = False
        for hotkey in new_hotkeys:
            # Skip if miner is in persisted eliminations
            if hotkey in elimination_hotkeys:
                continue

            # Skip if miner is in local eliminations
            if hotkey in local_elimination_hotkeys:
                bt.logging.info(f"[CP_DEBUG] Skipping {hotkey[:16]}...{hotkey[-8:]} - in eliminations_with_reasons (not yet persisted)")
                continue

            if hotkey in maincomp_hotkeys or hotkey in probation_hotkeys or hotkey in plagiarism_hotkeys:
                continue

            first_order_time = hk_to_first_order_time.get(hotkey)
            if first_order_time is None:
                if not self.has_miner(hotkey):
                    self.set_miner_bucket(hotkey, MinerBucket.CHALLENGE, default_time)
                    bt.logging.info(f"Adding {hotkey} to challenge period with start time {default_time}")
                    any_changes = True
                continue

            # Has a first order time but not yet stored in memory or start time is set as default
            start_time = self.get_miner_start_time(hotkey)
            if not self.has_miner(hotkey) or start_time != first_order_time:
                self.set_miner_bucket(hotkey, MinerBucket.CHALLENGE, first_order_time)
                bt.logging.info(f"Adding {hotkey} to challenge period with first order time {first_order_time}")
                any_changes = True

        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

    def _refresh_challengeperiod_start_time(self, hk_to_first_order_time_ms: dict[str, int]):
        """Retroactively update the challengeperiod_testing start time based on time of first order."""
        bt.logging.info("Refreshing challengeperiod start times")

        any_changes = False
        for hotkey in self.get_testing_miners().keys():
            start_time_ms = self.get_miner_start_time(hotkey)
            if hotkey not in hk_to_first_order_time_ms:
                continue
            first_order_time_ms = hk_to_first_order_time_ms[hotkey]

            if start_time_ms != first_order_time_ms:
                bt.logging.info(f"Challengeperiod start time for {hotkey} updated from: {datetime.fromtimestamp(start_time_ms/1000)} "
                                f"to: {datetime.fromtimestamp(first_order_time_ms/1000)}, {(start_time_ms-first_order_time_ms)/1000}s delta")
                self.set_miner_bucket(hotkey, MinerBucket.CHALLENGE, first_order_time_ms)
                any_changes = True

        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

        bt.logging.info("All challengeperiod start times up to date")

    def add_all_miners_to_success(self, current_time_ms, run_elimination=True):
        """Used to bypass running challenge period, but still adds miners to success for statistics."""
        assert self.is_backtesting, "This function is only for backtesting"
        eliminations = []
        if run_elimination:
            eliminations = self.elim_client.get_eliminations_from_memory()
            self.remove_eliminated(eliminations=eliminations)

        challenge_hk_to_positions, challenge_hk_to_first_order_time = self._position_client.filtered_positions_for_scoring(
            hotkeys=self._metagraph_client.get_hotkeys())

        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self._metagraph_client.get_hotkeys(),
            eliminations=eliminations,
            hk_to_first_order_time=challenge_hk_to_first_order_time,
            default_time=current_time_ms
        )

        miners_to_promote = self.get_hotkeys_by_bucket(MinerBucket.CHALLENGE) \
                          + self.get_hotkeys_by_bucket(MinerBucket.PROBATION)

        # Finally promote all testing miners to success
        self._promote_challengeperiod_in_memory(miners_to_promote, current_time_ms)

    # ==================== Internal Getter/Setter Methods ====================

    def set_miner_bucket(
        self,
        hotkey: str,
        bucket: MinerBucket,
        start_time: int,
        prev_bucket: MinerBucket = None,
        prev_time: int = None
    ) -> bool:
        """Set or update a miner's bucket information."""
        is_new = hotkey not in self.active_miners
        self.active_miners[hotkey] = (bucket, start_time, prev_bucket, prev_time)
        return is_new

    def get_miner_start_time(self, hotkey: str) -> int:
        """Get the start time of a miner's current bucket."""
        info = self.active_miners.get(hotkey)
        return info[1] if info else None

    def get_miner_previous_bucket(self, hotkey: str) -> MinerBucket:
        """Get the previous bucket of a miner."""
        info = self.active_miners.get(hotkey)
        return info[2] if info else None

    def get_miner_previous_time(self, hotkey: str) -> int:
        """Get the start time of a miner's previous bucket."""
        info = self.active_miners.get(hotkey)
        return info[3] if info else None

    def has_miner(self, hotkey: str) -> bool:
        """Fast check if a miner is in active_miners (O(1))."""
        return hotkey in self.active_miners

    def remove_miner(self, hotkey: str) -> bool:
        """Remove a miner from active_miners."""
        if hotkey in self.active_miners:
            del self.active_miners[hotkey]
            return True
        return False

    def clear_active_miners(self):
        """Clear all miners from active_miners."""
        self.active_miners.clear()

    def update_active_miners(self, miners_dict: dict) -> int:
        """
        Bulk update active_miners from a dict.

        Args:
            miners_dict: Can be either:
                - Dict mapping hotkey to tuple (bucket, start_time, prev_bucket, prev_time)
                - Dict mapping hotkey to dict with keys: bucket, start_time, prev_bucket, prev_time
                  (for RPC serialization compatibility)

        Returns:
            Number of miners updated
        """
        # Handle both tuple format and dict format (for RPC compatibility)
        normalized_dict = {}
        for hotkey, data in miners_dict.items():
            if isinstance(data, tuple):
                # Already in tuple format
                normalized_dict[hotkey] = data
            elif isinstance(data, dict):
                # Convert from RPC dict format to tuple format
                bucket = MinerBucket(data["bucket"]) if isinstance(data["bucket"], str) else data["bucket"]
                start_time = data["start_time"]
                prev_bucket = MinerBucket(data["prev_bucket"]) if (data.get("prev_bucket") and isinstance(data["prev_bucket"], str)) else data.get("prev_bucket")
                prev_time = data.get("prev_time")

                normalized_dict[hotkey] = (bucket, start_time, prev_bucket, prev_time)
            else:
                raise ValueError(f"Invalid data type for miner {hotkey}: {type(data)}")

        count = len(normalized_dict)
        self.active_miners.update(normalized_dict)
        return count

    def iter_active_miners(self):
        """Iterate over active miners."""
        for hotkey, (bucket, start_time, prev_bucket, prev_time) in self.active_miners.items():
            yield hotkey, bucket, start_time, prev_bucket, prev_time

    def get_all_miner_hotkeys(self) -> list:
        """Get list of all active miner hotkeys."""
        return list(self.active_miners.keys())

    def get_all_elimination_reasons(self) -> dict:
        """Get all elimination reasons as a dict."""
        with self.eliminations_lock:
            return dict(self.eliminations_with_reasons)

    def has_elimination_reasons(self) -> bool:
        """Check if there are any elimination reasons."""
        with self.eliminations_lock:
            return bool(self.eliminations_with_reasons)

    def pop_elimination_reason(self, hotkey: str) -> Optional[Tuple[str, float]]:
        """Atomically get and remove an elimination reason for a single hotkey."""
        with self.eliminations_lock:
            return self.eliminations_with_reasons.pop(hotkey, None)

    def clear_elimination_reasons(self):
        """Clear all elimination reasons."""
        with self.eliminations_lock:
            self.eliminations_with_reasons.clear()

    def update_elimination_reasons(self, reasons_dict: dict) -> int:
        """Accumulate elimination reasons from a dict."""
        with self.eliminations_lock:
            self.eliminations_with_reasons.update(reasons_dict)
        return len(self.eliminations_with_reasons)

    def get_miner_bucket(self, hotkey):
        """Get the bucket of a miner."""
        return self.active_miners.get(hotkey, [None])[0]

    def get_testing_miners(self):
        """Get all CHALLENGE bucket miners."""
        return copy.deepcopy(self._bucket_view(MinerBucket.CHALLENGE))

    def get_success_miners(self):
        """Get all MAINCOMP bucket miners."""
        return copy.deepcopy(self._bucket_view(MinerBucket.MAINCOMP))

    def get_probation_miners(self):
        """Get all PROBATION bucket miners."""
        return copy.deepcopy(self._bucket_view(MinerBucket.PROBATION))

    def get_plagiarism_miners(self):
        """Get all PLAGIARISM bucket miners."""
        return copy.deepcopy(self._bucket_view(MinerBucket.PLAGIARISM))

    def _bucket_view(self, bucket: MinerBucket):
        """Get all miners in a specific bucket as {hotkey: start_time} dict."""
        return {hk: ts for hk, (b, ts, _, _) in self.active_miners.items() if b == bucket}

    def to_checkpoint_dict(self):
        """Get challenge period data as a checkpoint dict for serialization."""
        json_dict = {
            hotkey: {
                "bucket": bucket.value,
                "bucket_start_time": start_time,
                "previous_bucket": previous_bucket.value if previous_bucket else None,
                "previous_bucket_start_time": previous_bucket_time
            }
            for hotkey, bucket, start_time, previous_bucket, previous_bucket_time in self.iter_active_miners()
        }
        return json_dict

    @staticmethod
    def parse_checkpoint_dict(json_dict):
        """Parse checkpoint dict from disk."""
        formatted_dict = {}

        if "testing" in json_dict.keys() and "success" in json_dict.keys():
            testing = json_dict.get("testing", {})
            success = json_dict.get("success", {})
            for hotkey, start_time in testing.items():
                formatted_dict[hotkey] = (MinerBucket.CHALLENGE, start_time, None, None)
            for hotkey, start_time in success.items():
                formatted_dict[hotkey] = (MinerBucket.MAINCOMP, start_time, None, None)
        else:
            for hotkey, info in json_dict.items():
                bucket = MinerBucket(info["bucket"]) if info.get("bucket") else None
                bucket_start_time = info.get("bucket_start_time")
                previous_bucket = MinerBucket(info["previous_bucket"]) if info.get("previous_bucket") else None
                previous_bucket_start_time = info.get("previous_bucket_start_time")

                formatted_dict[hotkey] = (bucket, bucket_start_time, previous_bucket, previous_bucket_start_time)

        return formatted_dict
