# developer: trdougherty
from collections import defaultdict
import time
import bittensor as bt
import copy

from datetime import datetime

from vali_objects.utils.asset_segmentation import AssetSegmentation
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePairCategory, ValiConfig
from shared_objects.cache_controller import CacheController
from vali_objects.scoring.scoring import Scoring
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.perf_ledger import PerfLedgerManager, PerfLedger
from vali_objects.utils.ledger_utils import LedgerUtils
from vali_objects.utils.position_manager import PositionManager
from vali_objects.position import Position
from vali_objects.utils.elimination_manager import EliminationReason
from vali_objects.utils.miner_bucket_enum import MinerBucket

class ChallengePeriodManager(CacheController):
    def __init__(
            self,
            metagraph,
            perf_ledger_manager : PerfLedgerManager=None,
            position_manager: PositionManager=None,
            ipc_manager=None,
            contract_manager=None,
            plagiarism_manager=None,
            *,
            running_unit_tests=False,
            is_backtesting=False):
        super().__init__(metagraph, running_unit_tests=running_unit_tests, is_backtesting=is_backtesting)
        self.perf_ledger_manager = perf_ledger_manager if perf_ledger_manager else \
            PerfLedgerManager(metagraph, running_unit_tests=running_unit_tests)
        self.position_manager = position_manager
        self.elimination_manager = self.position_manager.elimination_manager
        self.eliminations_with_reasons: dict[str, tuple[str, float]] = {}
        self.contract_manager = contract_manager
        self.plagiarism_manager = plagiarism_manager

        self.CHALLENGE_FILE = ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=running_unit_tests)

        self.active_miners = {}
        initial_active_miners = {}
        if not self.is_backtesting:
            disk_data = ValiUtils.get_vali_json_file_dict(self.CHALLENGE_FILE)
            initial_active_miners = self.parse_checkpoint_dict(disk_data)

        if ipc_manager:
            self.active_miners = ipc_manager.dict(initial_active_miners)
        else:
            self.active_miners = initial_active_miners

        if not self.is_backtesting and len(self.active_miners) == 0:
            self._write_challengeperiod_from_memory_to_disk()

        self.refreshed_challengeperiod_start_time = False

    #Used to bypass running challenge period, but still adds miners to success for statistics
    def add_all_miners_to_success(self, current_time_ms, run_elimination=True):
        assert self.is_backtesting, "This function is only for backtesting"
        eliminations = []
        if run_elimination:
            # The refresh should just read the current eliminations
            eliminations = self.elimination_manager.get_eliminations_from_memory()

            # Collect challenge period and update with new eliminations criteria
            self.remove_eliminated(eliminations=eliminations)

        challenge_hk_to_positions, challenge_hk_to_first_order_time = self.position_manager.filtered_positions_for_scoring(
            hotkeys=self.metagraph.hotkeys)

        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.metagraph.hotkeys,
            eliminations=eliminations,
            hk_to_first_order_time=challenge_hk_to_first_order_time,
            default_time=current_time_ms
        )

        miners_to_promote = self.get_hotkeys_by_bucket(MinerBucket.CHALLENGE) \
                          + self.get_hotkeys_by_bucket(MinerBucket.PROBATION)

        #Finally promote all testing miners to success
        self._promote_challengeperiod_in_memory(miners_to_promote, current_time_ms)

    def _add_challengeperiod_testing_in_memory_and_disk(
            self,
            new_hotkeys: list[str],
            eliminations: list[dict],
            hk_to_first_order_time: dict[str, int],
            default_time: int
    ):
        if not eliminations:
            eliminations = self.elimination_manager.get_eliminations_from_memory()

        elimination_hotkeys = set(x['hotkey'] for x in eliminations)
        maincomp_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)
        probation_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.PROBATION)
        plagiarism_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.PLAGIARISM)

        any_changes = False
        for hotkey in new_hotkeys:
            if hotkey in elimination_hotkeys:
                continue

            if hotkey in maincomp_hotkeys or hotkey in probation_hotkeys or hotkey in plagiarism_hotkeys:
                continue

            first_order_time = hk_to_first_order_time.get(hotkey)
            if first_order_time is None:
                if hotkey not in self.active_miners:
                    self.active_miners[hotkey] = (MinerBucket.CHALLENGE, default_time, None, None)
                    bt.logging.info(f"Adding {hotkey} to challenge period with start time {default_time}")
                    any_changes = True
                continue

            # Has a first order time but not yet stored in memory
            # Has a first order time but start time is set as default
            if hotkey not in self.active_miners or self.active_miners[hotkey][1] != first_order_time:
                self.active_miners[hotkey] = (MinerBucket.CHALLENGE, first_order_time, None, None)
                bt.logging.info(f"Adding {hotkey} to challenge period with first order time {first_order_time}")
                any_changes = True

        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

    def _refresh_challengeperiod_start_time(self, hk_to_first_order_time_ms: dict[str, int]):
        """
        retroactively update the challengeperiod_testing start time based on time of first order.
        used when a miner is un-eliminated, and positions are preserved.
        """
        bt.logging.info("Refreshing challengeperiod start times")

        any_changes = False
        for hotkey in self.get_testing_miners().keys():
            start_time_ms = self.active_miners[hotkey][1]
            if hotkey not in hk_to_first_order_time_ms:
                #bt.logging.warning(f"Hotkey {hotkey} in challenge period has no first order time. Skipping for now.")
                continue
            first_order_time_ms = hk_to_first_order_time_ms[hotkey]

            if start_time_ms != first_order_time_ms:
                bt.logging.info(f"Challengeperiod start time for {hotkey} updated from: {datetime.utcfromtimestamp(start_time_ms/1000)} "
                                f"to: {datetime.utcfromtimestamp(first_order_time_ms/1000)}, {(start_time_ms-first_order_time_ms)/1000}s delta")
                self.active_miners[hotkey] = (MinerBucket.CHALLENGE, first_order_time_ms, None, None)
                any_changes = True

        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

        bt.logging.info("All challengeperiod start times up to date")

    def refresh(self, current_time: int):
        if not self.refresh_allowed(ValiConfig.CHALLENGE_PERIOD_REFRESH_TIME_MS):
            time.sleep(1)
            return
        bt.logging.info(f"Refreshing challenge period. invalidation data {self.perf_ledger_manager.perf_ledger_hks_to_invalidate}")
        # The refresh should just read the current eliminations
        eliminations = self.elimination_manager.get_eliminations_from_memory()

        self.update_plagiarism_miners(current_time, self.get_plagiarism_miners())

        # Collect challenge period and update with new eliminations criteria
        self.remove_eliminated(eliminations=eliminations)

        hk_to_positions, hk_to_first_order_time = self.position_manager.filtered_positions_for_scoring(hotkeys=self.metagraph.hotkeys)

        # challenge period adds to testing if not in eliminated, already in the challenge period, or in the new eliminations list from disk
        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self.metagraph.hotkeys,
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

        ledger = self.perf_ledger_manager.filtered_ledger_for_scoring(hotkeys=all_miners)
        ledger = {hotkey: ledger.get(hotkey, None) for hotkey in all_miners}

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

        self.eliminations_with_reasons = challengeperiod_eliminations

        any_changes = bool(challengeperiod_success) or bool(challengeperiod_eliminations) or bool(challengeperiod_demoted)

        # Moves challenge period testing to challenge period success in memory
        self._promote_challengeperiod_in_memory(challengeperiod_success, current_time)
        self._demote_challengeperiod_in_memory(challengeperiod_demoted, current_time)
        self._eliminate_challengeperiod_in_memory(eliminations_with_reasons=challengeperiod_eliminations)

        # Now remove any miners who are no longer in the metagraph
        any_changes |= self._prune_deregistered_metagraph()

        # Now sync challenge period with the disk
        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

        self.set_last_update_time()

        bt.logging.info(
            "Challenge Period snapshot after refresh "
            f"(MAINCOMP, {len(self.get_success_miners())}) "
            f"(PROBATION, {len(self.get_probation_miners())}) "
            f"(CHALLENGE, {len(self.get_testing_miners())}) "
            f"(PLAGIARISM, {len(self.get_plagiarism_miners())})"
        )

    def _prune_deregistered_metagraph(self, hotkeys=None) -> bool:
        """
        Prune the challenge period of all miners who are no longer in the metagraph
        """
        if not hotkeys:
            hotkeys = self.metagraph.hotkeys

        any_changes = False
        for hotkey in list(self.active_miners.keys()):
            if hotkey not in hotkeys:
                del self.active_miners[hotkey]
                any_changes = True

        return any_changes

    @staticmethod
    def is_recently_re_registered(ledger, hotkey, hk_to_first_order_time):
        """
        A miner can re-register and their perf ledger may still be old.
        This function checks for that condition and blocks challenge period failure so that
        the perf ledger can rebuild.
        """
        if not hk_to_first_order_time:
            return False
        if ledger:
            time_of_ledger_start = ledger.start_time_ms
        else:
            # No ledger? No edge case.
            return False

        first_order_time = hk_to_first_order_time.get(hotkey, None)
        if first_order_time is None:
            # No positions? Perf ledger must be stale.
            msg = f'No positions for hotkey {hotkey} - ledger start time: {time_of_ledger_start}'
            print(msg)
            return True

        # A perf ledger can never begin before the first order. Edge case confirmed.
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
        success_scores_dict: dict[str, dict] | None = None,
        inspection_scores_dict: dict[str, dict] | None = None,
        hk_to_first_order_time: dict[str, int] | None = None,
    ) -> tuple[list[str], list[str], dict[str, tuple[str, float]]]:
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period. Does not modify the challenge period in memory.

        Args:
            success_scores_dict (dict[str, dict]) - a dictionary with a similar structure to config with keys being
            function names of metrics and values having "scores" (scores of miners that passed challenge)
            and "weight" which is the weight of the metric. Only provided if running tests

            inspection_scores_dict (dict[str, dict]) - identical to success_scores_dict in structure, but only has data
            for one inspection hotkey. Only provided if running tests

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
        miners_recently_reregistered = set()
        miners_not_enough_positions = []

        # Used for checking base cases
        #TODO revisit this
        portfolio_only_ledgers = {hotkey: asset_ledgers.get("portfolio") for hotkey, asset_ledgers in ledger.items() if asset_ledgers is not None}
        valid_candidate_hotkeys = []
        for hotkey, bucket_start_time in inspection_hotkeys.items():

            if ChallengePeriodManager.is_recently_re_registered(portfolio_only_ledgers.get(hotkey), hotkey, hk_to_first_order_time):
                miners_recently_reregistered.add(hotkey)
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

            # Get hotkey to positions dict that only includes the inspection miner
            has_minimum_positions, inspection_positions = ChallengePeriodManager.screen_minimum_positions(positions, hotkey)
            if not has_minimum_positions:
                miners_not_enough_positions.append(hotkey)
                continue

            # Get hotkey to ledger dict that only includes the inspection miner
            has_minimum_ledger, inspection_ledger = ChallengePeriodManager.screen_minimum_ledger(portfolio_only_ledgers, hotkey)
            if not has_minimum_ledger:
                continue

            # This step we want to check their drawdown. If they fail, we can move on.
            ledger_element = inspection_ledger[hotkey]
            exceeds_max_drawdown, recorded_drawdown_percentage = LedgerUtils.is_beyond_max_drawdown(ledger_element)
            if exceeds_max_drawdown:
                bt.logging.info(f'Hotkey {hotkey} has failed the {miner_bucket.value} period due to drawdown {recorded_drawdown_percentage}. cp_failed')
                miners_to_eliminate[hotkey] = (EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value, recorded_drawdown_percentage)
                continue

            if not self.screen_minimum_interaction(ledger_element):
                continue

            valid_candidate_hotkeys.append(hotkey)

        # Calculate dynamic minimum participation days for asset classes
        maincomp_ledger = {hotkey: ledger_data for hotkey, ledger_data in ledger.items() if hotkey in [*success_hotkeys, *probation_hotkeys]}   # ledger of all miners in maincomp, including probation
        asset_classes = list(AssetSegmentation.distill_asset_classes(ValiConfig.ASSET_CLASS_BREAKDOWN))
        asset_class_min_days = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            maincomp_ledger, asset_classes
        )
        bt.logging.info(f"challengeperiod_manager asset class minimum days: {asset_class_min_days}")

        all_miner_account_sizes = self.contract_manager.get_all_miner_account_sizes(timestamp_ms=current_time)

        # If success_scoring_dict is already calculated, no need to calculate scores. Useful for testing
        if not success_scores_dict:
            success_positions = {hotkey: miner_positions for hotkey, miner_positions in positions.items() if hotkey in success_hotkeys}
            success_ledger = {hotkey: ledger_data for hotkey, ledger_data in ledger.items() if hotkey in success_hotkeys}
            # Get the penalized scores of all successful miners
            success_scores_dict = Scoring.score_miners(
                    ledger_dict=success_ledger,
                    positions=success_positions,
                    asset_class_min_days=asset_class_min_days,
                    evaluation_time_ms=current_time,
                    weighting=True,
                    all_miner_account_sizes=all_miner_account_sizes)

        if not inspection_scores_dict:
            candidates_positions = {hotkey: positions[hotkey] for hotkey in valid_candidate_hotkeys}
            candidates_ledgers = {hotkey: ledger[hotkey] for hotkey in valid_candidate_hotkeys}

            inspection_scores_dict = Scoring.score_miners(
                    ledger_dict=candidates_ledgers,
                    positions=candidates_positions,
                    asset_class_min_days=asset_class_min_days,
                    evaluation_time_ms=current_time,
                    weighting=True,
                    all_miner_account_sizes=all_miner_account_sizes)

        hotkeys_to_promote, hotkeys_to_demote = ChallengePeriodManager.evaluate_promotions(success_hotkeys,
                                                                                           success_scores_dict,
                                                                                           valid_candidate_hotkeys,
                                                                                           inspection_scores_dict)

        bt.logging.info(f"Challenge Period: evaluating {len(valid_candidate_hotkeys)}/{len(inspection_hotkeys)} miners eligible for promotion")
        bt.logging.info(f"Challenge Period: evaluating {len(success_hotkeys)} miners eligible for demotion")
        bt.logging.info(f"Hotkeys to promote: {hotkeys_to_promote}")
        bt.logging.info(f"Hotkeys to demote: {hotkeys_to_demote}")
        bt.logging.info(f"Hotkeys to eliminate: {list(miners_to_eliminate.keys())}")
        bt.logging.info(f"Miners with no positions (skipped): {len(miners_not_enough_positions)}")
        bt.logging.info(f"Miners recently re-registered (skipped): {list(miners_recently_reregistered)}")

        return hotkeys_to_promote, hotkeys_to_demote, miners_to_eliminate

    @staticmethod
    def evaluate_promotions(
            success_hotkeys,
            success_scores_dict,
            candidate_hotkeys,
            inspection_scores_dict
            ) -> tuple[list[str], list[str]]:
        # combine maincomp and challenge/probation miners into one scoring dict
        combined_scores_dict = copy.deepcopy(success_scores_dict)
        for asset_class, candidate_scores_dict in inspection_scores_dict.items():
            for metric_name, candidate_metric in candidate_scores_dict["metrics"].items():
                combined_scores_dict[asset_class]['metrics'][metric_name]["scores"] += candidate_metric["scores"]
            combined_scores_dict[asset_class]["penalties"].update(candidate_scores_dict["penalties"])

        # score them based on asset class
        asset_combined_scores = Scoring.combine_scores(combined_scores_dict)
        asset_softmaxed_scores = Scoring.softmax_by_asset(asset_combined_scores)

        # combine asset
        weighted_scores = defaultdict(lambda: defaultdict(float))
        for asset_class, miner_scores in asset_softmaxed_scores.items():
            weight = ValiConfig.ASSET_CLASS_BREAKDOWN[asset_class]["emission"]

            for hotkey, score in miner_scores.items():
                weighted_scores[asset_class][hotkey] += weight * score

        maincomp_hotkeys = set()
        promotion_threshold_rank = ValiConfig.PROMOTION_THRESHOLD_RANK
        for asset_scores in weighted_scores.values():
            threshold_score = 0
            if len(asset_scores) >= promotion_threshold_rank:
                sorted_scores = sorted(asset_scores.values(), reverse=True)
                threshold_score = sorted_scores[promotion_threshold_rank-1]

            for hotkey, score in asset_scores.items():
                if score >= threshold_score and score > 0:
                    maincomp_hotkeys.add(hotkey)

            # logging
            for hotkey in success_hotkeys:
                if hotkey not in asset_scores:
                    bt.logging.warning(f"Could not find MAINCOMP hotkey {hotkey} when scoring, miner will not be evaluated")
            for hotkey in candidate_hotkeys:
                if hotkey not in asset_scores:
                    bt.logging.warning(f"Could not find CHALLENGE/PROBATION hotkey {hotkey} when scoring, miner will not be evaluated")

        promote_hotkeys = maincomp_hotkeys - set(success_hotkeys)
        demote_hotkeys = set(success_hotkeys) - maincomp_hotkeys

        return list(promote_hotkeys), list(demote_hotkeys)

    @staticmethod
    def screen_minimum_interaction(ledger_element) -> bool:
        """
        Returns False if the miner doesn't have the minimum number of trading days.
        """
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
    ) -> tuple[bool, dict[str, PerfLedger]]:
        """
        Ensures there is enough ledger data globally and for the specific miner to evaluate challenge period.
        """
        if ledger is None or len(ledger) == 0:
            bt.logging.info(f"No ledgers for any miner to evaluate for challenge period. ledger: {ledger}")
            return False, {}

        single_ledger = ledger.get(inspection_hotkey, None)
        if single_ledger is None:
            return False, {}

        has_minimum_ledger = len(single_ledger.cps) > 0

        if not has_minimum_ledger:
            bt.logging.info(f"Hotkey: {inspection_hotkey} doesn't have the minimum ledger for challenge period. ledger: {single_ledger}")

        inspection_ledger = {inspection_hotkey: single_ledger} if has_minimum_ledger else {}

        return has_minimum_ledger, inspection_ledger

    @staticmethod
    def screen_minimum_positions(
            positions: dict[str, list[Position]],
            inspection_hotkey: str
    ) -> tuple[bool, dict[str, list[Position]]]:
        """
        Ensures there are enough positions globally and for the specific miner to evaluate challenge period.
        """

        if positions is None or len(positions) == 0:
            bt.logging.info(f"No positions for any miner to evaluate for challenge period. positions: {positions}")
            return False, {}

        positions_list = positions.get(inspection_hotkey, None)
        has_minimum_positions = positions_list is not None and len(positions_list) > 0

        inspection_positions = {inspection_hotkey: positions_list} if has_minimum_positions else {}

        return has_minimum_positions, inspection_positions

    def sync_challenge_period_data(self, active_miners_sync):
        if not active_miners_sync:
            bt.logging.error(f'challenge_period_data {active_miners_sync} appears invalid')

        synced_miners = self.parse_checkpoint_dict(active_miners_sync)

        self.active_miners.clear()
        self.active_miners.update(synced_miners)
        self._write_challengeperiod_from_memory_to_disk()

    def get_hotkeys_by_bucket(self, bucket: MinerBucket) -> list[str]:
        return [hotkey for hotkey, (b, _, _, _) in self.active_miners.items() if b == bucket]

    def _remove_eliminated_from_memory(self, eliminations: list[dict] = None) -> bool:
        if eliminations is None:
            eliminations_hotkeys = self.elimination_manager.get_eliminated_hotkeys()
        else:
            eliminations_hotkeys = set([x['hotkey'] for x in eliminations])

        any_changes = False
        for hotkey in eliminations_hotkeys:
            if hotkey in self.active_miners:
                del self.active_miners[hotkey]
                any_changes = True

        return any_changes

    def remove_eliminated(self, eliminations=None):
        # Pass eliminations directly to _remove_eliminated_from_memory
        # Don't convert None to [] - let the inner function handle None properly
        any_changes = self._remove_eliminated_from_memory(eliminations=eliminations)
        if any_changes:
            self._write_challengeperiod_from_memory_to_disk()

    def _clear_challengeperiod_in_memory_and_disk(self):
        self.active_miners.clear()
        self._write_challengeperiod_from_memory_to_disk()

    def update_plagiarism_miners(self, current_time, plagiarism_miners):

        new_plagiarism_miners, whitelisted_miners = self.plagiarism_manager.update_plagiarism_miners(current_time, plagiarism_miners)
        self._demote_plagiarism_in_memory(new_plagiarism_miners, current_time)
        self._promote_plagiarism_to_previous_bucket_in_memory(whitelisted_miners, current_time)

    def prepare_plagiarism_elimination_miners(self, current_time):

        miners_to_eliminate = self.plagiarism_manager.plagiarism_miners_to_eliminate(current_time)
        elim_miners_to_return = {}
        for hotkey in miners_to_eliminate:
            if hotkey in self.active_miners:
                bt.logging.info(
                    f'Hotkey {hotkey} is overdue in {MinerBucket.PLAGIARISM} at time {current_time}')
                elim_miners_to_return[hotkey] = (EliminationReason.PLAGIARISM.value, -1)
                self.plagiarism_manager.send_plagiarism_elimination_notification(hotkey)

        return elim_miners_to_return

    def _promote_challengeperiod_in_memory(self, hotkeys: list[str], current_time: int):
        if len(hotkeys) > 0:
            bt.logging.info(f"Promoting {len(hotkeys)} miners to main competition.")

        for hotkey in hotkeys:
            bucket_value = self.get_miner_bucket(hotkey)
            if bucket_value is None:
                bt.logging.error(f"Hotkey {hotkey} is not an active miner. Skipping promotion")
                continue
            bt.logging.info(f"Promoting {hotkey} from {self.get_miner_bucket(hotkey).value} to MAINCOMP")
            self.active_miners[hotkey] = (MinerBucket.MAINCOMP, current_time, None, None)

    def _promote_plagiarism_to_previous_bucket_in_memory(self, hotkeys: list[str], current_time):
        if len(hotkeys) > 0:
            bt.logging.info(f"Promoting {len(hotkeys)} plagiarism miners to probation.")

        for hotkey in hotkeys:
            try:
                bucket_value = self.get_miner_bucket(hotkey)
                if bucket_value is None or bucket_value != MinerBucket.PLAGIARISM:
                    bt.logging.error(f"Hotkey {hotkey} is not an active miner. Skipping promotion")
                    continue
                # Extra tuple values are set when demoting due to plagiarism
                previous_bucket = self.active_miners.get(hotkey)[2]
                previous_time = self.active_miners.get(hotkey)[3]
                #TODO Possibly calculate how long miner has been in plagiarism, give them this time back

                # Miner is a plagiarist
                bt.logging.info(f"Promoting {hotkey} from {bucket_value.value} to {previous_bucket.value} with time {previous_time}")
                self.active_miners[hotkey] = (previous_bucket, previous_time, None, None)

                # Send Slack notification
                self.plagiarism_manager.send_plagiarism_promotion_notification(hotkey)
            except Exception as e:
                bt.logging.error(f"Failed to promote {hotkey} from plagiarism at time {current_time}: {e}")

    def _eliminate_challengeperiod_in_memory(self, eliminations_with_reasons: dict[str, tuple[str, float]]):
        hotkeys = eliminations_with_reasons.keys()
        if hotkeys:
            bt.logging.info(f"Removing {len(hotkeys)} hotkeys from challenge period.")

        for hotkey in hotkeys:
            if hotkey in self.active_miners:
                bt.logging.info(f"Eliminating {hotkey}")
                del self.active_miners[hotkey]
            else:
                bt.logging.error(f"Hotkey {hotkey} was not in challengeperiod_testing but demotion to failure was attempted.")

    def _demote_challengeperiod_in_memory(self, hotkeys: list[str], current_time):
        if hotkeys:
            bt.logging.info(f"Demoting {len(hotkeys)} miners to probation")

        for hotkey in hotkeys:
            bucket_value = self.get_miner_bucket(hotkey)
            if bucket_value is None:
                bt.logging.error(f"Hotkey {hotkey} is not an active miner. Skipping demotion")
                continue
            bt.logging.info(f"Demoting {hotkey} to PROBATION")
            self.active_miners[hotkey] = (MinerBucket.PROBATION, current_time, None, None)

    def _demote_plagiarism_in_memory(self, hotkeys: list[str], current_time):
        for hotkey in hotkeys:
            try:
                prev_bucket_value = self.get_miner_bucket(hotkey)
                # Check if miner is an active miner, if not, no need to demote
                if prev_bucket_value is None:
                    continue
                prev_bucket_time = self.active_miners.get(hotkey)[1]
                bt.logging.info(f"Demoting {hotkey} to PLAGIARISM from {prev_bucket_value}")
                # Maintain previous state to make reverting easier
                self.active_miners[hotkey] = (MinerBucket.PLAGIARISM, current_time, prev_bucket_value, prev_bucket_time)

                # Send Slack notification
                self.plagiarism_manager.send_plagiarism_demotion_notification(hotkey)
            except Exception as e:
                bt.logging.error(f"Failed to demote {hotkey} for plagiarism at time {current_time}: {e}")


    def _write_challengeperiod_from_memory_to_disk(self):
        if self.is_backtesting:
            return
        challengeperiod_data = self.to_checkpoint_dict()
        ValiBkpUtils.write_file(self.CHALLENGE_FILE, challengeperiod_data)

    def get_miner_bucket(self, hotkey): return self.active_miners.get(hotkey, [None])[0]
    def get_testing_miners(self):   return copy.deepcopy(self._bucket_view(MinerBucket.CHALLENGE))
    def get_success_miners(self):   return copy.deepcopy(self._bucket_view(MinerBucket.MAINCOMP))
    def get_probation_miners(self): return copy.deepcopy(self._bucket_view(MinerBucket.PROBATION))
    def get_plagiarism_miners(self): return copy.deepcopy(self._bucket_view(MinerBucket.PLAGIARISM))

    def _bucket_view(self, bucket: MinerBucket):
        return {hk: ts for hk, (b, ts, _, _) in self.active_miners.items() if b == bucket}

    def to_checkpoint_dict(self):
        snapshot = list(self.active_miners.items())
        json_dict = {
            hotkey: {
                "bucket": bucket.value,
                "bucket_start_time": start_time,
                "previous_bucket": previous_bucket.value if previous_bucket else None,
                "previous_bucket_start_time": previous_bucket_time
            }
            for hotkey, (bucket, start_time, previous_bucket, previous_bucket_time) in snapshot
        }
        return json_dict

    @staticmethod
    def parse_checkpoint_dict(json_dict):
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

