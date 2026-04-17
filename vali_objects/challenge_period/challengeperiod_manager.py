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
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePairCategory, ValiConfig, RPCConnectionMode
from vali_objects.utils.asset_selection.asset_selection_manager import ASSET_CLASS_SELECTION_TIME_MS
from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
from shared_objects.cache_controller import CacheController
from vali_objects.scoring.scoring import Scoring
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.elimination.elimination_manager import EliminationReason
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.plagiarism.plagiarism_client import PlagiarismClient
from vali_objects.miner_account.miner_account_client import MinerAccountClient
from shared_objects.rpc.common_data_client import CommonDataClient
from entity_management.entity_utils import is_synthetic_hotkey


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

        self._miner_account_client = MinerAccountClient(
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

        self._debt_ledger_client = DebtLedgerClient(
            connect_immediately=False,
            connection_mode=connection_mode
        )

        self.eliminations_with_reasons: Dict[str, Tuple[str, float, int]] = {}  # (reason, mdd, detection_time_ms)
        self.active_miners: Dict[str, List[BucketEntry]] = {}

        # Cached scores for MinerStatisticsManager
        self._cached_asset_softmaxed_scores: Dict[str, Dict[str, float]] = {}
        self._cached_asset_competitiveness: Dict[str, float] = {}

        # Cached drawdown stats per synthetic hotkey — updated by _evaluate_synthetic_challenge,
        # read by get_drawdown_stats so dashboard values match the evaluation loop exactly.
        self._drawdown_stats_cache: Dict[str, dict] = {}

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
            hotkeys=self._position_client.get_all_hotkeys()
        )

        # Add to testing if not in eliminated, already in the challenge period, or in the new eliminations list
        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self._position_client.get_all_hotkeys(),
            eliminations=eliminations,
            hk_to_first_order_time=hk_to_first_order_time,
            default_time=current_time
        )

        challengeperiod_success_hotkeys = (
            self.get_hotkeys_by_bucket(MinerBucket.MAINCOMP) +
            self.get_hotkeys_by_bucket(MinerBucket.SUBACCOUNT_ALPHA)
        )
        challengeperiod_testing_hotkeys = (
            self.get_hotkeys_by_bucket(MinerBucket.CHALLENGE) +
            self.get_hotkeys_by_bucket(MinerBucket.SUBACCOUNT_CHALLENGE)
        )
        challengeperiod_probation_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.PROBATION)
        challengeperiod_funded_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.SUBACCOUNT_FUNDED)
        all_miners = challengeperiod_success_hotkeys + challengeperiod_testing_hotkeys + challengeperiod_probation_hotkeys + challengeperiod_funded_hotkeys

        if not self.refreshed_challengeperiod_start_time:
            self.refreshed_challengeperiod_start_time = True
            self._refresh_challengeperiod_start_time(hk_to_first_order_time)
            self._sync_all_miner_buckets()

        ledger = self._perf_ledger_client.filtered_ledger_for_scoring(hotkeys=all_miners)

        inspection_miners = self.get_testing_miners() | self.get_probation_miners() | self.get_funded_miners()
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
        """
        Prune the challenge period of miners who are no longer valid.

        Uses position_client.get_all_hotkeys() to determine valid hotkeys,
        which includes regular miners and synthetic hotkeys with positions.
        Skip entity miners.
        Elimination system handles removing truly invalid miners.
        """
        if not hotkeys:
            # Get all hotkeys with positions (includes synthetic hotkeys)
            hotkeys = set(self._position_client.get_all_hotkeys())
        else:
            hotkeys = set(hotkeys)

        any_changes = False
        for hotkey in self.get_all_miner_hotkeys():
            if hotkey not in hotkeys:
                bucket = self.get_miner_bucket(hotkey)
                # Entity miners do not have positions. skip pruning
                if bucket in [MinerBucket.ENTITY, MinerBucket.SUBACCOUNT_FUNDED]:
                    continue
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

    # ==================== Unified Helper Methods ====================

    def _get_time_limit_for_bucket(self, bucket: MinerBucket) -> int:
        """Get time limit based on bucket (same for regular and synthetic)."""
        if bucket == MinerBucket.CHALLENGE:
            return ValiConfig.CHALLENGE_PERIOD_MAXIMUM_MS  # 61-90 days
        elif bucket == MinerBucket.PROBATION:
            return ValiConfig.PROBATION_MAXIMUM_MS  # 60 days
        # MAINCOMP, SUBACCOUNT_CHALLENGE, SUBACCOUNT_FUNDED, SUBACCOUNT_ALPHA, ENTITY have no max time limit
        return 0

    def _check_time_limit(
        self,
        hotkey: str,
        bucket_start_time: int,
        current_time: int,
        time_limit_ms: int
    ) -> tuple[bool, tuple[str, float, int] | None]:
        """Unified time limit check."""
        if time_limit_ms == 0:
            return False, None

        if current_time > (bucket_start_time + time_limit_ms):
            bucket = self.get_miner_bucket(hotkey)
            is_synthetic = is_synthetic_hotkey(hotkey)
            context = "SYNTHETIC" if is_synthetic else "REGULAR"
            days = time_limit_ms / (24 * 60 * 60 * 1000)

            bt.logging.info(
                f"[{context}_CP] {hotkey} failed {bucket.value} period - "
                f"time expired ({days:.0f} days)"
            )
            return True, (EliminationReason.FAILED_CHALLENGE_PERIOD_TIME.value, -1, current_time)

        return False, None

    def _check_minimum_ledger(
        self,
        portfolio_only_ledgers: dict[str, PerfLedger],
        hotkey: str
    ) -> tuple[bool, PerfLedger | None]:
        """Unified minimum ledger check."""
        return ChallengePeriodManager.screen_minimum_ledger(
            portfolio_only_ledgers, hotkey
        )

    def _check_drawdown_limit(
        self,
        hotkey: str,
        ledger: PerfLedger,
        drawdown_threshold_percentage: float,
        current_time: int = None
    ) -> tuple[bool, tuple[str, float, int] | None]:
        """
        Unified drawdown check with configurable threshold.

        Args:
            hotkey: Miner hotkey
            ledger: Performance ledger
            drawdown_threshold_percentage: Threshold in 0-100 scale (e.g., 5.0 for 5%)
            current_time: Detection timestamp in ms (used as elimination_initiated_time_ms)

        Returns:
            (should_eliminate, elimination_reason_tuple)
        """
        _, recorded_drawdown_percentage = LedgerUtils.is_beyond_max_drawdown(ledger, drawdown_threshold_percentage)

        # recorded_drawdown_percentage is in 0-100 scale (e.g., 1.0 for 1% drawdown)
        # Compare against threshold in same scale
        if recorded_drawdown_percentage >= drawdown_threshold_percentage:
            is_synthetic = is_synthetic_hotkey(hotkey)
            context = "SYNTHETIC" if is_synthetic else "REGULAR"

            bt.logging.info(
                f"[{context}_CP] {hotkey} failed challenge period - "
                f"drawdown {recorded_drawdown_percentage}% >= {drawdown_threshold_percentage}%"
            )
            t_ms = current_time if current_time is not None else TimeUtil.now_in_millis()
            return True, (
                EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value,
                recorded_drawdown_percentage,
                t_ms
            )

        return False, None

    def _check_minimum_positions(
        self,
        positions: dict[str, list[Position]],
        hotkey: str
    ) -> tuple[bool, dict[str, list[Position]]]:
        """Unified minimum positions check."""
        return ChallengePeriodManager.screen_minimum_positions(positions, hotkey)

    def _check_returns_threshold(self, hotkey: str, threshold: float) -> bool:
        """Check if returns meet threshold (for synthetic instantaneous pass)."""
        try:
            returns = self._perf_ledger_client.get_returns(hotkey)

            if returns is None:
                bt.logging.debug(f"[SYNTHETIC_CP] {hotkey} has no returns data yet")
                return False

            if returns >= threshold:
                bt.logging.info(
                    f"[SYNTHETIC_CP] {hotkey} PASSED instantaneous check! "
                    f"Returns: {returns:.2%} >= {threshold:.2%}"
                )
                return True
            else:
                bt.logging.debug(
                    f"[SYNTHETIC_CP] {hotkey} still in challenge - "
                    f"returns {returns:.2%} < {threshold:.2%}"
                )
                return False

        except Exception as e:
            bt.logging.warning(f"[SYNTHETIC_CP] Error checking returns for {hotkey}: {e}")
            return False

    def get_drawdown_stats(self, synthetic_hotkey: str) -> Optional[dict]:
        """
        Return drawdown statistics for a synthetic hotkey for dashboard display.

        Values are populated by _evaluate_synthetic_challenge so the dashboard
        always reflects the same state as the evaluation loop — no live recomputation.

        Returns None if the hotkey has not been evaluated yet.
        """
        return self._drawdown_stats_cache.get(synthetic_hotkey)

    def _reset_drawdown_stats_cache(self, hotkey: str) -> None:
        """Reset a hotkey's drawdown stats cache to neutral default values.

        Sets equity fields to 1.0 (starting equity), drawdown percentages to 0.0,
        and thresholds to the challenge-period config defaults.
        """
        intraday_threshold, eod_threshold = self._get_drawdown_thresholds(hotkey)

        self._drawdown_stats_cache[hotkey] = {
            'current_equity': 1.0,
            'daily_open_equity': 1.0,
            'eod_hwm': 1.0,
            'last_eod_equity': 1.0,
            'intraday_drawdown_pct': 0.0,
            'eod_drawdown_pct': 0.0,
            'intraday_drawdown_threshold': intraday_threshold,
            'eod_drawdown_threshold': eod_threshold,
            # TODO: remove legacy fields below
            'subaccount_challenge_intraday_drawdown_threshold': intraday_threshold,
            'subaccount_challenge_eod_drawdown_threshold': eod_threshold,
        }

    def _compute_portfolio_return(self, hotkey: str, account: Optional[dict] = None) -> tuple[float, float] | None:
        """Compute current portfolio return as (balance + unrealized_pnl) / account_size.

        Returns None if account data is unavailable.
        """
        if account is None:
            return None
        account_size = account.get('account_size', 0)
        if account_size <= 0:
            return None
        balance = account.get('balance', 0)
        unrealized_pnl = self._position_client.get_unrealized_pnl(hotkey)
        equity = balance + unrealized_pnl

        equity_ret = equity / account_size
        balance_ret = balance / account_size

        return equity_ret, balance_ret

    # ==================== Evaluation Methods ====================

    # Miners registered in SUBACCOUNT_CHALLENGE before this timestamp use V0 funded thresholds
    _SUBACCOUNT_FUNDED_V0_CUTOFF_MS = 1773532799000  # Mar 14, 2026 23:59:59 UTC

    def _parse_eod_checkpoints(
        self,
        ledger: PerfLedger,
        now_ms: int
    ) -> tuple[list, float, float, float]:
        """
        Parse midnight checkpoints from a ledger.
        Returns (midnight_cps, last_eod, daily_open_equity, eod_hwm).
        """
        midnight_cps = [cp for cp in ledger.cps if cp.last_update_ms % 86400000 == 0 and cp.equity_ret > 0]
        last_eod = midnight_cps[-1].equity_ret if midnight_cps else 1.0
        today_midnight_ms = (now_ms // 86400000) * 86400000
        today_open_cp = next((cp for cp in midnight_cps if cp.last_update_ms == today_midnight_ms), None)
        daily_open_equity = today_open_cp.equity_ret if today_open_cp else last_eod
        eod_hwm = max(max(cp.equity_ret for cp in midnight_cps), 1.0) if midnight_cps else 1.0
        return midnight_cps, last_eod, daily_open_equity, eod_hwm

    def _get_drawdown_thresholds(self, hotkey: str) -> tuple[float, float]:
        """
        Returns (intraday_threshold, eod_threshold) for a subaccount miner.
        - SUBACCOUNT_CHALLENGE: returns challenge thresholds.
        - SUBACCOUNT_FUNDED: returns V0 thresholds if the miner's SUBACCOUNT_CHALLENGE bucket
          started before Mar 14, 2026, otherwise V1 thresholds.
        """
        if self.get_miner_bucket(hotkey) == MinerBucket.SUBACCOUNT_CHALLENGE:
            return (
                ValiConfig.SUBACCOUNT_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD,
                ValiConfig.SUBACCOUNT_CHALLENGE_EOD_DRAWDOWN_THRESHOLD,
            )
        history = self.active_miners.get(hotkey, [])
        challenge_entry = next((e for e in history if e.bucket == MinerBucket.SUBACCOUNT_CHALLENGE), None)
        if challenge_entry and challenge_entry.start_time_ms < self._SUBACCOUNT_FUNDED_V0_CUTOFF_MS:
            return (
                ValiConfig.SUBACCOUNT_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD_V0,
                ValiConfig.SUBACCOUNT_FUNDED_EOD_DRAWDOWN_THRESHOLD_V0,
            )
        return (
            ValiConfig.SUBACCOUNT_FUNDED_INTRADAY_DRAWDOWN_THRESHOLD,
            ValiConfig.SUBACCOUNT_FUNDED_EOD_DRAWDOWN_THRESHOLD,
        )

    def _evaluate_synthetic_challenge(
        self,
        inspection_hotkeys: dict[str, int],
        portfolio_only_ledgers: dict[str, PerfLedger],
        current_time: int = None
    ) -> tuple[list[str], dict[str, tuple[str, float, int]]]:
        """
        Evaluate synthetic hotkeys in CHALLENGE bucket with instantaneous pass criteria.

        Elimination criteria:
        **Rule 1: Daily Loss Limit (5% for CHALLENGE, 8% for FUNDED)** - Account equity cannot drop more than 5% from the day's opening equity at any point during the day.

        **Rule 2: EOD Trailing Loss Limit (5% for CHALLENGE, 8% for FUNDED)** - End-of-day account equity cannot drop more than 5% from the end-of-day high water mark.

        The trading day resets at 00:00 UTC for both crypto and forex. Breaching either rule results in immediate elimination.

        Promotion criteria:
        - Returns >= 8% (SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD)
          Returns immediately promoted as soon as they hit 8% returns.
        """
        hotkeys_to_promote = []
        miners_to_eliminate = {}

        asset_classes = self.asset_selection_client.get_asset_selections()
        accounts = self._miner_account_client.get_accounts(list(inspection_hotkeys.keys()))

        for hotkey, bucket_start_time in inspection_hotkeys.items():

            # Unified check: Minimum ledger
            # NOTE not needed?
            has_minimum_ledger, ledger = self._check_minimum_ledger(
                portfolio_only_ledgers, hotkey
            )
            if not has_minimum_ledger or not ledger:
                self._reset_drawdown_stats_cache(hotkey)
                continue

            # Compute portfolio return: (balance + unrealized_pnl) / account_size
            current_return, balance_return = self._compute_portfolio_return(hotkey, accounts.get(hotkey))
            if current_return is None:
                continue

            # returns_percentage = current_return - 1.0 (e.g. 1.08 -> 8%)
            returns_percentage = min(current_return, balance_return) - 1.0

            subaccount_asset_class = asset_classes.get(hotkey)
            if subaccount_asset_class is None:
                bt.logging.error(f"[SYNTH_EVAL {hotkey}] Subaccount does not have asset class - unexpected")
                continue

            returns_threshold = ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD
            if subaccount_asset_class == TradePairCategory.CRYPTO:
                returns_threshold = ValiConfig.SUBACCOUNT_CRYPTO_CHALLENGE_RETURNS_THRESHOLD

            now_ms = current_time if current_time is not None else TimeUtil.now_in_millis()
            midnight_cps, last_eod, daily_open_equity, eod_hwm = self._parse_eod_checkpoints(ledger, now_ms)
            intraday_drawdown_pct = (1.0 - current_return / daily_open_equity) * 100.0
            eod_drawdown_pct = (1.0 - last_eod / eod_hwm) * 100.0

            # Cache stats before rule checks so dashboard reflects what triggered elimination
            self._drawdown_stats_cache[hotkey] = {
                'current_equity': current_return,
                'daily_open_equity': daily_open_equity,
                'eod_hwm': eod_hwm,
                'last_eod_equity': last_eod,
                'intraday_drawdown_pct': intraday_drawdown_pct,
                'eod_drawdown_pct': eod_drawdown_pct,
                'intraday_drawdown_threshold': ValiConfig.SUBACCOUNT_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD,
                'eod_drawdown_threshold': ValiConfig.SUBACCOUNT_CHALLENGE_EOD_DRAWDOWN_THRESHOLD,
                # TODO: remove legacy fields below
                'subaccount_challenge_intraday_drawdown_threshold': ValiConfig.SUBACCOUNT_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD,
                'subaccount_challenge_eod_drawdown_threshold': ValiConfig.SUBACCOUNT_CHALLENGE_EOD_DRAWDOWN_THRESHOLD,
            }

            # Promote if returns meet threshold
            if returns_percentage >= returns_threshold:
                bt.logging.info(
                    f"[SYNTHETIC_CP] {hotkey} promoted - "
                    f"returns {returns_percentage:.2f}% >= {returns_threshold}% "
                    f"balance {accounts.get(hotkey).get('balance')}, unrealized_pnl {self._position_client.get_unrealized_pnl(hotkey)} "
                    f"drawdown stats: {self._drawdown_stats_cache[hotkey]}"
                )
                hotkeys_to_promote.append(hotkey)
                continue

            # Rule 1: Intraday drawdown — current equity cannot drop >5% from today's opening equity
            if current_return < daily_open_equity * (1.0 - ValiConfig.SUBACCOUNT_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD):
                bt.logging.info(
                    f"[SYNTHETIC_CP] {hotkey} intraday drawdown violation, failed challenge period - "
                    f"drawdown stats: {self._drawdown_stats_cache[hotkey]}"
                )
                miners_to_eliminate[hotkey] = (
                    EliminationReason.FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN.value,
                    intraday_drawdown_pct,
                    now_ms
                )
                continue

            # Rule 2: EOD trailing drawdown — last EOD equity cannot drop >5% from highest-ever EOD equity
            if midnight_cps and last_eod < eod_hwm * (1.0 - ValiConfig.SUBACCOUNT_CHALLENGE_EOD_DRAWDOWN_THRESHOLD):
                bt.logging.info(
                    f"[SYNTHETIC_CP] {hotkey} EOD drawdown violation, failed challenge period - "
                    f"drawdown stats: {self._drawdown_stats_cache[hotkey]}"
                )
                miners_to_eliminate[hotkey] = (
                    EliminationReason.FAILED_CHALLENGE_PERIOD_EOD_DRAWDOWN.value,
                    eod_drawdown_pct,
                    midnight_cps[-1].last_update_ms
                )
                continue

            threshold_pct = max(ValiConfig.SUBACCOUNT_CHALLENGE_INTRADAY_DRAWDOWN_THRESHOLD,
                                ValiConfig.SUBACCOUNT_CHALLENGE_EOD_DRAWDOWN_THRESHOLD) * 100.0
            worst_drawdown_pct = max(intraday_drawdown_pct, eod_drawdown_pct)
            near_elimination = worst_drawdown_pct >= threshold_pct * 0.75
            near_promotion = returns_percentage >= returns_threshold * 0.75
            if near_elimination or near_promotion:
                bt.logging.info(
                    f"[SYNTH_EVAL {hotkey}] near elimination - "
                    f"drawdown stats: {self._drawdown_stats_cache[hotkey]}"
                )

        bt.logging.info(
            f"[SYNTH_EVAL] Evaluation complete: {len(inspection_hotkeys)} evaluated, "
            f"{len(hotkeys_to_promote)} to promote, {len(miners_to_eliminate)} to eliminate"
        )

        return hotkeys_to_promote, miners_to_eliminate

    def _evaluate_subaccount_funded(
        self,
        inspection_hotkeys: dict[str, int],
        portfolio_only_ledgers: dict[str, PerfLedger],
        current_time: int = None
    ) -> dict[str, tuple[str, float, int]]:
        """
        Evaluate synthetic hotkeys in SUBACCOUNT_FUNDED bucket.

        Applies the same two drawdown rules as SUBACCOUNT_CHALLENGE, with version-aware thresholds:
        - V0 thresholds (10%) for miners whose SUBACCOUNT_CHALLENGE bucket started before Mar 14, 2026
        - V1 thresholds (8%) for all others

        No returns threshold or promotion logic — funded miners cannot be promoted past this bucket.
        """
        miners_to_eliminate = {}
        accounts = self._miner_account_client.get_accounts(list(inspection_hotkeys.keys()))

        for hotkey, bucket_start_time in inspection_hotkeys.items():
            intraday_threshold, eod_threshold = self._get_drawdown_thresholds(hotkey)

            has_minimum_ledger, ledger = self._check_minimum_ledger(portfolio_only_ledgers, hotkey)
            if not has_minimum_ledger or not ledger:
                self._reset_drawdown_stats_cache(hotkey)
                continue

            current_return, _ = self._compute_portfolio_return(hotkey, accounts.get(hotkey))
            if current_return is None:
                continue

            now_ms = current_time if current_time is not None else TimeUtil.now_in_millis()
            midnight_cps, last_eod, daily_open_equity, eod_hwm = self._parse_eod_checkpoints(ledger, now_ms)

            intraday_drawdown_pct = (1.0 - current_return / daily_open_equity) * 100.0
            eod_drawdown_pct = (1.0 - last_eod / eod_hwm) * 100.0

            # Cache stats before rule checks so dashboard reflects what triggered elimination
            self._drawdown_stats_cache[hotkey] = {
                'current_equity': current_return,
                'daily_open_equity': daily_open_equity,
                'eod_hwm': eod_hwm,
                'last_eod_equity': last_eod,
                'intraday_drawdown_pct': intraday_drawdown_pct,
                'eod_drawdown_pct': eod_drawdown_pct,
                'intraday_drawdown_threshold': intraday_threshold,
                'eod_drawdown_threshold': eod_threshold,
                # TODO: remove legacy fields below
                'subaccount_challenge_intraday_drawdown_threshold': intraday_threshold,
                'subaccount_challenge_eod_drawdown_threshold': eod_threshold,
            }

            # Rule 1: Intraday drawdown
            if current_return < daily_open_equity * (1.0 - intraday_threshold):
                bt.logging.info(
                    f"[FUNDED_EVAL {hotkey}] intraday drawdown violation - "
                    f"drawdown stats: {self._drawdown_stats_cache[hotkey]}"
                )
                miners_to_eliminate[hotkey] = (
                    EliminationReason.FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN.value,
                    intraday_drawdown_pct,
                    now_ms
                )
                continue

            # Rule 2: EOD trailing drawdown
            if midnight_cps and last_eod < eod_hwm * (1.0 - eod_threshold):
                bt.logging.info(
                    f"[FUNDED_EVAL {hotkey}] EOD drawdown violation - "
                    f"drawdown stats: {self._drawdown_stats_cache[hotkey]}"
                )
                miners_to_eliminate[hotkey] = (
                    EliminationReason.FAILED_FUNDED_PERIOD_EOD_DRAWDOWN.value,
                    eod_drawdown_pct,
                    midnight_cps[-1].last_update_ms
                )
                continue

            threshold_pct = max(intraday_threshold, eod_threshold) * 100.0
            worst_drawdown_pct = max(intraday_drawdown_pct, eod_drawdown_pct)
            if worst_drawdown_pct >= threshold_pct * 0.75:
                bt.logging.info(
                    f"[FUNDED_EVAL {hotkey}] near elimination - "
                    f"drawdown stats: {self._drawdown_stats_cache[hotkey]}"
                )

        bt.logging.info(
            f"[FUNDED_EVAL] Evaluation complete: {len(inspection_hotkeys)} evaluated, "
            f"{len(miners_to_eliminate)} to eliminate"
        )

        return miners_to_eliminate

    def _evaluate_rank_based(
        self,
        inspection_hotkeys: dict[str, int],
        positions: dict[str, list[Position]],
        ledger: dict[str, PerfLedger],
        success_hotkeys: list[str],
        probation_hotkeys: list[str],
        current_time: int,
        hk_to_first_order_time: dict[str, int] | None,
        asset_softmaxed_scores: dict[TradePairCategory, dict] | None
    ) -> tuple[list[str], list[str], dict[str, tuple[str, float, int]]]:
        """
        Evaluate hotkeys using rank-based criteria.

        Applied to:
        - All regular hotkeys (CHALLENGE, PROBATION)
        - Synthetic hotkeys in (FUNDED, ALPHA)

        Note: Synthetic hotkeys in MAINCOMP aren't in inspection_hotkeys,
        but can be in success_hotkeys for demotion evaluation.
        """
        miners_to_eliminate = {}
        miners_not_enough_positions = []

        promotion_eligible_hotkeys = []
        rank_eligible_hotkeys = []

        accounts = self._miner_account_client.get_accounts(list(inspection_hotkeys.keys()))

        for hotkey, bucket_start_time in inspection_hotkeys.items():
            bucket = self.get_miner_bucket(hotkey)

            # Get appropriate time limit based on bucket (same for regular and synthetic)
            time_limit_ms = self._get_time_limit_for_bucket(bucket)

            # Unified check: Time limit (bucket-specific)
            should_eliminate, reason = self._check_time_limit(
                hotkey=hotkey,
                bucket_start_time=bucket_start_time,
                current_time=current_time,
                time_limit_ms=time_limit_ms
            )
            if should_eliminate:
                miners_to_eliminate[hotkey] = reason
                continue

            # Unified check: Minimum ledger
            has_minimum_ledger, inspection_ledger = self._check_minimum_ledger(
                ledger, hotkey
            )
            if not has_minimum_ledger:
                continue

            # Unified check: Drawdown during challenge/probation period
            # NOTE: This is for FAILING the challenge period (FAILED_CHALLENGE_PERIOD_DRAWDOWN)
            # EliminationManager separately handles ongoing 10% max drawdown for all miners
            current_return, _ = self._compute_portfolio_return(hotkey, accounts.get(hotkey))
            account = accounts.get(hotkey, {})
            max_return = account.get('max_return', 1.0)
            if current_return is not None:
                drawdown_pct = (1 - current_return / max_return) * 100
                if drawdown_pct >= ValiConfig.DRAWDOWN_MAXVALUE_PERCENTAGE:
                    miners_to_eliminate[hotkey] = (
                        EliminationReason.FAILED_CHALLENGE_PERIOD_DRAWDOWN.value,
                        drawdown_pct,
                        current_time
                    )
                    continue

            # Regular-specific checks (only for regular hotkeys, not synthetic)
            if not is_synthetic_hotkey(hotkey):
                # Re-registration detection
                if not self.running_unit_tests and self.is_recently_re_registered(
                    inspection_ledger, hotkey, hk_to_first_order_time
                ):
                    bt.logging.warning(f'Re-registered hotkey detected: {hotkey}')
                    continue

                # Note: Minimum positions check not necessary if they have ledger.
                # If they have a ledger, they can be scored.

                # Asset class selection check
                if (current_time >= ASSET_CLASS_SELECTION_TIME_MS and
                    not self.asset_selection_client.get_asset_selection(hotkey)):
                    continue

            # Passed basic checks - eligible for ranking
            rank_eligible_hotkeys.append(hotkey)

            # Additional check for promotion: minimum trading days
            min_trading_days = ValiConfig.SUBACCOUNT_FUNDED_MINIMUM_DAYS if bucket == MinerBucket.SUBACCOUNT_FUNDED else ValiConfig.CHALLENGE_PERIOD_MINIMUM_DAYS
            if self.screen_minimum_interaction(inspection_ledger, min_trading_days):
                promotion_eligible_hotkeys.append(hotkey)

        # Calculate dynamic minimum participation days for asset classes
        combined_hotkeys = set(success_hotkeys + probation_hotkeys)
        maincomp_ledger = {hotkey: ledger_data for hotkey, ledger_data in ledger.items() if hotkey in combined_hotkeys}
        asset_classes = list(ValiConfig.ASSET_CLASS_BREAKDOWN.keys())
        asset_class_min_days = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(
            maincomp_ledger, asset_classes
        )
        bt.logging.info(f"challengeperiod_manager asset class minimum days: {asset_class_min_days}")

        all_miner_account_sizes = self._miner_account_client.get_all_miner_account_sizes(timestamp_ms=current_time)

        # Use provided scores dict if available (for testing), otherwise compute scores
        if asset_softmaxed_scores is None:
            # Score all rank-eligible miners (including those without minimum days) for accurate threshold
            scoring_hotkeys = success_hotkeys + rank_eligible_hotkeys
            scoring_ledgers = {hotkey: ledger for hotkey, ledger in ledger.items() if hotkey in scoring_hotkeys}
            scoring_positions = {hotkey: pos_list for hotkey, pos_list in positions.items() if hotkey in scoring_hotkeys}

            asset_competitiveness, asset_softmaxed_scores = Scoring.score_miner_asset_classes(
                    ledger_dict=scoring_ledgers,
                    positions=scoring_positions,
                    asset_class_min_days=asset_class_min_days,
                    evaluation_time_ms=current_time,
                    weighting=True,
                    all_miner_account_sizes=all_miner_account_sizes
            )

            # Cache scores for MinerStatisticsManager
            self._cached_asset_softmaxed_scores = asset_softmaxed_scores
            self._cached_asset_competitiveness = asset_competitiveness


        hotkeys_to_promote, hotkeys_to_demote = self.evaluate_promotions(
            success_hotkeys,
            promotion_eligible_hotkeys,
            asset_softmaxed_scores,
            accounts=accounts
        )

        bt.logging.info(f"[RANK_BASED] Challenge Period: evaluated {len(promotion_eligible_hotkeys)}/{len(inspection_hotkeys)} miners eligible for promotion")
        bt.logging.info(f"[RANK_BASED] Challenge Period: evaluated {len(success_hotkeys)} miners eligible for demotion")
        bt.logging.info(f"[RANK_BASED] Hotkeys to promote: {hotkeys_to_promote}")
        bt.logging.info(f"[RANK_BASED] Hotkeys to demote: {hotkeys_to_demote}")
        bt.logging.info(f"[RANK_BASED] Hotkeys to eliminate: {list(miners_to_eliminate.keys())}")
        bt.logging.info(f"[RANK_BASED] Miners with no positions (skipped): {len(miners_not_enough_positions)}")

        return hotkeys_to_promote, hotkeys_to_demote, miners_to_eliminate

    def _inspect_hotkeys_unified(
        self,
        inspection_hotkeys: dict[str, int],
        current_time: int,
        positions: dict[str, list[Position]],
        ledger: dict[str, PerfLedger],
        success_hotkeys: list[str],
        probation_hotkeys: list[str],
        hk_to_first_order_time: dict[str, int] | None = None,
        asset_softmaxed_scores: dict[TradePairCategory, dict] | None = None
    ) -> tuple[list[str], list[str], dict[str, tuple[str, float, int]]]:
        """
        Unified inspection logic for all hotkeys.

        Branches on: is_synthetic_hotkey(hotkey) AND bucket == MinerBucket.CHALLENGE
        - True: Instantaneous pass criteria (3% returns, 6% drawdown, 90 days)
        - False: Rank-based evaluation (regular miners + synthetic miners post-challenge)
        """
        hotkeys_to_promote = []
        hotkeys_to_demote = []
        miners_to_eliminate = {}

        # Separate into per-bucket evaluation groups
        synthetic_challenge_hotkeys = {}
        synthetic_funded_hotkeys = {}
        rank_based_hotkeys = {}

        for hotkey, bucket_start_time in inspection_hotkeys.items():
            bucket = self.get_miner_bucket(hotkey)

            if is_synthetic_hotkey(hotkey) and bucket == MinerBucket.SUBACCOUNT_CHALLENGE:
                synthetic_challenge_hotkeys[hotkey] = bucket_start_time
            elif is_synthetic_hotkey(hotkey) and bucket == MinerBucket.SUBACCOUNT_FUNDED:
                synthetic_funded_hotkeys[hotkey] = bucket_start_time
            else:
                # Regular miners + synthetic miners in SUBACCOUNT_ALPHA
                rank_based_hotkeys[hotkey] = bucket_start_time

        bt.logging.info(
            f"Inspection split: {len(synthetic_challenge_hotkeys)} synthetic-challenge, "
            f"{len(synthetic_funded_hotkeys)} synthetic-funded, "
            f"{len(rank_based_hotkeys)} rank-based (regular + synthetic post-funded)"
        )

        # PHASE 1: Process synthetic hotkeys in challenge period (instantaneous pass)
        if synthetic_challenge_hotkeys:
            synthetic_promotions, synthetic_eliminations = self._evaluate_synthetic_challenge(
                synthetic_challenge_hotkeys,
                ledger,
                current_time
            )
            hotkeys_to_promote.extend(synthetic_promotions)
            miners_to_eliminate.update(synthetic_eliminations)

        # PHASE 2: Process synthetic hotkeys in funded period (drawdown rules only, no promotion)
        if synthetic_funded_hotkeys:
            funded_eliminations = self._evaluate_subaccount_funded(
                synthetic_funded_hotkeys,
                ledger,
                current_time
            )
            miners_to_eliminate.update(funded_eliminations)

        # PHASE 3: Process rank-based hotkeys (regular flow for all others)
        if rank_based_hotkeys:
            rank_promotions, rank_demotions, rank_eliminations = self._evaluate_rank_based(
                rank_based_hotkeys,
                positions,
                ledger,
                success_hotkeys,
                probation_hotkeys,
                current_time,
                hk_to_first_order_time,
                asset_softmaxed_scores
            )
            hotkeys_to_promote.extend(rank_promotions)
            hotkeys_to_demote.extend(rank_demotions)
            miners_to_eliminate.update(rank_eliminations)

        return hotkeys_to_promote, hotkeys_to_demote, miners_to_eliminate

    def inspect(
        self,
        positions: dict[str, list[Position]],
        ledger: dict[str, PerfLedger],
        success_hotkeys: list[str],
        probation_hotkeys: list[str],
        inspection_hotkeys: dict[str, int],
        current_time: int,
        hk_to_first_order_time: dict[str, int] | None = None,
        asset_softmaxed_scores: dict[TradePairCategory, dict] | None = None,
    ) -> tuple[list[str], list[str], dict[str, tuple[str, float, int]]]:
        """
        Runs a screening process to eliminate miners who didn't pass the challenge period.

        Routes evaluation based on hotkey type:
        - Synthetic hotkeys (entity subaccounts): Threshold-based evaluation (returns ≥3%, drawdown ≤6%, 90 days)
        - Regular hotkeys: Rank-based evaluation with promotion/demotion logic

        Args:
            positions: All miner positions
            ledger: Full ledger data
            success_hotkeys: MAINCOMP hotkeys
            probation_hotkeys: PROBATION hotkeys
            inspection_hotkeys: Dict {hotkey: bucket_start_time}
            current_time: Current time in milliseconds
            hk_to_first_order_time: Dict mapping hotkey to first order time
            asset_softmaxed_scores (dict[TradePairCategory, dict[str, float]) - Optional pre-computed scores dict for testing.
                If provided, skips score calculation. Useful for unit tests.

        Returns:
            hotkeys_to_promote - list of miners that should be promoted from challenge/probation to maincomp
            hotkeys_to_demote - list of miners whose scores were lower than the threshold rank, to be demoted to probation
            miners_to_eliminate - dictionary of hotkey to a tuple of the form (reason failed challenge period, maximum drawdown, detection_time_ms)
        """
        if len(inspection_hotkeys) == 0:
            return [], [], {}  # no hotkeys to inspect

        if not current_time:
            current_time = TimeUtil.now_in_millis()

        # Use unified inspection logic
        hotkeys_to_promote, hotkeys_to_demote, miners_to_eliminate = self._inspect_hotkeys_unified(
            inspection_hotkeys=inspection_hotkeys,
            current_time=current_time,
            positions=positions,
            ledger=ledger,
            success_hotkeys=success_hotkeys,
            probation_hotkeys=probation_hotkeys,
            hk_to_first_order_time=hk_to_first_order_time,
            asset_softmaxed_scores=asset_softmaxed_scores
        )

        bt.logging.info(
            f"Challenge Period: Final results - "
            f"{len(hotkeys_to_promote)} promotions, "
            f"{len(hotkeys_to_demote)} demotions, "
            f"{len(miners_to_eliminate)} eliminations"
        )

        return hotkeys_to_promote, hotkeys_to_demote, miners_to_eliminate

    def evaluate_promotions(
            self,
            success_hotkeys,
            promotion_eligible_hotkeys,
            asset_softmaxed_scores,
            accounts: dict
            ) -> tuple[list[str], list[str]]:
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

        # Filter promotion candidates by minimum returns threshold.
        exceeds_ret_threshold = []
        for hotkey in promote_hotkeys:
            asset_class = miner_asset_selections.get(hotkey)
            returns_threshold = (
                ValiConfig.SUBACCOUNT_CRYPTO_CHALLENGE_RETURNS_THRESHOLD
                if asset_class == TradePairCategory.CRYPTO
                else ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD
            )
            account = accounts.get(hotkey, None)
            result = self._compute_portfolio_return(hotkey, account)
            if result is None:
                bt.logging.info(
                    f"[RANK_BASED] {hotkey} ranked for promotion but blocked - no miner account"
                )
                continue
            returns = result[0] - 1.0
            if returns >= returns_threshold:
                exceeds_ret_threshold.append(hotkey)
            else:
                bt.logging.info(
                    f"[RANK_BASED] {hotkey} ranked for promotion but blocked - "
                    f"returns {returns:.2%} < required {returns_threshold:.2%}"
                )
        promote_hotkeys = exceeds_ret_threshold

        # Demote miners who are no longer in top ranks
        # IMPORTANT: Synthetic hotkeys (subaccounts) can NEVER be demoted from MAINCOMP
        # They stay in MAINCOMP until eliminated by 10% drawdown
        demote_candidates = set(success_hotkeys) - maincomp_hotkeys
        demote_hotkeys = {hk for hk in demote_candidates if not is_synthetic_hotkey(hk)}

        return list(promote_hotkeys), list(demote_hotkeys)

    def get_miner_scores(self) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Get cached miner scores for MinerStatisticsManager.

        Returns:
            tuple containing:
            - asset_softmaxed_scores: dict[asset_class, dict[hotkey, score]]
            - asset_competitiveness: dict[asset_class, competitiveness_score]
        """
        return self._cached_asset_softmaxed_scores, self._cached_asset_competitiveness

    @staticmethod
    def screen_minimum_interaction(ledger_element, min_trading_days) -> bool:
        """Check if miner has minimum number of trading days."""
        if ledger_element is None:
            bt.logging.warning("Ledger element is None. Returning False.")
            return False

        miner_returns = LedgerUtils.daily_return_log(ledger_element)
        return len(miner_returns) >= min_trading_days

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
        return [hotkey for hotkey, history in self.active_miners.items() if history[0].bucket == bucket]

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
                elim_miners_to_return[hotkey] = (EliminationReason.PLAGIARISM.value, -1, current_time)
                self._plagiarism_client.send_plagiarism_elimination_notification(hotkey)

        return elim_miners_to_return

    def _promote_challengeperiod_in_memory(self, hotkeys: list[str], current_time: int):
        """Promote miners to next tier."""
        if len(hotkeys) > 0:
            bt.logging.info(f"Promoting {len(hotkeys)} miners.")

        for hotkey in hotkeys:
            bucket_value = self.get_miner_bucket(hotkey)
            if bucket_value is None:
                bt.logging.error(f"Hotkey {hotkey} is not an active miner. Skipping promotion")
                continue

            # Determine target bucket based on current bucket
            if bucket_value == MinerBucket.CHALLENGE:
                target_bucket = MinerBucket.MAINCOMP
            elif bucket_value == MinerBucket.PROBATION:
                target_bucket = MinerBucket.MAINCOMP
            elif bucket_value == MinerBucket.SUBACCOUNT_CHALLENGE:
                target_bucket = MinerBucket.SUBACCOUNT_FUNDED

                # Close all existing positions
                self._position_client.close_all_positions(
                    hotkey=hotkey,
                    close_time_ms=current_time,
                    order_source=OrderSource.SUBACCOUNT_PROMOTION
                )
                # Reset account fields (PnL, capital used, borrowed amount, interest)
                self._miner_account_client.reset_account_fields(hotkey)
                # Archive all positions (disk move + memory removal)
                self._position_client.archive_positions_for_hotkey(hotkey, archive_all=True)
                # Wipe perf ledgers so funded-period performance is tracked from scratch
                self._perf_ledger_client.wipe_miners_perf_ledgers([hotkey])
                # Delete debt ledger to match new perf ledger checkpoints
                self._debt_ledger_client.delete_debt_ledger(hotkey)

                self._reset_drawdown_stats_cache(hotkey)

            elif bucket_value == MinerBucket.SUBACCOUNT_FUNDED:
                # Synthetic funded miners cannot be promoted past SUBACCOUNT_FUNDED
                if is_synthetic_hotkey(hotkey):
                    bt.logging.error(f"Unexpected promotion attempt for synthetic funded hotkey {hotkey} — skipping")
                    continue
                target_bucket = MinerBucket.SUBACCOUNT_ALPHA
            else:
                bt.logging.error(f"Cannot promote {hotkey} from bucket {bucket_value.value}")
                continue

            bt.logging.info(f"Promoting {hotkey} from {bucket_value.value} to {target_bucket.value}")
            self.set_miner_bucket(hotkey, target_bucket, current_time)

    def _promote_plagiarism_to_previous_bucket_in_memory(self, hotkeys: list[str], current_time):
        """Promote plagiarism miners back to their most recent non-plagiarism bucket."""
        if len(hotkeys) > 0:
            bt.logging.info(f"Promoting {len(hotkeys)} plagiarism miners to probation.")

        for hotkey in hotkeys:
            try:
                history = self.active_miners.get(hotkey)
                if not history or history[0].bucket != MinerBucket.PLAGIARISM:
                    bt.logging.error(f"Hotkey {hotkey} is not an active plagiarism miner. Skipping promotion")
                    continue

                # Remove the most recent plagiarism entry, restoring the previous bucket
                history.pop(0)

                if not history:
                    bt.logging.error(f"No previous bucket found for {hotkey} after removing PLAGIARISM entry. Skipping promotion")
                    self.remove_miner(hotkey)
                    continue

                bt.logging.info(f"Promoting {hotkey} from PLAGIARISM to {history[0].bucket.value}")

                # Push restored bucket to MinerAccount
                try:
                    self._miner_account_client.set_miner_bucket(hotkey, history[0].bucket)
                except Exception as e:
                    bt.logging.warning(f"Failed to push miner_bucket to MinerAccount for {hotkey}: {e}")

                # Send Slack notification
                self._plagiarism_client.send_plagiarism_promotion_notification(hotkey)
            except Exception as e:
                bt.logging.error(f"Failed to promote {hotkey} from plagiarism at time {current_time}: {e}")

    def _eliminate_challengeperiod_in_memory(self, eliminations_with_reasons: dict[str, tuple[str, float, int]]):
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
                bt.logging.info(f"Demoting {hotkey} to PLAGIARISM from {prev_bucket_value}")
                self.set_miner_bucket(hotkey, MinerBucket.PLAGIARISM, current_time)

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

        # Get all buckets that should NOT be re-added to challenge period
        maincomp_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.MAINCOMP)
        probation_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.PROBATION)
        plagiarism_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.PLAGIARISM)
        subaccount_funded_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.SUBACCOUNT_FUNDED)
        subaccount_alpha_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.SUBACCOUNT_ALPHA)
        subaccount_challenge_hotkeys = self.get_hotkeys_by_bucket(MinerBucket.SUBACCOUNT_CHALLENGE)

        # Combine all buckets that should skip re-adding
        skip_buckets = (
            set(maincomp_hotkeys) | set(probation_hotkeys) | set(plagiarism_hotkeys) |
            set(subaccount_funded_hotkeys) | set(subaccount_alpha_hotkeys) | set(subaccount_challenge_hotkeys)
        )

        any_changes = False
        for hotkey in new_hotkeys:
            # Skip if miner is in persisted eliminations
            if hotkey in elimination_hotkeys:
                continue

            # Skip if miner is in local eliminations
            if hotkey in local_elimination_hotkeys:
                bt.logging.info(f"[CP_DEBUG] Skipping {hotkey[:16]}...{hotkey[-8:]} - in eliminations_with_reasons (not yet persisted)")
                continue

            # Skip if miner is already in a bucket (success, probation, plagiarism, or subaccount buckets)
            if hotkey in skip_buckets:
                continue

            first_order_time = hk_to_first_order_time.get(hotkey)
            if first_order_time is None:
                if not self.has_miner(hotkey):
                    if is_synthetic_hotkey(hotkey):
                        self.set_miner_bucket(hotkey, MinerBucket.SUBACCOUNT_CHALLENGE, default_time)
                    else:
                        self.set_miner_bucket(hotkey, MinerBucket.CHALLENGE, default_time)
                    bt.logging.info(f"Adding {hotkey} to challenge period with start time {default_time}")
                    any_changes = True
                continue

            # Has a first order time but not yet stored in memory or start time is set as default
            start_time = self.get_miner_start_time(hotkey)
            if not self.has_miner(hotkey) or start_time != first_order_time:
                if is_synthetic_hotkey(hotkey):
                    self.set_miner_bucket(hotkey, MinerBucket.SUBACCOUNT_CHALLENGE, default_time)
                else:
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
            bucket = self.get_miner_bucket(hotkey)

            # Fix synthetic hotkeys incorrectly in CHALLENGE bucket
            if is_synthetic_hotkey(hotkey) and bucket == MinerBucket.CHALLENGE:
                bt.logging.info(f"Fixing synthetic hotkey {hotkey} from CHALLENGE to SUBACCOUNT_CHALLENGE")
                self.set_miner_bucket(hotkey, MinerBucket.SUBACCOUNT_CHALLENGE, start_time_ms, replace_bucket=True)
                any_changes = True

            if hotkey not in hk_to_first_order_time_ms:
                continue
            first_order_time_ms = hk_to_first_order_time_ms[hotkey]

            if start_time_ms != first_order_time_ms:
                bt.logging.info(f"Challengeperiod start time for {hotkey} updated from: {datetime.fromtimestamp(start_time_ms/1000)} "
                                f"to: {datetime.fromtimestamp(first_order_time_ms/1000)}, {(start_time_ms-first_order_time_ms)/1000}s delta")
                if is_synthetic_hotkey(hotkey):
                    self.set_miner_bucket(hotkey, MinerBucket.SUBACCOUNT_CHALLENGE, first_order_time_ms)
                else:
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
            hotkeys=self._position_client.get_all_hotkeys())

        self._add_challengeperiod_testing_in_memory_and_disk(
            new_hotkeys=self._position_client.get_all_hotkeys(),
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
        replace_bucket: bool = False,
    ) -> bool:
        """
        Set or update a miner's bucket information.

        Prepends a new BucketEntry to the history on bucket change; updates in-place for
        same-bucket refreshes. The previous bucket is always preserved as history[1].

        Args:
            hotkey: Miner's hotkey
            bucket: New bucket to assign
            start_time: Start time for new bucket
            replace_bucket: Update newest bucket in place

        Returns:
            True if this is a new miner, False if updating existing
        """
        is_new = hotkey not in self.active_miners
        new_entry = BucketEntry(bucket, start_time)

        if is_new:
            self.active_miners[hotkey] = [new_entry]
        else:
            history = self.active_miners[hotkey]
            if replace_bucket or history[0].bucket == bucket:
                # Same bucket — update in place
                history[0] = new_entry
            else:
                # Bucket changed — prepend new entry, keeping full history
                history.insert(0, new_entry)

        # Push bucket to MinerAccount
        try:
            self._miner_account_client.set_miner_bucket(hotkey, bucket)
        except Exception as e:
            bt.logging.warning(f"Failed to push miner_bucket to MinerAccount for {hotkey}: {e}")

        return is_new

    def get_miner_start_time(self, hotkey: str) -> Optional[int]:
        """Get the start time of a miner's current bucket."""
        history = self.active_miners.get(hotkey)
        return history[0].start_time_ms if history else None

    def has_miner(self, hotkey: str) -> bool:
        """Fast check if a miner is in active_miners (O(1))."""
        return hotkey in self.active_miners

    def remove_miner(self, hotkey: str) -> bool:
        """Remove a miner from active_miners."""
        if hotkey in self.active_miners:
            del self.active_miners[hotkey]
            # Clear bucket on MinerAccount
            try:
                self._miner_account_client.set_miner_bucket(hotkey, None)
            except Exception as e:
                bt.logging.warning(f"Failed to clear miner_bucket on MinerAccount for {hotkey}: {e}")
            return True
        return False

    def _sync_all_miner_buckets(self):
        """Push all current miner buckets to MinerAccount on startup."""
        synced = 0
        for hotkey, history in self.active_miners.items():
            try:
                self._miner_account_client.set_miner_bucket(hotkey, history[0].bucket)
                synced += 1
            except Exception as e:
                bt.logging.warning(f"Failed to sync miner_bucket for {hotkey}: {e}")
        bt.logging.info(f"[CP_MANAGER] Synced {synced}/{len(self.active_miners)} miner buckets to MinerAccount")

    def clear_active_miners(self):
        """Clear all miners from active_miners."""
        self.active_miners.clear()

    def update_active_miners(self, miners_dict: dict) -> int:
        """
        Bulk update active_miners from a dict.

        Args:
            miners_dict: Can be either:
                - Dict mapping hotkey to List[BucketEntry]
                - Dict mapping hotkey to list of dicts [{"bucket": ..., "bucket_start_time": ...}, ...] (RPC)

        Returns:
            Number of miners updated
        """
        normalized_dict = {}
        for hotkey, data in miners_dict.items():
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], BucketEntry):
                # Already in List[BucketEntry] format
                normalized_dict[hotkey] = data
            elif isinstance(data, list):
                # RPC list of dicts format (or empty list)
                normalized_dict[hotkey] = [
                    BucketEntry(
                        bucket=MinerBucket(entry["bucket"]) if isinstance(entry["bucket"], str) else entry["bucket"],
                        start_time_ms=entry["bucket_start_time"]
                    )
                    for entry in data
                ]
            else:
                raise ValueError(f"Invalid data type for miner {hotkey}: {type(data)}")

        count = len(normalized_dict)
        self.active_miners.update(normalized_dict)
        return count

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

    def pop_elimination_reason(self, hotkey: str) -> Optional[Tuple[str, float, int]]:
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

    def get_miner_bucket(self, hotkey, timestamp_ms: Optional[int] = None) -> Optional[MinerBucket]:
        """Get the bucket of a miner, optionally at a specific timestamp.

        Args:
            hotkey: Miner's hotkey
            timestamp_ms: If provided, returns the bucket active at that time.
                          If None, returns the current bucket (history[0]).

        Returns:
            The MinerBucket active at timestamp_ms, or None if not found.
        """
        history = self.active_miners.get(hotkey)
        if not history:
            return None
        if timestamp_ms is None:
            return history[0].bucket
        # History is newest-first; find the first entry whose start_time_ms <= timestamp_ms
        for entry in history:
            if entry.start_time_ms <= timestamp_ms:
                return entry.bucket
        return None

    def get_dashboard(self, hotkey) -> dict | None:
        history = self.active_miners.get(hotkey)
        if history is None:
            return None

        return {
            "bucket": history[0].bucket.value,
            "start_time_ms": history[0].start_time_ms,
        }

    # TODO: revisit to separate regular and subaccount miners
    def get_testing_miners(self):
        """Get all testing bucket miners (CHALLENGE + SUBACCOUNT_CHALLENGE)."""
        challenge = self._bucket_view(MinerBucket.CHALLENGE)
        subaccount_challenge = self._bucket_view(MinerBucket.SUBACCOUNT_CHALLENGE)
        return copy.deepcopy({**challenge, **subaccount_challenge})

    # TODO: revisit to separate regular and subaccount miners
    def get_success_miners(self):
        """Get all success bucket miners (MAINCOMP + SUBACCOUNT_FUNDED + SUBACCOUNT_ALPHA)."""
        maincomp = self._bucket_view(MinerBucket.MAINCOMP)
        funded = self._bucket_view(MinerBucket.SUBACCOUNT_FUNDED)
        alpha = self._bucket_view(MinerBucket.SUBACCOUNT_ALPHA)
        return copy.deepcopy({**maincomp, **funded, **alpha})

    def get_funded_miners(self):
        """Get all SUBACCOUNT_FUNDED bucket miners."""
        return copy.deepcopy(self._bucket_view(MinerBucket.SUBACCOUNT_FUNDED))

    def get_probation_miners(self):
        """Get all PROBATION bucket miners."""
        return copy.deepcopy(self._bucket_view(MinerBucket.PROBATION))

    def get_plagiarism_miners(self):
        """Get all PLAGIARISM bucket miners."""
        return copy.deepcopy(self._bucket_view(MinerBucket.PLAGIARISM))

    def _bucket_view(self, bucket: MinerBucket):
        """Get all miners in a specific bucket as {hotkey: start_time} dict."""
        return {hk: history[0].start_time_ms for hk, history in self.active_miners.items() if history[0].bucket == bucket}

    def to_checkpoint_dict(self):
        """Get challenge period data as a checkpoint dict for serialization."""
        json_dict = {}
        for hotkey, history in self.active_miners.items():
            json_dict[hotkey] = [
                {"bucket": entry.bucket.value, "bucket_start_time": entry.start_time_ms}
                for entry in history
            ]
        return json_dict

    @staticmethod
    def parse_checkpoint_dict(json_dict) -> Dict[str, List['BucketEntry']]:
        """Parse checkpoint dict from disk. Handles 3 formats:
        1. Legacy testing/success format: {"testing": {hk: time}, "success": {hk: time}}
        2. Current dict format: {hk: {"bucket": ..., "bucket_start_time": ..., "previous_bucket": ..., ...}}
        3. New list format: {hk: [{"bucket": ..., "bucket_start_time": ...}, ...]}
        """
        formatted_dict = {}

        if "testing" in json_dict.keys() and "success" in json_dict.keys():
            # Legacy format
            testing = json_dict.get("testing", {})
            success = json_dict.get("success", {})
            for hotkey, start_time in testing.items():
                formatted_dict[hotkey] = [BucketEntry(MinerBucket.CHALLENGE, start_time)]
            for hotkey, start_time in success.items():
                formatted_dict[hotkey] = [BucketEntry(MinerBucket.MAINCOMP, start_time)]
        else:
            for hotkey, info in json_dict.items():
                if isinstance(info, list):
                    # New list format
                    formatted_dict[hotkey] = [
                        BucketEntry(
                            bucket=MinerBucket(entry["bucket"]),
                            start_time_ms=entry["bucket_start_time"]
                        )
                        for entry in info
                    ]
                elif isinstance(info, dict):
                    # Current dict format
                    bucket = MinerBucket(info["bucket"]) if info.get("bucket") else None
                    bucket_start_time = info.get("bucket_start_time")
                    history = [BucketEntry(bucket, bucket_start_time)]

                    previous_bucket = MinerBucket(info["previous_bucket"]) if info.get("previous_bucket") else None
                    previous_bucket_start_time = info.get("previous_bucket_start_time")
                    if previous_bucket is not None and previous_bucket_start_time is not None:
                        history.append(BucketEntry(previous_bucket, previous_bucket_start_time))

                    formatted_dict[hotkey] = history

        return formatted_dict
