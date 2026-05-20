# developer: trdougherty, jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ChallengePeriodManager - Core business logic for challenge period management.

This manager handles all heavy logic for challenge period operations.
ChallengePeriodServer wraps this and exposes methods via RPC.

This follows the same pattern as EliminationManager.
"""
from dataclasses import asdict, dataclass, field

import bittensor as bt
import threading
from typing import Dict, List, Optional
from datetime import datetime

from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePairCategory, ValiConfig, RPCConnectionMode
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

@dataclass
class DrawdownStats:
    current_equity: float = 1.0
    current_balance: float = 1.0
    daily_open_equity: float = 1.0
    eod_hwm: float = 1.0
    last_eod_equity: float = 1.0
    intraday_drawdown_pct: float = 0.0
    eod_drawdown_pct: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MinerBucketState:
    entries: List[BucketEntry]
    drawdown: DrawdownStats = field(default_factory=DrawdownStats)
    rank: int | None = None

    def __post_init__(self):
        if not self.entries:
            raise ValueError("BucketInfo must be initialized with BucketEntries")
        self.entries = sorted(self.entries, key=lambda x: x.start_time_ms)

    def add_bucket_entry(self, bucket: MinerBucket, time_ms: int, *, replace_top: bool = False) -> bool:
        if self.current_bucket == bucket:
            if replace_top:
                self.entries[-1] = BucketEntry(bucket, time_ms)
                return True
            return False
        else:
            self.entries.append(BucketEntry(bucket, time_ms))
            return False

    def pop_bucket_entry(self, top_bucket: MinerBucket) -> BucketEntry | None:
        if self.current_bucket == top_bucket:
            return self.entries.pop()
        return None

    def bucket(self, time_ms: int | None = None) -> MinerBucket:
        if time_ms is None:
            return self.entries[-1].bucket
        for entry in reversed(self.entries):
            if entry.start_time_ms >= time_ms:
                return entry.bucket
        raise ValueError(f"No bucket found for time {time_ms}")

    def to_json(self):
        """Only sync or save bucket entries - drawdown/rank should not be synced."""
        return [bucket.to_dict for bucket in self.entries]

    @property
    def current_bucket(self):
        return self.bucket()

    @property
    def current_bucket_start_ms(self) -> int:
        return self.entries[-1].start_time_ms

    @property
    def current_bucket_entry(self) -> BucketEntry:
        return self.entries[-1]

    @property
    def intraday_drawdown_threshold(self):
        return self.current_bucket.intraday_drawdown_threshold(self.current_bucket_start_ms)

    @property
    def intraday_drawdown_threshold_pct(self):
        return self.current_bucket.intraday_drawdown_threshold(self.current_bucket_start_ms) * 100

    @property
    def eod_drawdown_threshold(self):
        return self.current_bucket.eod_drawdown_threshold(self.current_bucket_start_ms)

    @property
    def eod_drawdown_threshold_pct(self):
        return self.current_bucket.eod_drawdown_threshold(self.current_bucket_start_ms) * 100


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

        self._perf_ledger_client = PerfLedgerClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._position_client = PositionManagerClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._elimination_client = EliminationClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._plagiarism_client = PlagiarismClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._miner_account_client = MinerAccountClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._common_data_client = CommonDataClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._asset_selection_client = AssetSelectionClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._debt_ledger_client = DebtLedgerClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)

        self.CHALLENGE_FILE = ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=running_unit_tests)
        self._current_iteration_epoch = None

        self._buckets_lock = threading.Lock()
        self.miner_states: dict[str, MinerBucketState] = {}
        self.miner_states: dict[str, MinerBucketState] = self._read_qwer_from_disk()

        # Cached scores for MinerStatisticsManager
        self._cached_asset_softmaxed_scores: Dict[TradePairCategory, Dict[str, float]] = {}
        self._cached_asset_competitiveness: Dict[TradePairCategory, float] = {}

        bt.logging.info("[CP_MANAGER] ChallengePeriodManager initialized with {len(self.qwer_buckets)} state data")

    # ==================== Core Business Logic ====================

    def refresh(self, current_time_ms: int | None = None, iteration_epoch: int | None = None):
        if current_time_ms is None:
            current_time_ms = TimeUtil.now_in_millis()

        asset_selections = self._asset_selection_client.get_asset_selections()
        hk_to_positions, hk_to_first_order_time = self._position_client.filtered_positions_for_scoring(
            hotkeys=self._position_client.get_all_hotkeys()
        )

        hotkeys_elimination_sync = list(self._elimination_client.get_eliminated_hotkeys())
        hotkeys_plagiarism_sync = list(self._plagiarism_client.get_plagiarism_miners())

        updated_from_sync = False
        updated_from_sync |= self._sync_positions(
            hotkeys=list(hk_to_positions.keys()),
            eliminated_hotkeys=hotkeys_elimination_sync,
            hk_to_first_order_time_ms=hk_to_first_order_time,
            default_time=current_time_ms
        )

        updated_from_sync |= self.sync_plagiarism_miners(hotkeys_plagiarism_sync, current_time_ms)
        updated_from_sync |= self.sync_elimination_miners(hotkeys_elimination_sync)
        updated_from_sync |= self._prune_deregistered_metagraph()

        self._current_iteration_epoch = iteration_epoch

        evaluation_hotkeys = [hotkey for hotkey, state in self.miner_states.items() if state.current_bucket.is_evaluation_eligible]
        rank_hotkeys = [hotkey for hotkey, state in self.miner_states.items() if state.current_bucket.is_rank_based]
        accounts = self._miner_account_client.get_accounts(evaluation_hotkeys)
        ledgers = self._perf_ledger_client.filtered_ledger_for_scoring(evaluation_hotkeys)

        self._refresh_drawdown_cache(evaluation_hotkeys, accounts, ledgers, current_time_ms)
        self._refresh_rank_cache(rank_hotkeys, ledgers, hk_to_positions, accounts, asset_selections, current_time_ms)

        eliminations = {}
        promotions, demotions = [], []
        for hotkey, state in self.miner_states.items():
            # Check time-based eliminations first for regular challenge, probation miners
            if self._failed_time(state.current_bucket_entry, current_time_ms):
                eliminations[hotkey] = EliminationReason.FAILED_CHALLENGE_PERIOD_TIME
                continue

            # Rule 1: Intraday drawdown — current equity cannot drop below  from today's opening equity
            # intraday_drawdown_pct = (1.0 - current_return / daily_open_equity) * 100.0
            if state.drawdown.intraday_drawdown_pct > state.intraday_drawdown_threshold_pct:
                eliminations[hotkey] = EliminationReason.FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN
                continue

            # Rule 2: EOD trailing drawdown — last EOD equity cannot drop below threshold(0.0n) from highest-ever EOD equity
            # eod_drawdown_pct = (1.0 - last_eod / eod_hwm) * 100.0
            if state.drawdown.eod_drawdown_pct > state.eod_drawdown_threshold_pct:
                eliminations[hotkey] = EliminationReason.FAILED_CHALLENGE_PERIOD_EOD_DRAWDOWN
                continue

            _asset = asset_selections.get(hotkey)
            if _asset is None: continue
            asset_class = TradePairCategory(_asset)

            # Check demotions for regular maincomp before promotion
            returns_threshold = ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD[asset_class]
            if self._check_demotion(state, returns_threshold=returns_threshold):
                demotions.append(hotkey)
                continue

            if self._check_promotion(state, returns_threshold, current_time_ms):
                promotions.append(hotkey)
                continue

        updated = False
        updated |= self.eliminate_hotkeys(eliminations, current_time_ms)
        updated |= self.demote_hotkeys(demotions, current_time_ms)
        updated |= self.promote_hotkeys(promotions, current_time_ms)

        if updated_from_sync or updated:
            self._sync_buckets_to_accounts()
            self._save_to_disk()

    # ==================== Evaluation Methods ====================

    @staticmethod
    def _failed_time(current_bucket_entry: BucketEntry, current_time_ms: int) -> bool:
        bucket = current_bucket_entry.bucket
        if bucket.time_limit_ms is None:
            return False
        return current_time_ms - current_bucket_entry.start_time_ms > bucket.time_limit_ms

    @staticmethod
    def _check_demotion(state: MinerBucketState, returns_threshold: float = ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD_DEFAULT):
        if state.current_bucket == MinerBucket.MAINCOMP:
            return (state.rank > ValiConfig.PROMOTION_THRESHOLD_RANK
                    or state.drawdown.current_equity < returns_threshold
                    if state.rank is not None
                    else False)
        return False

    @staticmethod
    def _check_promotion(state: MinerBucketState, returns_threshold: float, current_time_ms: int):
        if state.current_bucket == MinerBucket.CHALLENGE:
            if current_time_ms - state.current_bucket_start_ms < ValiConfig.CHALLENGE_PERIOD_MINIMUM_MS:
                return False

        if min(state.drawdown.current_equity, state.drawdown.current_balance) > returns_threshold:
            if state.current_bucket.is_rank_based:
                return state.rank <= ValiConfig.PROMOTION_THRESHOLD_RANK if state.rank else False
            else:
                return True
        return False

    def eliminate_hotkeys(self, eliminations: dict[str, EliminationReason], current_time_ms: int):
        if eliminations:
            bt.logging.info(f"Elimination {len(eliminations)} miners from challenge period")

        elimination_data = {}
        with self._buckets_lock:
            for hotkey, elimination_reason in eliminations.items():
                current_bucket = self.miner_states[hotkey].current_bucket
                if not current_bucket.is_elimination_eligible:
                    bt.logging.warning(f"Attempted to eliminate {hotkey} in {current_bucket.value}")

                elimination_data[hotkey] = {
                        "hotkey": hotkey,
                        "reason": elimination_reason.value,
                        "elimination_initiated_time_ms": current_time_ms
                        }

        for hotkey, data in elimination_data.items():
            bt.logging.info(f"[CHALLENGE] Eliminating {hotkey}")
            self._elimination_client.add_elimination(hotkey, data)

        return self.remove_hotkeys(list(eliminations.keys()))

    def demote_hotkeys(self, hotkeys: list[str], current_time_ms) -> bool:
        """Demote miners to probation."""
        if hotkeys:
            bt.logging.info(f"[CHALLENGE] Demoting {len(hotkeys)} miners to probation")

        updated = False
        with self._buckets_lock:
            for hotkey in hotkeys:
                bt.logging.info(f"[CHALLENGE] Demoting {hotkey} to PROBATION")
                updated |= self.miner_states[hotkey].add_bucket_entry(MinerBucket.PROBATION, current_time_ms)
        return updated

    def promote_hotkeys(self, hotkeys: list[str], current_time_ms: int) -> bool:
        """Promote miners to next tier."""
        if len(hotkeys) > 0:
            bt.logging.info(f"[CHALLENGE] Promoting {len(hotkeys)} miners.")

        updated = False
        for hotkey in hotkeys:
            current_bucket = self.miner_states[hotkey].current_bucket
            target_bucket = current_bucket.next_bucket
            if target_bucket is None:
                bt.logging.warning(f"[CHALLENGE] Attempted to promote {hotkey} in {current_bucket.value}")
                continue

            bt.logging.info(f"[CHALLENGE] Promoting {hotkey} from {current_bucket.value} to {target_bucket.value}")
            if target_bucket == MinerBucket.SUBACCOUNT_FUNDED:
                # Close all existing positions
                self._position_client.close_all_positions(
                    hotkey=hotkey,
                    close_time_ms=current_time_ms,
                    order_source=OrderSource.SUBACCOUNT_PROMOTION
                )
                # Reset account fields (PnL, capital used, borrowed amount, interest)
                self._miner_account_client.reset_account_fields(hotkey, target_bucket)
                # Archive all positions (disk move + memory removal)
                self._position_client.archive_positions_for_hotkey(hotkey, archive_all=True)
                # Wipe perf ledgers so funded-period performance is tracked from scratch
                self._perf_ledger_client.wipe_miners_perf_ledgers([hotkey])
                # Delete debt ledger to match new perf ledger checkpoints
                self._debt_ledger_client.delete_debt_ledger(hotkey)
                # Reset drawdown cache
                self._reset_drawdown_stats_cache(hotkey)

            with self._buckets_lock:
                updated |= self.miner_states[hotkey].add_bucket_entry(target_bucket, current_time_ms)

        return updated

    # ==================== Drawdown/Rank Refresh methods ====================

    def _compute_portfolio_return(self, hotkey: str, account: Optional[dict] = None) -> tuple[float | None, float | None]:
        """Compute current portfolio return as (balance + unrealized_pnl) / account_size.

        Returns None if account data is unavailable.
        """
        if account is None:
            return None, None
        account_size = account.get('account_size', 0)
        if account_size <= 0:
            return None, None
        balance = account.get('balance', 0)
        unrealized_pnl = self._position_client.get_unrealized_pnl(hotkey)
        equity = balance + unrealized_pnl

        equity_ret = equity / account_size
        balance_ret = balance / account_size

        return equity_ret, balance_ret

    def _parse_eod_checkpoints(self, ledger: PerfLedger, now_ms: int) -> tuple[float, float, float]:
        """
        Parse midnight checkpoints from a ledger.
        Returns (last_eod, daily_open_equity, eod_hwm).
        """
        midnight_cps = [cp for cp in ledger.cps if cp.last_update_ms % 86400000 == 0 and cp.equity_ret > 0]
        last_eod = midnight_cps[-1].equity_ret if midnight_cps else 1.0
        today_midnight_ms = (now_ms // 86400000) * 86400000
        today_open_cp = next((cp for cp in midnight_cps if cp.last_update_ms == today_midnight_ms), None)
        daily_open_equity = today_open_cp.equity_ret if today_open_cp else last_eod
        eod_hwm = max(max(cp.equity_ret for cp in midnight_cps), 1.0) if midnight_cps else 1.0
        return last_eod, daily_open_equity, eod_hwm

    def _refresh_drawdown_cache(self, hotkeys, accounts, ledgers, current_time_ms) -> None:
        for hotkey in hotkeys:
            # Compute portfolio return: (balance + unrealized_pnl) / account_size
            current_equity, current_balance = self._compute_portfolio_return(hotkey, accounts.get(hotkey))
            if current_equity is None or current_balance is None:
                continue

            ledger = ledgers.get(hotkey)
            if ledger is None:
                continue

            now_ms = current_time_ms if current_time_ms is not None else TimeUtil.now_in_millis()
            last_eod, daily_open_equity, eod_hwm = self._parse_eod_checkpoints(ledger, now_ms)
            intraday_drawdown_pct = (1.0 - current_equity / daily_open_equity) * 100.0
            eod_drawdown_pct = (1.0 - last_eod / eod_hwm) * 100.0

            # Cache stats before rule checks so dashboard reflects what triggered elimination
            self.miner_states[hotkey].drawdown = DrawdownStats(
                current_equity=current_equity,
                current_balance=current_balance,
                daily_open_equity=daily_open_equity,
                eod_drawdown_pct=eod_drawdown_pct,
                eod_hwm=eod_hwm,
                intraday_drawdown_pct=intraday_drawdown_pct,
                last_eod_equity=last_eod
            )

    def _refresh_rank_cache(
        self,
        hotkeys: list[str],
        ledgers: dict[str,PerfLedger],
        positions: dict[str,list[Position]],
        accounts: dict[str,dict],
        asset_selections: dict[str,TradePairCategory],
        current_time_ms: int
    ):
        asset_classes = list(set(asset_selections.values()))
        asset_class_min_days = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(ledgers, asset_classes)

        rank_ledgers = {hk: ledger for hk, ledger in ledgers.items() if hk in hotkeys}
        rank_positions = {hk: pos for hk, pos in positions.items() if hk in hotkeys}
        account_sizes = {hk: account["account_size"] for hk, account in accounts.items() if hk in hotkeys}

        # Score all rank-eligible miners (including those without minimum days) for accurate threshold
        asset_competitiveness, asset_softmaxed_scores = Scoring.score_miner_asset_classes(
            ledger_dict=rank_ledgers,
            positions=rank_positions,
            asset_class_min_days=asset_class_min_days,
            evaluation_time_ms=current_time_ms,
            weighting=True,
            all_miner_account_sizes=account_sizes
        )

        # Cache scores for MinerStatisticsManager
        self._cached_asset_softmaxed_scores = asset_softmaxed_scores
        self._cached_asset_competitiveness = asset_competitiveness

        for asset_class, asset_scores in asset_softmaxed_scores.items():
            # Filter to only include miners who selected this asset class when calculating threshold
            miner_scores = {
                hotkey: score for hotkey, score in asset_scores.items()
                if asset_selections[hotkey] == asset_class
            }
            sorted_scores = sorted(miner_scores.items(), key=lambda item: item[1], reverse=True)
            for i, (hotkey, _) in enumerate(sorted_scores):
                self.miner_states[hotkey].rank = i + 1

    def _reset_drawdown_stats_cache(self, hotkey: str) -> None:
        """Reset a hotkey's drawdown stats cache to neutral default values. """
        self.miner_states[hotkey].drawdown = DrawdownStats()


    # ==================== Sync Methods ====================

    def sync_challenge_period_data(self, qwer_buckets_data):
        """Sync challenge period data from another validator."""
        if not qwer_buckets_data:
            bt.logging.error(f'challenge_period_data appears invalid')

        with self._buckets_lock:
            self.miner_states.clear()
            self.miner_states.update(self.parse_checkpoint_dict(qwer_buckets_data))
            self._save_to_disk()

    def sync_plagiarism_miners(self, plagiarism_miners: list[str], current_time_ms: int) -> bool:
        """Sync plagiarism miners status from plagiarism api."""
        with self._buckets_lock:
            updated = False
            for hotkey in plagiarism_miners:
                updated |= self.miner_states[hotkey].add_bucket_entry(MinerBucket.PLAGIARISM, current_time_ms)

            whitelisted_miners = set(self.get_hotkeys_by_bucket(MinerBucket.PLAGIARISM)) - set(plagiarism_miners)
            for hotkey in whitelisted_miners:
                if self.miner_states[hotkey].current_bucket != MinerBucket.PLAGIARISM:
                    continue
                popped_bucket_entry = self.miner_states[hotkey].pop_bucket_entry(top_bucket=MinerBucket.PLAGIARISM)
                updated |= popped_bucket_entry is not None

        return updated

    def sync_elimination_miners(self, elimination_miners: list[str]) -> bool:
        """Sync eliminated miners from elimination manager. Method for peace of mind."""
        return self.remove_hotkeys(elimination_miners)

    def get_hotkeys_by_bucket(self, buckets: MinerBucket | list[MinerBucket]) -> list[str]:
        """Get all hotkeys in bucket or a list of buckets."""
        bucket_set = {buckets} if isinstance(buckets, MinerBucket) else set(buckets)
        return [hotkey for hotkey, state in self.miner_states.items() if state.current_bucket in bucket_set]

    def _save_to_disk(self):
        """Write challenge period data from memory to disk."""
        if self.is_backtesting:
            return

        # Epoch-based validation: check if sync occurred during our iteration
        if self._current_iteration_epoch is not None:
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

        state_changed = False
        hotkeys_prune = []
        for hotkey in self.miner_states.keys():
            if hotkey not in hotkeys:
                bucket = self.get_miner_bucket(hotkey)
                # Entity miners do not have positions. skip pruning
                if bucket in [MinerBucket.ENTITY, MinerBucket.SUBACCOUNT_FUNDED]:
                    continue
                state_changed = True

        self.remove_hotkeys(hotkeys_prune)
        return state_changed

    def _sync_positions(
        self,
        hotkeys: list[str],
        eliminated_hotkeys: list[str],
        hk_to_first_order_time_ms: dict[str, int],
        default_time: int
    ) -> bool:
        """Add new hotkeys and correct start times for existing CHALLENGE miners."""
        skip_hotkeys = set(self.miner_states.keys()) | set(eliminated_hotkeys)
        state_changed = False

        for hotkey in hotkeys:
            if hotkey in skip_hotkeys:
                continue
            start_time = hk_to_first_order_time_ms.get(hotkey, default_time)
            bucket = MinerBucket.SUBACCOUNT_CHALLENGE if is_synthetic_hotkey(hotkey) else MinerBucket.CHALLENGE
            self.set_miner_bucket(hotkey, bucket, start_time)
            bt.logging.info(f"Adding {hotkey} to challenge period with start time {start_time}")
            state_changed = True

        for hotkey in self.get_hotkeys_by_bucket(MinerBucket.CHALLENGE):
            first_order_time_ms = hk_to_first_order_time_ms.get(hotkey)
            if not first_order_time_ms:
                continue
            start_time_ms = self.miner_states[hotkey].current_bucket_start_ms
            if start_time_ms != first_order_time_ms:
                bt.logging.info(f"Challengeperiod start time for {hotkey} updated from: {datetime.fromtimestamp(start_time_ms/1000)} "
                                f"to: {datetime.fromtimestamp(first_order_time_ms/1000)}, {(start_time_ms-first_order_time_ms)/1000}s delta")
                self.set_miner_bucket(hotkey, MinerBucket.CHALLENGE, first_order_time_ms, replace_top=True)
                state_changed = True

        return state_changed

    # ==================== External+Internal Getter/Setter Methods ====================

    def set_miner_bucket(
        self,
        hotkey: str,
        bucket: MinerBucket,
        start_time: int,
        *,
        replace_top=False
    ) -> bool:
        """
        Set or update a miner's bucket information.

        Prepends a new BucketEntry to the history on bucket change; updates in-place for
        same-bucket refreshes. The previous bucket are always preserved

        Args:
            hotkey: Miner's hotkey
            bucket: New bucket to assign
            start_time: Start time for new bucket
            replace_bucket: Update newest bucket in place

        Returns:
            True if this is a new miner, False if updating existing
        """
        with self._buckets_lock:
            if not hotkey in self.miner_states:
                self.miner_states[hotkey] = MinerBucketState([BucketEntry(bucket, start_time)])
                return True
            else:
                self.miner_states[hotkey].add_bucket_entry(bucket, start_time, replace_top=replace_top)
                return False

    def get_miner_bucket(self, hotkey, timestamp_ms: Optional[int] = None) -> Optional[MinerBucket]:
        """Get the bucket of a miner, optionally at a specific timestamp."""
        if hotkey not in self.miner_states:
            return None
        return self.miner_states[hotkey].bucket(timestamp_ms)

    def get_miner_start_time(self, hotkey: str) -> Optional[int]:
        """Get the start time of a miner's current bucket."""
        if hotkey not in self.miner_states:
            return None
        return self.miner_states[hotkey].current_bucket_start_ms

    def has_miner(self, hotkey: str) -> bool:
        """Fast check if a miner is in active_miners (O(1))."""
        return hotkey in self.miner_states

    def remove_hotkeys(self, hotkeys: list[str]) -> bool:
        """Remove hotkeys from memory - CALL OUTSIDE OF LOCK"""
        updated = False
        with self._buckets_lock:
            for hotkey in hotkeys:
                if hotkey in self.miner_states:
                    del self.miner_states[hotkey]
                    updated = True

        for hotkey in hotkeys:
            self._miner_account_client.set_miner_bucket(hotkey, None)

        return updated

    def _sync_buckets_to_accounts(self):
        """Push all current miner buckets to MinerAccount on startup."""
        synced = 0
        for hotkey, state in self.miner_states.items():
            try:
                self._miner_account_client.set_miner_bucket(hotkey, state.current_bucket)
                synced += 1
            except Exception as e:
                bt.logging.warning(f"Failed to sync miner_bucket for {hotkey}: {e}")
        bt.logging.info(f"[CP_MANAGER] Synced {synced}/{len(self.miner_states)} miner buckets to MinerAccount")

    def clear_active_miners(self):
        """Clear all miners from active_miners."""
        self.miner_states.clear()

    def get_all_miner_hotkeys(self) -> list:
        """Get list of all active miner hotkeys."""
        return list(self.miner_states.keys())

    def get_miner_scores(self) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Get cached miner scores for MinerStatisticsManager.

        Returns:
            tuple containing:
            - asset_softmaxed_scores: dict[asset_class, dict[hotkey, score]]
            - asset_competitiveness: dict[asset_class, competitiveness_score]
        """
        _cached_asset_softmaxed_scores = {asset.value: data for asset, data in self._cached_asset_softmaxed_scores.items()}
        _cached_asset_competitiveness = {asset.value: data for asset, data in self._cached_asset_competitiveness.items()}
        return _cached_asset_softmaxed_scores, _cached_asset_competitiveness

    def get_dashboard(self, hotkey) -> dict | None:
        """
        returns {
            "bucket": bucket.value, (str)
            "start_time_ms": start_time_ms, (int)
        }
        """
        state = self.miner_states.get(hotkey)
        if not state:
            return None
        return state.current_bucket_entry.to_dict()

    def to_checkpoint_dict(self):
        """Get challenge period data as a checkpoint dict for serialization."""
        json_dict = {}
        for hotkey, state in self.miner_states.items():
            json_dict[hotkey] = state.to_json()
        return json_dict

    def get_drawdown_stats(self, synthetic_hotkey: str) -> Optional[dict]:
        """
        Return drawdown statistics for a synthetic hotkey for dashboard display.

        Values are populated by _evaluate_synthetic_challenge so the dashboard
        always reflects the same state as the evaluation loop — no live recomputation.

        Returns None if the hotkey has not been evaluated yet.
        """
        state = self.miner_states.get(synthetic_hotkey)
        if not state:
            return None

        intraday_threshold = state.intraday_drawdown_threshold
        eod_threshold = state.eod_drawdown_threshold
        return {
            **state.drawdown.to_dict(),
            # TODO remove fields someday...
            "intraday_drawdown_threshold": intraday_threshold,
            "eod_drawdown_threshold": eod_threshold,
            "subaccount_challenge_intraday_drawdown_threshold": intraday_threshold,
            "subaccount_challenge_eod_drawdown_threshold": eod_threshold,
        }

    def _read_qwer_from_disk(self) -> dict[str, MinerBucketState]:
        # Load initial active_miners from disk
        qwer_buckets = {}
        if not self.is_backtesting:
            disk_data = ValiUtils.get_vali_json_file_dict(self.CHALLENGE_FILE)
            qwer_buckets = self.parse_checkpoint_dict(disk_data)

        return qwer_buckets

    @staticmethod
    def parse_checkpoint_dict(json_dict) -> dict[str, MinerBucketState]:
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
                formatted_dict[hotkey] = MinerBucketState([BucketEntry(MinerBucket.CHALLENGE, start_time)])
            for hotkey, start_time in success.items():
                formatted_dict[hotkey] = MinerBucketState([BucketEntry(MinerBucket.MAINCOMP, start_time)])
        else:
            for hotkey, info in json_dict.items():
                if isinstance(info, list):
                    # New list format
                    formatted_dict[hotkey] = MinerBucketState([
                        BucketEntry(
                            bucket=MinerBucket(entry["bucket"]),
                            start_time_ms=entry["bucket_start_time"]
                        )
                        for entry in info
                    ])
                elif isinstance(info, dict):
                    entries = []
                    # Current dict format
                    bucket = MinerBucket(info["bucket"]) if info.get("bucket") else None
                    bucket_start_time = info.get("bucket_start_time")
                    if bucket and bucket_start_time:
                        entries.append(BucketEntry(bucket, bucket_start_time))

                    previous_bucket = MinerBucket(info["previous_bucket"]) if info.get("previous_bucket") else None
                    previous_bucket_start_time = info.get("previous_bucket_start_time")
                    if previous_bucket is not None and previous_bucket_start_time is not None:
                        entries.append(BucketEntry(previous_bucket, previous_bucket_start_time))

                    formatted_dict[hotkey] = MinerBucketState(entries)

        return formatted_dict
