# developer: trdougherty, jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ChallengePeriodManager - Core business logic for challenge period management.

This manager handles all heavy logic for challenge period operations.
ChallengePeriodServer wraps this and exposes methods via RPC.

This follows the same pattern as EliminationManager.
"""
from dataclasses import asdict, dataclass, field, fields
from typing import Tuple

from shared_objects.log import logger
import threading
from datetime import datetime

from vali_objects.enums.order_source_enum import OrderSource
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.limit_order.limit_order_client import LimitOrderClient
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, RPCConnectionMode
from vali_objects.enums.miner_asset_class_enum import MinerAssetClass
from vali_objects.utils.asset_selection.asset_selection_client import AssetSelectionClient
from shared_objects.cache_controller import CacheController
from vali_objects.scoring.scoring import Scoring
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
from vali_objects.vali_dataclasses.position import Position
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.drawdown_criteria_enum import DrawdownCriteria
from vali_objects.enums.miner_bucket_enum import BucketEntry, MinerBucket
from vali_objects.plagiarism.plagiarism_client import PlagiarismClient
from vali_objects.miner_account.miner_account_client import MinerAccountClient
from vali_objects.miner_account.miner_account_manager import MinerAccount
from shared_objects.rpc.common_data_client import CommonDataClient
from entity_management.entity_utils import is_synthetic_hotkey
from entity_management.entity_client import EntityClient


@dataclass
class DrawdownStats:
    current_equity: float = 1.0
    current_balance: float = 1.0
    daily_open_equity: float | None = 1.0
    eod_hwm: float = 1.0
    last_eod_equity: float = 1.0
    last_eod_checked_ms: int | None = None

    @property
    def intraday_drawdown_pct(self) -> float:
        if not self.daily_open_equity:
            return 0.0
        return (1.0 - self.current_equity / self.daily_open_equity) * 100.0

    @property
    def eod_drawdown_pct(self) -> float:
        return (1.0 - self.last_eod_equity / self.eod_hwm) * 100.0

    @property
    def static_drawdown_pct(self) -> float:
        return (1.0 - self.current_balance) * 100.0

    @property
    def static_eod_drawdown_pct(self) -> float:
        return (1.0 - self.last_eod_equity) * 100.0

    @property
    def current_return(self) -> float:
        return min(self.current_equity, self.current_balance) - 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d['intraday_drawdown_pct'] = self.intraday_drawdown_pct
        d['eod_drawdown_pct'] = self.eod_drawdown_pct
        d['static_drawdown_pct'] = self.static_drawdown_pct
        d['static_eod_drawdown_pct'] = self.static_eod_drawdown_pct
        d['current_return'] = self.current_return
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> 'DrawdownStats':
        if not d:
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys and v is not None})


@dataclass
class MinerBucketState:
    hotkey: str
    entries: list[BucketEntry]
    drawdown: DrawdownStats = field(default_factory=DrawdownStats)
    drawdown_criteria: DrawdownCriteria = DrawdownCriteria.TRAILING
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
            return True

    def pop_bucket_entry(self, top_bucket: MinerBucket) -> BucketEntry | None:
        if self.current_bucket == top_bucket:
            return self.entries.pop()
        return None

    def bucket(self, time_ms: int | None = None) -> MinerBucket:
        if time_ms is None:
            return self.entries[-1].bucket
        for entry in reversed(self.entries):
            if entry.start_time_ms <= time_ms:
                return entry.bucket
        return MinerBucket.UNKNOWN

    def to_json(self) -> list:
        """Only sync bucket entries - drawdown/rank should not be synced across validators."""
        return [entry.to_dict() for entry in self.entries]

    def to_checkpoint_dict(self) -> dict:
        """Serialize full state (entries + drawdown + drawdown_criteria + rank) for on-disk checkpoint."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "drawdown": self.drawdown.to_dict(),
            "drawdown_criteria": self.drawdown_criteria.value,
            "rank": self.rank,
        }

    @classmethod
    def from_checkpoint_dict(cls, hotkey: str, data: dict) -> 'MinerBucketState':
        entries = [BucketEntry.from_dict(entry) for entry in data.get("entries", [])]
        criteria_str = data.get("drawdown_criteria")
        criteria = DrawdownCriteria(criteria_str) if criteria_str else DrawdownCriteria.TRAILING
        return cls(
            hotkey=hotkey,
            entries=entries,
            drawdown=DrawdownStats.from_dict(data.get("drawdown")),
            drawdown_criteria=criteria,
            rank=data.get("rank"),
        )

    def __str__(self) -> str:
        from datetime import datetime
        start = datetime.fromtimestamp(self.current_bucket_start_ms / 1000).strftime("%Y-%m-%d %H:%M")
        dd = self.drawdown
        criteria_str = self.drawdown_criteria.value
        return (
            f"{self.hotkey} {self.current_bucket.value} {start} rank={self.rank} criteria={criteria_str} "
            f"equity={dd.current_equity:.4f} balance={dd.current_balance:.4f} daily_open={f'{dd.daily_open_equity:.4f}' if dd.daily_open_equity is not None else 'None'} | "
            f"intraday_dd={dd.intraday_drawdown_pct:.2f}% eod_dd={dd.eod_drawdown_pct:.2f}% "
            f"static_dd={dd.static_drawdown_pct:.2f}% static_eod_dd={dd.static_eod_drawdown_pct:.2f}% "
            f"eod_hwm={dd.eod_hwm:.4f}"
        )

    @property
    def is_eliminated(self):
        return self.current_bucket == MinerBucket.ELIMINATED

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
    def _threshold_time_ms(self) -> int | None:
        """For SUBACCOUNT_FUNDED, returns the SUBACCOUNT_CHALLENGE entry's start time
        so versioned thresholds (V0/V1) are selected based on when the miner registered,
        not when they were promoted. For all other buckets, returns current bucket start."""
        if self.current_bucket == MinerBucket.SUBACCOUNT_FUNDED:
            challenge_entry = next((e for e in self.entries if e.bucket == MinerBucket.SUBACCOUNT_CHALLENGE), None)
            return challenge_entry.start_time_ms if challenge_entry else None
        return self.current_bucket_start_ms

    @property
    def intraday_drawdown_threshold(self):
        if self.current_bucket == MinerBucket.ELIMINATED:
            if len(self.entries) >= 2:
                prev_entry = self.entries[-2]
                if prev_entry.bucket == MinerBucket.SUBACCOUNT_FUNDED:
                    challenge_entry = next((e for e in self.entries if e.bucket == MinerBucket.SUBACCOUNT_CHALLENGE), None)
                    prev_threshold_time_ms = challenge_entry.start_time_ms if challenge_entry else None
                else:
                    prev_threshold_time_ms = prev_entry.start_time_ms
                return prev_entry.bucket.intraday_drawdown_threshold(prev_threshold_time_ms)
            else:
                # Dummy threshold value for eliminated accounts to not raise Error
                return MinerBucket.SUBACCOUNT_CHALLENGE.intraday_drawdown_threshold()
        return self.current_bucket.intraday_drawdown_threshold(self._threshold_time_ms)

    @property
    def intraday_drawdown_threshold_pct(self):
        return self.intraday_drawdown_threshold * 100

    @property
    def eod_drawdown_threshold(self):
        if self.current_bucket == MinerBucket.ELIMINATED:
            if len(self.entries) >= 2:
                prev_entry = self.entries[-2]
                if prev_entry.bucket == MinerBucket.SUBACCOUNT_FUNDED:
                    challenge_entry = next((e for e in self.entries if e.bucket == MinerBucket.SUBACCOUNT_CHALLENGE), None)
                    prev_threshold_time_ms = challenge_entry.start_time_ms if challenge_entry else None
                else:
                    prev_threshold_time_ms = prev_entry.start_time_ms
                return prev_entry.bucket.eod_drawdown_threshold(prev_threshold_time_ms)
            else:
                # Dummy threshold value for eliminated accounts to not raise Error
                return MinerBucket.SUBACCOUNT_CHALLENGE.eod_drawdown_threshold()
        return self.current_bucket.eod_drawdown_threshold(self._threshold_time_ms)

    @property
    def eod_drawdown_threshold_pct(self):
        return self.eod_drawdown_threshold * 100



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
    DRAWDOWN_ACTIVATION_MS = TimeUtil.formatted_date_str_to_millis("2026-09-05 00:00:00")   # TODO: remove

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
        self._limit_order_client = LimitOrderClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._elimination_client = EliminationClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._plagiarism_client = PlagiarismClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._miner_account_client = MinerAccountClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._common_data_client = CommonDataClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._asset_selection_client = AssetSelectionClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._debt_ledger_client = DebtLedgerClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)
        self._entity_client = EntityClient(connection_mode=connection_mode, running_unit_tests=running_unit_tests)

        self.CHALLENGE_FILE = ValiBkpUtils.get_challengeperiod_file_location(running_unit_tests=running_unit_tests)
        self._current_iteration_epoch = None

        self._buckets_lock = threading.Lock()
        self.miner_states: dict[str, MinerBucketState] = self._read_states_from_disk()

        # Cached scores for MinerStatisticsManager
        self._cached_asset_softmaxed_scores: dict[MinerAssetClass, dict[str, float]] = {}
        self._cached_asset_competitiveness: dict[MinerAssetClass, float] = {}

        logger.info("[CP_MANAGER] ChallengePeriodManager initialized with {len(self.miner_states)} state data")

    # ==================== Core Business Logic ====================

    def refresh(self, current_time_ms: int | None = None, iteration_epoch: int | None = None):
        if current_time_ms is None:
            current_time_ms = TimeUtil.now_in_millis()

        logger.info(f"[CHALLENGE] Starting challenge period loop {current_time_ms} iteration_epoch={iteration_epoch}")

        asset_selections = self._asset_selection_client.get_asset_selections()
        all_hotkeys = self._position_client.get_all_hotkeys()
        filtered_positions, hk_to_first_order_time = self._position_client.filtered_positions_for_scoring(
            hotkeys=all_hotkeys
        )
        hotkeys_elimination_sync = list(self._elimination_client.get_eliminated_hotkeys())
        hotkeys_plagiarism_sync = list(self._plagiarism_client.get_plagiarism_miners())

        state_changed = False
        state_changed |= self._sync_positions(
            hotkeys=list(filtered_positions.keys()),
            eliminated_hotkeys=hotkeys_elimination_sync,
            hk_to_first_order_time_ms=hk_to_first_order_time,
            default_time=current_time_ms,
        )
        state_changed |= self.sync_plagiarism_miners(hotkeys_plagiarism_sync, current_time_ms)
        state_changed |= self.sync_elimination_miners(hotkeys_elimination_sync, current_time_ms)
        state_changed |= self._prune_hotkeys_no_positions()

        self._current_iteration_epoch = iteration_epoch

        evaluation_hotkeys = [hotkey for hotkey, state in self.miner_states.items() if state.current_bucket.is_active]
        rank_hotkeys = [hotkey for hotkey, state in self.miner_states.items() if state.current_bucket.is_rank_based]

        accounts = self._miner_account_client.get_accounts(evaluation_hotkeys)
        ledgers = self._perf_ledger_client.filtered_ledger_for_scoring(evaluation_hotkeys)
        positions = self._position_client.get_positions_for_hotkeys(evaluation_hotkeys)
        self._refresh_drawdown_cache(evaluation_hotkeys, accounts, ledgers, positions, current_time_ms)
        self._refresh_rank_cache(rank_hotkeys, ledgers, filtered_positions, accounts, asset_selections, current_time_ms)

        eliminations = {}
        promotions, demotions = [], []
        for hotkey, state in self.miner_states.items():
            if hotkey not in evaluation_hotkeys:
                if state.current_bucket == MinerBucket.UNKNOWN:
                    logger.warning(f"[CHALLENGE] {hotkey} bucket unknown")
                continue

            if reason := self._check_time(state, current_time_ms):
                eliminations[hotkey] = reason
                continue

            if state.drawdown_criteria == DrawdownCriteria.STATIC:
                # Static rules for subaccounts registered after the effective time (Hyperscaled excluded) —
                # both measured against starting balance
                # Rule 1: Static drawdown — balance (excl. unrealized PnL) cannot drop more than 5% below starting balance
                if reason := self._check_static_drawdown(state):
                    eliminations[hotkey] = reason
                    continue

                # Rule 2: Static EOD drawdown — equity at the 00:00 UTC check cannot be more than 5% below starting balance
                if reason := self._check_static_eod_drawdown(state):
                    eliminations[hotkey] = reason
                    continue
            else:
                # Legacy rules: pre-effective subaccounts eliminate immediately; regular miners only after activation
                # Rule 1: Intraday drawdown — current equity cannot drop below from today's opening equity
                if reason := self._check_intraday_drawdown(state):
                    if state.current_bucket.is_subaccount or current_time_ms > self.DRAWDOWN_ACTIVATION_MS:
                        eliminations[hotkey] = reason
                    continue

                # Rule 2: EOD trailing drawdown — last EOD equity cannot drop below threshold from highest-ever EOD equity
                if reason := self._check_eod_drawdown(state):
                    if state.current_bucket.is_subaccount or current_time_ms > self.DRAWDOWN_ACTIVATION_MS:
                        eliminations[hotkey] = reason
                    continue

            _asset = asset_selections.get(hotkey)
            if _asset is None:
                logger.warning(f"[CHALLENGE] {hotkey} no asset selection, skipping evaluation")
                continue
            asset_class = MinerAssetClass(_asset)

            if self._check_demotion(state):
                demotions.append(hotkey)
                continue

            returns_threshold = ValiConfig.SUBACCOUNT_CHALLENGE_RETURNS_THRESHOLD[asset_class]
            if hotkey == "5EPeU7Y8bqokEVf31ZWPZkP3F7Kv1v3ALuhnpp5T5Fvfjp85_87": # remove once eliminated or promoted
                returns_threshold = 0.08
            if self._check_promotion(state, returns_threshold, current_time_ms):
                promotions.append(hotkey)
                continue

        state_changed |= self.eliminate_hotkeys(eliminations, current_time_ms)
        state_changed |= self.demote_hotkeys(demotions, current_time_ms)
        state_changed |= self.promote_hotkeys(promotions, current_time_ms)

        if state_changed:
            self._sync_buckets_to_accounts(hotkeys=list({*eliminations, *demotions, *promotions}))

        self._save_to_disk()

        counts = {}
        for state in self.miner_states.values():
            counts[state.current_bucket] = counts.get(state.current_bucket, 0) + 1
        snapshot = " | ".join(f"{b.value}={n}" for b, n in sorted(counts.items(), key=lambda x: x[0].value))
        logger.info(f"[CHALLENGE] snapshot: {snapshot} (total={len(self.miner_states)})")

        return self._to_slack_message(promotions)

    # ==================== Evaluation Methods ====================

    @staticmethod
    def _check_time(state: MinerBucketState, current_time_ms: int) -> EliminationReason | None:
        bucket = state.current_bucket_entry.bucket
        if bucket.max_time_ms is None:
            return None
        if bucket == MinerBucket.CHALLENGE:
            first_challenge_entry = next((e for e in state.entries if e.bucket == MinerBucket.CHALLENGE), state.current_bucket_entry)
            bucket_start_ms = first_challenge_entry.start_time_ms
        else:
            bucket_start_ms = state.current_bucket_entry.start_time_ms
        if current_time_ms - bucket_start_ms > bucket.max_time_ms:
            reason_map = {
                MinerBucket.CHALLENGE: EliminationReason.FAILED_CHALLENGE_PERIOD_TIME,
                MinerBucket.PROBATION: EliminationReason.FAILED_PROBATION_TIME,
                MinerBucket.PLAGIARISM: EliminationReason.PLAGIARISM,
            }
            reason = reason_map.get(bucket)
            if reason:
                logger.warning(f"[CHALLENGE] ELIMINATION time limit expired: {state}")
            return reason
        return None

    @staticmethod
    def _check_intraday_drawdown(state: MinerBucketState) -> EliminationReason | None:
        if state.drawdown.intraday_drawdown_pct > state.intraday_drawdown_threshold_pct:
            logger.warning(f"[CHALLENGE] ELIMINATION intraday drawdown {state.intraday_drawdown_threshold_pct}%: {state}")
            if state.current_bucket in (MinerBucket.CHALLENGE, MinerBucket.SUBACCOUNT_CHALLENGE):
                return EliminationReason.FAILED_CHALLENGE_PERIOD_INTRADAY_DRAWDOWN
            else:
                return EliminationReason.FAILED_FUNDED_PERIOD_INTRADAY_DRAWDOWN
        elif state.drawdown.intraday_drawdown_pct > state.intraday_drawdown_threshold_pct * 0.75:
            logger.info(f"[CHALLENGE] near intraday drawdown {state.intraday_drawdown_threshold_pct}%: {state}")
        return None

    @staticmethod
    def _check_eod_drawdown(state: MinerBucketState) -> EliminationReason | None:
        if state.drawdown.eod_drawdown_pct > state.eod_drawdown_threshold_pct:
            logger.warning(f"[CHALLENGE] ELIMINATION EOD drawdown {state.eod_drawdown_threshold_pct}%: {state}")
            if state.current_bucket in (MinerBucket.CHALLENGE, MinerBucket.SUBACCOUNT_CHALLENGE):
                return EliminationReason.FAILED_CHALLENGE_PERIOD_EOD_DRAWDOWN
            else:
                return EliminationReason.FAILED_FUNDED_PERIOD_EOD_DRAWDOWN

        if 1 - state.drawdown.current_equity / state.drawdown.eod_hwm > state.eod_drawdown_threshold:
            logger.info(f"[CHALLENGE] near trailing EOD with current equity {state.eod_drawdown_threshold_pct}%: {state}")

        return None

    @staticmethod
    def _check_static_drawdown(state: MinerBucketState) -> EliminationReason | None:
        threshold_pct = ValiConfig.SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD * 100
        if state.drawdown.static_drawdown_pct > threshold_pct:
            logger.warning(f"[CHALLENGE] ELIMINATION static drawdown {threshold_pct}%: {state}")
            if state.current_bucket == MinerBucket.SUBACCOUNT_CHALLENGE:
                return EliminationReason.FAILED_CHALLENGE_PERIOD_STATIC_DRAWDOWN
            else:
                return EliminationReason.FAILED_FUNDED_PERIOD_STATIC_DRAWDOWN
        elif state.drawdown.static_drawdown_pct > threshold_pct * 0.75:
            logger.info(f"[CHALLENGE] near static drawdown {threshold_pct}%: {state}")
        return None

    @staticmethod
    def _check_static_eod_drawdown(state: MinerBucketState) -> EliminationReason | None:
        threshold_pct = ValiConfig.SUBACCOUNT_STATIC_EOD_DRAWDOWN_THRESHOLD * 100
        if state.drawdown.static_eod_drawdown_pct > threshold_pct:
            logger.warning(f"[CHALLENGE] ELIMINATION static EOD drawdown {threshold_pct}%: {state}")
            if state.current_bucket == MinerBucket.SUBACCOUNT_CHALLENGE:
                return EliminationReason.FAILED_CHALLENGE_PERIOD_STATIC_EOD_DRAWDOWN
            else:
                return EliminationReason.FAILED_FUNDED_PERIOD_STATIC_EOD_DRAWDOWN

        if (1.0 - state.drawdown.current_equity) * 100.0 > threshold_pct:
            logger.info(f"[CHALLENGE] near static EOD with current equity {threshold_pct}%: {state}")

        return None

    @staticmethod
    def _check_demotion(state: MinerBucketState) -> bool:
        if state.current_bucket == MinerBucket.MAINCOMP:
            result = (state.rank > ValiConfig.PROMOTION_THRESHOLD_RANK
                      if state.rank is not None
                      else False)
            if result:
                logger.info(f"[CHALLENGE] DEMOTION {state}")
            return result
        return False

    @staticmethod
    def _check_promotion(state: MinerBucketState, returns_threshold: float, current_time_ms: int) -> bool:
        if state.current_bucket == MinerBucket.CHALLENGE:
            if current_time_ms - state.current_bucket_start_ms < ValiConfig.CHALLENGE_PERIOD_MINIMUM_MS:
                return False

        if state.current_bucket.next_bucket is None:
            return False

        if state.drawdown.current_return > returns_threshold:
            if state.current_bucket.is_rank_based:
                result = state.rank <= ValiConfig.PROMOTION_THRESHOLD_RANK if state.rank else False
            else:
                result = True
            if result:
                logger.info(f"[CHALLENGE] PROMOTION {state}")
            return result
        elif state.drawdown.current_return > returns_threshold * 0.75:
            logger.info(f"[CHALLENGE] near promotion: {state}")

        return False

    def eliminate_hotkeys(self, eliminations: dict[str, EliminationReason], current_time_ms: int):
        if eliminations:
            logger.info(f"[CHALLENGE] eliminating {len(eliminations)} miners {list(eliminations.keys())}")

        for hotkey, elimination_reason in eliminations.items():
            state = self.miner_states[hotkey]
            if not state.current_bucket.is_active:
                logger.warning(f"[CHALLENGE] attempted elimination in non-eligible bucket: {state}")

            logger.info(f"[CHALLENGE] eliminating reason={elimination_reason.value}: {state}")

            elimination_time_ms = current_time_ms
            if elimination_reason.is_eod_drawdown:
                elimination_drawdown_pct = state.drawdown.eod_drawdown_pct
                elimination_time_ms = state.drawdown.last_eod_checked_ms or current_time_ms
            elif elimination_reason.is_intraday_drawdown:
                elimination_drawdown_pct = state.drawdown.intraday_drawdown_pct
            elif elimination_reason.is_static_drawdown:
                elimination_drawdown_pct = state.drawdown.static_drawdown_pct
            elif elimination_reason.is_static_eod_drawdown:
                elimination_drawdown_pct = state.drawdown.static_eod_drawdown_pct
                elimination_time_ms = state.drawdown.last_eod_checked_ms or current_time_ms
            else:
                elimination_drawdown_pct = max(state.drawdown.intraday_drawdown_pct, state.drawdown.eod_drawdown_pct)

            is_static = elimination_reason.is_static_drawdown or elimination_reason.is_static_eod_drawdown
            intraday_drawdown_pct = state.drawdown.static_drawdown_pct if is_static else state.drawdown.intraday_drawdown_pct
            eod_drawdown_pct = state.drawdown.static_eod_drawdown_pct if is_static else state.drawdown.eod_drawdown_pct

            self._elimination_client.append_elimination_row(
                hotkey=hotkey,
                reason=elimination_reason,
                elimination_drawdown_pct=elimination_drawdown_pct,
                intraday_drawdown_pct=intraday_drawdown_pct,
                eod_drawdown_pct=eod_drawdown_pct,
                elimination_time_ms=elimination_time_ms,
                bucket_at_elimination=state.current_bucket,
            )

        state_changed = False
        with self._buckets_lock:
            for hotkey in eliminations:
                state_changed |= self.miner_states[hotkey].add_bucket_entry(MinerBucket.ELIMINATED, current_time_ms)

        return state_changed

    def demote_hotkeys(self, hotkeys: list[str], current_time_ms) -> bool:
        """Demote miners to probation."""
        if hotkeys:
            logger.info(f"[CHALLENGE] demoting {len(hotkeys)} miners to probation")

        state_changed = False
        with self._buckets_lock:
            for hotkey in hotkeys:
                logger.info(f"[CHALLENGE] demoting: {self.miner_states[hotkey]}")
                state_changed |= self.miner_states[hotkey].add_bucket_entry(MinerBucket.PROBATION, current_time_ms)
        return state_changed

    def promote_hotkeys(self, hotkeys: list[str], current_time_ms: int) -> bool:
        """Promote miners to next tier."""
        if hotkeys:
            logger.info(f"[CHALLENGE] promoting {len(hotkeys)} miners")

        state_changed = False
        for hotkey in hotkeys:
            state = self.miner_states[hotkey]
            current_bucket = state.current_bucket
            target_bucket = current_bucket.next_bucket
            if target_bucket is None:
                logger.warning(f"[CHALLENGE] no next bucket for promotion: {state}")
                continue

            logger.info(f"[CHALLENGE] promoting to {target_bucket.value}: {state}")

            if target_bucket == MinerBucket.SUBACCOUNT_FUNDED:
                # Close all existing positions
                self._position_client.close_all_positions(
                    hotkey=hotkey,
                    close_time_ms=current_time_ms,
                    order_source=OrderSource.SUBACCOUNT_PROMOTION
                )
                # Reset account fields (PnL, capital used, borrowed amount, interest)
                self._miner_account_client.reset_account(hotkey, target_bucket)
                # Archive all positions (disk move + memory removal)
                self._position_client.archive_positions_for_hotkey(hotkey, archive_all=True)
                # Cancel all pending limit orders
                self._limit_order_client.cancel_limit_order(hotkey, None, "ALL", current_time_ms)
                # Wipe perf ledgers so funded-period performance is tracked from scratch
                self._perf_ledger_client.wipe_miners_perf_ledgers([hotkey])
                # Delete debt ledger to match new perf ledger checkpoints
                self._debt_ledger_client.delete_debt_ledger(hotkey)
                # Reset drawdown cache
                self._reset_drawdown_stats_cache(hotkey)

            with self._buckets_lock:
                state_changed |= self.miner_states[hotkey].add_bucket_entry(target_bucket, current_time_ms)

        return state_changed

    def revert_elimination(self, hotkey: str) -> bool:
        miner_state = self.miner_states.get(hotkey)
        if not miner_state:
            return False

        with self._buckets_lock:
            if miner_state.current_bucket != MinerBucket.ELIMINATED:
                return False

            if len(miner_state.entries) < 2:
                return False

            prev_bucket = miner_state.entries[-2].bucket

            miner_state.add_bucket_entry(prev_bucket, miner_state.entries[-2].start_time_ms)

        self._reset_drawdown_stats_cache(hotkey)
        self._sync_buckets_to_accounts(hotkeys=[hotkey])
        self._save_to_disk()
        return True

    # ==================== Drawdown/Rank Refresh methods ====================

    @staticmethod
    def _parse_eod_checkpoints(ledger: PerfLedger, now_ms: int) -> tuple[float, float | None, float, int | None]:
        """
        Parse midnight checkpoints from a ledger.
        Returns (last_eod, daily_open_equity, eod_hwm, last_eod_checked_ms).
        """
        midnight_cps = [cp for cp in ledger.cps if cp.last_update_ms % 86400000 == 0 and cp.equity_ret > 0]
        last_eod = midnight_cps[-1].equity_ret if midnight_cps else 1.0
        last_eod_checked_ms = midnight_cps[-1].last_update_ms if midnight_cps else None
        today_midnight_ms = (now_ms // 86400000) * 86400000
        today_open_cp = next((cp for cp in midnight_cps if cp.last_update_ms == today_midnight_ms), None)
        daily_open_equity = today_open_cp.equity_ret if today_open_cp else (None if last_eod else 1.0)
        eod_hwm = max(max(cp.equity_ret for cp in midnight_cps), 1.0) if midnight_cps else 1.0
        return last_eod, daily_open_equity, eod_hwm, last_eod_checked_ms

    def _refresh_drawdown_cache(
        self,
        hotkeys: list[str],
        accounts: dict[str, MinerAccount],
        ledgers: dict[str, PerfLedger],
        positions: dict[str, list[Position]],
        current_time_ms: int
    ) -> None:
        for hotkey in hotkeys:
            # Compute portfolio return: (balance + unrealized_pnl) / account_size
            account = accounts.get(hotkey)
            if not account or account.account_size <= 0:
                logger.warning(f"[CHALLENGE] {hotkey} has invalid account, skipping evaluation")
                continue

            current_equity = account.equity / account.account_size
            current_balance = account.balance / account.account_size
            if current_equity is None or current_balance is None:
                logger.error(f"[CHALLENGE] {hotkey} invalid account, skipping evaluation")
                continue

            if not positions.get(hotkey):
                # skip log: no positions mean new account or recently promoted, no ledger
                continue

            now_ms = current_time_ms if current_time_ms is not None else TimeUtil.now_in_millis()
            current_day_open_ms = TimeUtil.get_start_of_day_ms(now_ms)
            existing = self.miner_states[hotkey].drawdown

            # EOD fields are locked in for today (snapshot already captured); only update live equity
            if existing.last_eod_checked_ms == current_day_open_ms:
                existing.current_equity = current_equity
                existing.current_balance = current_balance
                continue

            # Use daily open snapshot from miner account as default and fallback to perf ledgers
            snapshot = account.daily_open_snapshot
            if snapshot and TimeUtil.get_start_of_day_ms(snapshot.snapshot_ms) == current_day_open_ms:
                last_eod_equity = daily_open_equity = snapshot.equity_return
                eod_hwm = max(existing.eod_hwm, daily_open_equity)
                last_eod_checked_ms = current_day_open_ms
            else:
                ledger = ledgers.get(hotkey)
                if ledger is None:
                    logger.warning(f"[CHALLENGE] {hotkey} missing ledger, skipping drawdown cache")
                    continue
                last_eod_equity, daily_open_equity, eod_hwm, last_eod_checked_ms = self._parse_eod_checkpoints(ledger, now_ms)

            if last_eod_checked_ms == current_day_open_ms:
                existing.last_eod_equity = last_eod_equity
                existing.daily_open_equity = daily_open_equity
                existing.eod_hwm = max(existing.eod_hwm, last_eod_equity, eod_hwm)
                existing.last_eod_checked_ms = last_eod_checked_ms

    def _refresh_rank_cache(
        self,
        hotkeys: list[str],
        ledgers: dict[str,PerfLedger],
        positions: dict[str,list[Position]],
        accounts: dict[str, MinerAccount],
        asset_selections: dict[str,MinerAssetClass],
        current_time_ms: int
    ):
        asset_classes = list(set(asset_selections.values()))
        asset_class_min_days = LedgerUtils.calculate_dynamic_minimum_days_for_asset_classes(ledgers, asset_classes)

        rank_ledgers = {hk: ledger for hk, ledger in ledgers.items() if hk in hotkeys}
        rank_positions = {hk: pos for hk, pos in positions.items() if hk in hotkeys}
        account_sizes = {hk: account.account_size for hk, account in accounts.items() if hk in hotkeys}

        # Score all rank-eligible miners (including those without minimum days) for accurate threshold
        asset_competitiveness, asset_softmaxed_scores = Scoring.score_miner_asset_classes(
            ledger_dict=rank_ledgers,
            positions=rank_positions,
            asset_class_min_days=asset_class_min_days,
            evaluation_time_ms=current_time_ms,
            weighting=True,
            all_miner_account_sizes=account_sizes
        )

        # Cache scores for MinerStatisticsManager, filtered to only hotkeys that selected each asset class
        self._cached_asset_softmaxed_scores = {
            asset_class: {
                hotkey: score for hotkey, score in asset_scores.items()
                if asset_selections.get(hotkey) == asset_class
            }
            for asset_class, asset_scores in asset_softmaxed_scores.items()
        }
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

    def sync_challenge_period_data(self, miner_states_data: dict):
        """Sync challenge period data from another validator."""
        if not miner_states_data:
            logger.error('challenge_period_data appears invalid')

        logger.info("syncing challenge period data")
        with self._buckets_lock:
            self.miner_states.clear()
            self.miner_states.update(self.parse_checkpoint_dict(miner_states_data))
            self._save_to_disk()

    def sync_plagiarism_miners(self, plagiarism_miners: list[str], current_time_ms: int) -> bool:
        """Sync plagiarism miners status from plagiarism api."""
        with self._buckets_lock:
            state_changed = False
            for hotkey in plagiarism_miners:
                if hotkey not in self.miner_states:
                    continue
                state_changed |= self.miner_states[hotkey].add_bucket_entry(MinerBucket.PLAGIARISM, current_time_ms)

            whitelisted_miners = set(self.get_hotkeys_by_bucket(MinerBucket.PLAGIARISM)) - set(plagiarism_miners)
            for hotkey in whitelisted_miners:
                if self.miner_states[hotkey].current_bucket != MinerBucket.PLAGIARISM:
                    continue
                popped_bucket_entry = self.miner_states[hotkey].pop_bucket_entry(top_bucket=MinerBucket.PLAGIARISM)
                state_changed |= popped_bucket_entry is not None

        return state_changed

    def sync_elimination_miners(self, eliminated_hotkeys: list[str], current_time_ms: int) -> bool:
        """Sync eliminated miners from elimination manager. Method for peace of mind."""
        new_eliminations = [
            hk for hk in eliminated_hotkeys
            if hk in self.miner_states and not self.miner_states[hk].is_eliminated
        ]
        if not new_eliminations:
            return False

        logger.warning(f"[CHALLENGE] syncing {len(new_eliminations)} eliminated miners: {new_eliminations}")
        with self._buckets_lock:
            for hotkey in new_eliminations:
                self.miner_states[hotkey].add_bucket_entry(MinerBucket.ELIMINATED, current_time_ms)

        return True

    def _prune_hotkeys_no_positions(self, hotkeys=None) -> bool:
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

        hotkeys_prune = []
        for hotkey, state in self.miner_states.items():
            if hotkey not in hotkeys:
                bucket = state.current_bucket
                if bucket in [MinerBucket.ENTITY, MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED, MinerBucket.ELIMINATED]:
                    continue
                hotkeys_prune.append(hotkey)
                logger.warning(f"[CHALLENGE] {hotkey} pruned from {bucket.value}: no longer in metagraph")

        self.remove_miners(hotkeys_prune)
        return bool(hotkeys_prune)

    def _sync_positions(
        self,
        hotkeys: list[str],
        eliminated_hotkeys: list[str],
        hk_to_first_order_time_ms: dict[str, int],
        default_time: int,
    ) -> bool:
        """Add new hotkeys and correct start times for existing CHALLENGE miners."""
        skip_hotkeys = set(self.miner_states.keys()) | set(eliminated_hotkeys)
        state_changed = False

        for hotkey in hotkeys:
            if hotkey in skip_hotkeys:
                continue
            start_time = hk_to_first_order_time_ms.get(hotkey, default_time)
            bucket = MinerBucket.SUBACCOUNT_CHALLENGE if is_synthetic_hotkey(hotkey) else MinerBucket.CHALLENGE
            self.set_miner_bucket(hotkey, bucket, start_time, drawdown_criteria=DrawdownCriteria.TRAILING)
            logger.info(f"Adding {hotkey} to challenge period with start time {start_time}")
            state_changed = True

        for hotkey in self.get_hotkeys_by_bucket(MinerBucket.CHALLENGE):
            first_order_time_ms = hk_to_first_order_time_ms.get(hotkey)
            if not first_order_time_ms:
                continue
            start_time_ms = self.miner_states[hotkey].current_bucket_start_ms
            if start_time_ms != first_order_time_ms:
                logger.info(f"Challengeperiod start time for {hotkey} updated from: {datetime.fromtimestamp(start_time_ms/1000)} "
                                f"to: {datetime.fromtimestamp(first_order_time_ms/1000)}, {(start_time_ms-first_order_time_ms)/1000}s delta")
                self.set_miner_bucket(hotkey, MinerBucket.CHALLENGE, first_order_time_ms, replace_top=True)
                state_changed = True

        return state_changed

    def _sync_buckets_to_accounts(self, hotkeys: list[str] | None = None):
        """Push current miner buckets to MinerAccount. If hotkeys is None, sync all (startup); otherwise only the given subset."""
        target_hotkeys = list(self.miner_states.keys()) if hotkeys is None else hotkeys
        hotkey_to_bucket: dict[str, MinerBucket] = {}
        for hotkey in target_hotkeys:
            state = self.miner_states.get(hotkey)
            if state is None:
                continue
            hotkey_to_bucket[hotkey] = state.current_bucket
        try:
            self._miner_account_client.set_miner_buckets(hotkey_to_bucket)
        except Exception as e:
            logger.warning(f"Failed to bulk sync miner buckets: {e}")
            return
        logger.info(f"[CP_MANAGER] Synced {len(hotkey_to_bucket)} miner buckets to MinerAccount")

    def set_miner_bucket(
        self,
        hotkey: str,
        bucket: MinerBucket,
        start_time_ms: int,
        *,
        replace_top=False,
        drawdown_criteria: DrawdownCriteria = DrawdownCriteria.TRAILING,
    ) -> bool:
        """
        Set or update a miner's bucket information, replace_top to override most recent entry.
        drawdown_criteria is set only on first creation and ignored for existing states.

        Only persists to disk on first creation - drawdown_criteria is write-once, and updates to
        an existing state are already covered by the caller's own batched save (see refresh()).

        Returns:
            True if this is a new miner, False if updating existing
        """
        with self._buckets_lock:
            if hotkey not in self.miner_states:
                self.miner_states[hotkey] = MinerBucketState(hotkey, [BucketEntry(bucket, start_time_ms)], drawdown_criteria=drawdown_criteria)
                is_new = True
                self._save_to_disk()
            else:
                self.miner_states[hotkey].add_bucket_entry(bucket, start_time_ms, replace_top=replace_top)
                is_new = False
        return is_new

    def update_drawdown_criteria(self, hotkey: str, criteria: DrawdownCriteria) -> Tuple[bool, str]:
        """Update drawdown_criteria for an existing miner state and persist to disk."""
        with self._buckets_lock:
            state = self.miner_states.get(hotkey)
            if not state:
                return False, f"{hotkey} not found in challenge period manager"
            state.drawdown_criteria = criteria
            self._save_to_disk()
        return True, f"drawdown_criteria updated to '{criteria.value}' for {hotkey}"

    def remove_miners(self, hotkeys: str | list[str]) -> bool:
        """Remove hotkeys from memory - CALL OUTSIDE OF LOCK"""
        hotkeys = [hotkeys] if isinstance(hotkeys, str) else hotkeys
        state_changed = False
        with self._buckets_lock:
            for hotkey in hotkeys:
                if hotkey in self.miner_states:
                    logger.info(f"[CHALLENGE] {hotkey} removed from bucket {self.miner_states[hotkey].current_bucket.value}")
                    del self.miner_states[hotkey]
                    state_changed = True

        for hotkey in hotkeys:
            self._miner_account_client.set_miner_bucket(hotkey, None)

        return state_changed

    # ==================== Getter Methods ====================

    def get_hotkeys_by_bucket(self, buckets: MinerBucket | list[MinerBucket]) -> list[str]:
        """Get all hotkeys in bucket or a list of buckets."""
        bucket_set = {buckets} if isinstance(buckets, MinerBucket) else set(buckets)
        return [hotkey for hotkey, state in self.miner_states.items() if state.current_bucket in bucket_set]

    def get_miner_bucket(self, hotkey, timestamp_ms: int | None = None) -> MinerBucket | None:
        """Get the bucket of a miner, optionally at a specific timestamp."""
        if hotkey not in self.miner_states:
            return None
        return self.miner_states[hotkey].bucket(timestamp_ms)

    def get_miner_start_time(self, hotkey: str) -> int | None:
        """Get the start time of a miner's current bucket."""
        if hotkey not in self.miner_states:
            return None
        return self.miner_states[hotkey].current_bucket_start_ms

    def has_miner(self, hotkey: str) -> bool:
        """Fast check if a miner is in active_miners (O(1))."""
        return hotkey in self.miner_states

    def get_all_miner_hotkeys(self) -> list:
        """Get list of all active miner hotkeys."""
        return list(self.miner_states.keys())

    def get_miners(self, buckets: MinerBucket | list[MinerBucket]) -> dict[str, int]:
        """Keep for legacy challengeperiod_client get_testing_miners"""
        bucket_set = {buckets} if isinstance(buckets, MinerBucket) else set(buckets)
        return {hotkey: state.current_bucket_start_ms for hotkey, state in self.miner_states.items() if state.current_bucket in bucket_set}

    def get_miner_scores(self) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        """
        Cached miner scores for MinerStatisticsManager.

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

    def get_drawdown_stats(self, synthetic_hotkey: str) -> dict | None:
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
        criteria = state.drawdown_criteria
        return {
            **state.drawdown.to_dict(),
            "intraday_drawdown_threshold": intraday_threshold,
            "eod_drawdown_threshold": eod_threshold,
            "static_drawdown_threshold": ValiConfig.SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD,
            "static_eod_drawdown_threshold": ValiConfig.SUBACCOUNT_STATIC_EOD_DRAWDOWN_THRESHOLD,
            "drawdown_criteria": criteria.value,
        }

    # ==================== Disk I/O ====================

    def to_checkpoint_dict(self):
        """Get challenge period data as a checkpoint dict for serialization."""
        json_dict = {}
        for hotkey, state in self.miner_states.items():
            json_dict[hotkey] = state.to_checkpoint_dict()
        return json_dict


    def _read_states_from_disk(self) -> dict[str, MinerBucketState]:
        # Load initial active_miners from disk
        miner_states = {}
        if not self.is_backtesting:
            disk_data = ValiUtils.get_vali_json_file_dict(self.CHALLENGE_FILE)
            miner_states = self.parse_checkpoint_dict(disk_data)

        return miner_states

    @staticmethod
    def parse_checkpoint_dict(json_dict) -> dict[str, MinerBucketState]:
        """Parse checkpoint dict from disk or validator sync.
        Expected format: {hk: {"entries": [...], "drawdown": {...}, "drawdown_criteria": ..., "rank": ...}}
        """
        return {
            hotkey: MinerBucketState.from_checkpoint_dict(hotkey, info)
            for hotkey, info in json_dict.items()
            if isinstance(info, dict) and "entries" in info
        }

    def _save_to_disk(self):
        """Write challenge period data from memory to disk."""
        if self.is_backtesting:
            logger.info("don't save checkpoint to disk")
            return

        # Epoch-based validation: check if sync occurred during our iteration
        if self._current_iteration_epoch is not None:
            current_epoch = self._common_data_client.get_sync_epoch()
            if current_epoch != self._current_iteration_epoch:
                logger.warning(
                    f"Sync occurred during ChallengePeriodManager iteration "
                    f"(epoch {self._current_iteration_epoch} -> {current_epoch}). "
                    f"Skipping save to avoid data corruption"
                )
                return

        checkpoint_data = self.to_checkpoint_dict()
        ValiBkpUtils.write_file(self.CHALLENGE_FILE, checkpoint_data)

    def _to_slack_message(self, promotions: list[str]) -> str | None:
        if not promotions:
            return None

        msg = ":moneybag: *Promotions!* :money_mouth_face:\n"
        msg += "\n".join(f"`{hotkey} -> {self.miner_states[hotkey].current_bucket.value}`" for hotkey in promotions)

        return msg
