# developer: trdougherty, jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
ChallengePeriodManager - Core business logic for challenge period management.

This manager handles all heavy logic for challenge period operations.
ChallengePeriodServer wraps this and exposes methods via RPC.

This follows the same pattern as EliminationManager.
"""
from dataclasses import asdict, dataclass, field, fields, replace
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
from vali_objects.utils.metrics import Metrics
from time_util.time_util import StageTimer, TimeUtil
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_client import PerfLedgerClient
from vali_objects.vali_dataclasses.ledger.ledger_utils import LedgerUtils
from vali_objects.vali_dataclasses.ledger.debt.debt_ledger_client import DebtLedgerClient
from vali_objects.vali_dataclasses.position import Position
from vali_objects.enums.elimination_reason_enum import EliminationReason
from vali_objects.enums.drawdown_criteria_enum import DrawdownCriteria
from vali_objects.enums.miner_bucket_enum import BucketEntry, BucketPhase, MinerBucket
from vali_objects.plagiarism.plagiarism_client import PlagiarismClient
from vali_objects.miner_account.miner_account_client import MinerAccountClient
from vali_objects.miner_account.miner_account_manager import MinerAccount
from shared_objects.rpc.common_data_client import CommonDataClient
from entity_management.entity_utils import is_synthetic_hotkey
from entity_management.entity_client import EntityClient
from vali_objects.challenge_period.bucket_history import BucketHistoryStore, BucketSegment
from vali_objects.miner_account.account_snapshot import read_all_snapshots


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
        if not self.eod_hwm:
            return 0.0
        return (1.0 - self.last_eod_equity / self.eod_hwm) * 100.0

    @property
    def trailing_drawdown_pct(self) -> float:
        if not self.eod_hwm:
            return 0.0
        return (1.0 - self.current_equity / self.eod_hwm) * 100.0

    @property
    def static_drawdown_pct(self) -> float:
        return (1.0 - self.current_equity) * 100.0

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
        d['trailing_drawdown_pct'] = self.trailing_drawdown_pct
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
class ProStats:
    calmar: float = 0.0
    daily_consistency: float = 1.0
    max_drawdown: float = 1.0  # Monotonic all-time worst drawdown, in mdd ratio form
    trading_days: int = 0  # Full days of tracked returns on this account
    # Payout week start (Monday 00:00 UTC) -> UTC day starts on which calmar was broken at any
    # refresh. Every week checked in a soft-breach bucket has a key, so an empty list is a clean
    # week. Never pruned: this is the record that decides whether a closed week was withheld.
    soft_breach_weeks: dict[int, list[int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d['soft_breach_weeks'] = {str(week): days for week, days in self.soft_breach_weeks.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict | None) -> 'ProStats':
        if not d:
            return cls()
        valid_keys = {f.name for f in fields(cls)}
        parsed = {k: v for k, v in d.items() if k in valid_keys and v is not None}
        parsed['soft_breach_weeks'] = {int(week): sorted({int(day) for day in days})
                                       for week, days in (parsed.get('soft_breach_weeks') or {}).items()}
        return cls(**parsed)

    def breach_days(self, week_start_ms: int) -> list[int]:
        return self.soft_breach_weeks.get(week_start_ms, [])


@dataclass
class MinerBucketState:
    hotkey: str
    entries: list[BucketEntry]
    drawdown: DrawdownStats = field(default_factory=DrawdownStats)
    drawdown_criteria: DrawdownCriteria = DrawdownCriteria.TRAILING
    # Subaccount's chosen intraday drawdown threshold (fraction). Set once at creation; overrides the
    # bucket's intraday threshold in every bucket (read through the intraday_drawdown_threshold
    # property). None keeps the bucket default.
    intraday_drawdown_threshold_override: float | None = None
    # EOD high-water-mark drawdown threshold chosen at creation (Instant Funded only). Like the
    # intraday override it is ignored in PRO_FUNDED, which always runs the pro defaults.
    eod_hwm_threshold_override: float | None = None
    rank: int | None = None
    pro_stats: ProStats = field(default_factory=ProStats)

    def __post_init__(self):
        if not self.entries:
            raise ValueError("BucketInfo must be initialized with BucketEntries")
        self.entries = sorted(self.entries, key=lambda x: x.start_time_ms)

    def add_bucket_entry(self, bucket: MinerBucket, time_ms: int, *, replace_top: bool = False,
                         in_transition: bool = False) -> bool:
        """Append a bucket entry. A SUBACCOUNT_FUNDED entry whose in_transition differs from the
        current one is a new entry: entering the transition week is recorded as its own span."""
        entry = BucketEntry(bucket, time_ms, in_transition=in_transition)
        if self.current_phase == entry.phase:
            if replace_top:
                self.entries[-1] = entry
                return True
            return False
        else:
            self.entries.append(entry)
            return True

    def pop_bucket_entry(self, top_bucket: MinerBucket) -> BucketEntry | None:
        if self.current_bucket == top_bucket:
            return self.entries.pop()
        return None

    def bucket(self, time_ms: int | None = None) -> MinerBucket:
        """The bucket in force, now or at a point in time."""
        return self.phase(time_ms).bucket

    def phase(self, time_ms: int | None = None) -> BucketPhase:
        """The bucket and transition flag in force, now or at a point in time."""
        if time_ms is None:
            return self.entries[-1].phase
        for entry in reversed(self.entries):
            if entry.is_reverted:
                continue
            if entry.start_time_ms <= time_ms:
                return entry.phase
        return BucketPhase(MinerBucket.UNKNOWN)

    def to_checkpoint_dict(self) -> dict:
        """Serialize full state (entries + drawdown + drawdown_criteria + drawdown threshold overrides +
        rank + pro_stats) for on-disk checkpoint. The latched
        soft-breach days ride along inside pro_stats."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "drawdown": self.drawdown.to_dict(),
            "drawdown_criteria": self.drawdown_criteria.value,
            "intraday_drawdown_threshold_override": self.intraday_drawdown_threshold_override,
            "eod_hwm_threshold_override": self.eod_hwm_threshold_override,
            "rank": self.rank,
            "pro_stats": self.pro_stats.to_dict(),
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
            intraday_drawdown_threshold_override=data.get("intraday_drawdown_threshold_override"),
            eod_hwm_threshold_override=data.get("eod_hwm_threshold_override"),
            rank=data.get("rank"),
            pro_stats=ProStats.from_dict(data.get("pro_stats")),
        )

    def __str__(self) -> str:
        from datetime import datetime
        start = datetime.fromtimestamp(self.current_bucket_start_ms / 1000).strftime("%Y-%m-%d %H:%M")
        dd = self.drawdown
        criteria_str = self.drawdown_criteria.value
        return (
            f"{self.hotkey} {self.current_phase} {start} rank={self.rank} criteria={criteria_str} idt_override={self.intraday_drawdown_threshold_override} "
            f"eod_override={self.eod_hwm_threshold_override} "
            f"equity={dd.current_equity:.4f} balance={dd.current_balance:.4f} daily_open={f'{dd.daily_open_equity:.4f}' if dd.daily_open_equity is not None else 'None'} | "
            f"intraday_dd={dd.intraday_drawdown_pct:.2f}% eod_dd={dd.eod_drawdown_pct:.2f}% "
            f"static_dd={dd.static_drawdown_pct:.2f}% static_eod_dd={dd.static_eod_drawdown_pct:.2f}% "
            f"eod_hwm={dd.eod_hwm:.4f} | "
            f"calmar={self.pro_stats.calmar:.2f} consistency={self.pro_stats.daily_consistency:.2f} "
            f"pro_mdd={self.pro_stats.max_drawdown:.4f} pro_days={self.pro_stats.trading_days}"
        )

    @property
    def is_eliminated(self):
        return self.current_bucket == MinerBucket.ELIMINATED

    @property
    def current_bucket(self):
        return self.bucket()

    @property
    def current_phase(self) -> BucketPhase:
        return self.phase()

    @property
    def in_transition(self) -> bool:
        """True while a SUBACCOUNT_FUNDED subaccount is in its transition week."""
        return self.entries[-1].in_transition

    @property
    def current_bucket_start_ms(self) -> int:
        return self.entries[-1].start_time_ms

    @property
    def current_bucket_entry(self) -> BucketEntry:
        return self.entries[-1]

    def _threshold_time_for(self, entry: BucketEntry) -> int | None:
        """For the funded-rule phases, returns the SUBACCOUNT_CHALLENGE entry's start time
        so versioned thresholds (V0/V1) are selected based on when the miner registered,
        not when they were promoted. For all other phases, returns the entry's own start."""
        if entry.phase.uses_versioned_funded_thresholds:
            challenge_entry = next((e for e in self.entries if e.bucket == MinerBucket.SUBACCOUNT_CHALLENGE), None)
            return challenge_entry.start_time_ms if challenge_entry else None
        return entry.start_time_ms

    @property
    def _threshold_time_ms(self) -> int | None:
        return self._threshold_time_for(self.current_bucket_entry)

    def _intraday_threshold_for(self, phase: BucketPhase, threshold_time_ms: int | None) -> float:
        """An intraday drawdown threshold chosen at creation applies in every bucket but PRO_FUNDED. Otherwise
        static accounts use a flat intraday threshold regardless of bucket or registration time, and pro buckets
        keep their own threshold even when the subaccount was created static."""
        if self.intraday_drawdown_threshold_override is not None and phase.bucket != MinerBucket.PRO_FUNDED:
            return self.intraday_drawdown_threshold_override
        if self.drawdown_criteria == DrawdownCriteria.STATIC and not phase.is_pro:
            return ValiConfig.SUBACCOUNT_STATIC_INTRADAY_DRAWDOWN_THRESHOLD
        return phase.intraday_drawdown_threshold(threshold_time_ms)

    @property
    def intraday_drawdown_threshold(self):
        if self.current_bucket == MinerBucket.ELIMINATED:
            if len(self.entries) >= 2:
                prev_entry = self.entries[-2]
                return self._intraday_threshold_for(prev_entry.phase, self._threshold_time_for(prev_entry))
            else:
                # Dummy threshold value for eliminated accounts to not raise Error
                return self._intraday_threshold_for(BucketPhase(MinerBucket.SUBACCOUNT_CHALLENGE), None)
        return self._intraday_threshold_for(self.current_phase, self._threshold_time_ms)

    def _eod_threshold_for(self, phase: BucketPhase, threshold_time_ms: int | None) -> float:
        """An EOD threshold chosen at creation applies in every bucket but PRO_FUNDED."""
        if self.eod_hwm_threshold_override is not None and phase.bucket != MinerBucket.PRO_FUNDED:
            return self.eod_hwm_threshold_override
        return phase.eod_drawdown_threshold(threshold_time_ms)

    @property
    def intraday_drawdown_threshold_pct(self):
        return self.intraday_drawdown_threshold * 100

    @property
    def eod_drawdown_threshold(self):
        if self.current_bucket == MinerBucket.ELIMINATED:
            if len(self.entries) >= 2:
                prev_entry = self.entries[-2]
                return self._eod_threshold_for(prev_entry.phase, self._threshold_time_for(prev_entry))
            else:
                # Dummy threshold value for eliminated accounts to not raise Error
                return self._eod_threshold_for(BucketPhase(MinerBucket.SUBACCOUNT_CHALLENGE), None)
        return self._eod_threshold_for(self.current_phase, self._threshold_time_ms)

    @property
    def eod_drawdown_threshold_pct(self):
        return self.eod_drawdown_threshold * 100

    @property
    def soft_breach(self) -> bool:
        """True when a pro rule is currently breached in a bucket that withholds the week's payout.
        """
        bucket = self.current_bucket
        if not bucket.soft_breach_applies:
            return False
        return self.pro_stats.calmar < bucket.calmar_threshold

    def latch_soft_breach(self, current_time_ms: int) -> bool:
        """Record the current week as checked, and today as breached when a pro rule is broken.

        Calmar moves every refresh but is only persisted onto 12h checkpoints, so a breach that
        opens and heals between two checkpoints would never reach the payout. Latching the UTC day
        makes the weekly decision read "breached at any point when checked".

        Returns True when anything was recorded (the caller must persist the state).
        """
        if not self.current_bucket.soft_breach_applies:
            return False
        weeks = self.pro_stats.soft_breach_weeks
        week_ms = TimeUtil.ms_at_start_of_week(current_time_ms)
        changed = week_ms not in weeks
        days = weeks.setdefault(week_ms, [])
        day_ms = TimeUtil.get_start_of_day_ms(current_time_ms)
        if self.soft_breach and day_ms not in days:
            days.append(day_ms)
            days.sort()
            changed = True
        return changed

    def is_soft_breach_latched(self, day_ms: int) -> bool:
        """True when the UTC day starting at day_ms was latched as breached."""
        return day_ms in self.pro_stats.breach_days(TimeUtil.ms_at_start_of_week(day_ms))



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
    DRAWDOWN_ACTIVATION_MS = TimeUtil.formatted_date_str_to_millis("2026-10-01 00:00:00")   # TODO: remove

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
        # Per-bucket record of every stint, kept after the live state is wound down. Off while
        # backtesting, which never persists anything (see _save_to_disk)
        self._bucket_history = BucketHistoryStore(
            running_unit_tests=running_unit_tests, enabled=not is_backtesting or running_unit_tests)

        # Cached scores for MinerStatisticsManager
        self._cached_asset_softmaxed_scores: dict[MinerAssetClass, dict[str, float]] = {}
        self._cached_asset_competitiveness: dict[MinerAssetClass, float] = {}

        logger.info(f"[CP_MANAGER] ChallengePeriodManager initialized with {len(self.miner_states)} state data")

    # ==================== Core Business Logic ====================

    def refresh(self, current_time_ms: int | None = None, iteration_epoch: int | None = None):
        if current_time_ms is None:
            current_time_ms = TimeUtil.now_in_millis()

        logger.info(f"[CHALLENGE] Starting challenge period loop {current_time_ms} iteration_epoch={iteration_epoch}")

        timer = StageTimer()

        asset_selections = self._asset_selection_client.get_asset_selections()
        # Only first order times here; full positions are fetched below for rank-based miners only
        hk_to_first_order_time = self._position_client.get_first_order_times()
        hotkeys_elimination_sync = list(self._elimination_client.get_eliminated_hotkeys())
        hotkeys_plagiarism_sync = list(self._plagiarism_client.get_plagiarism_miners())
        timer.lap("fetch_inputs")

        state_changed = False
        state_changed |= self._sync_positions(
            hotkeys=list(hk_to_first_order_time.keys()),
            eliminated_hotkeys=hotkeys_elimination_sync,
            hk_to_first_order_time_ms=hk_to_first_order_time,
            default_time=current_time_ms,
        )
        state_changed |= self.sync_plagiarism_miners(hotkeys_plagiarism_sync, current_time_ms)
        state_changed |= self.sync_elimination_miners(hotkeys_elimination_sync, current_time_ms)
        state_changed |= self._prune_hotkeys_no_positions()
        # Bucket moves made outside this manager's own promote/demote/eliminate paths (plagiarism,
        # elimination sync, new miners, entity registration) open and close their segments here
        self._reconcile_bucket_history(list(self.miner_states), current_time_ms)
        timer.lap("sync")

        self._current_iteration_epoch = iteration_epoch

        evaluation_hotkeys = [hotkey for hotkey, state in self.miner_states.items() if state.current_bucket.is_active]
        rank_hotkeys = [hotkey for hotkey, state in self.miner_states.items() if state.current_bucket.is_rank_based]

        accounts = self._miner_account_client.get_accounts(evaluation_hotkeys)
        ledgers = self._perf_ledger_client.filtered_ledger_for_scoring(evaluation_hotkeys)
        rank_positions, _ = self._position_client.filtered_positions_for_scoring(hotkeys=rank_hotkeys)
        timer.lap("fetch_eval_data")
        self._refresh_drawdown_cache(evaluation_hotkeys, accounts, ledgers, set(hk_to_first_order_time), current_time_ms)
        timer.lap("drawdown_cache")
        self._refresh_pro_stats(evaluation_hotkeys, ledgers, accounts)
        timer.lap("pro_stats")
        # Latch before any bucket move below: soft_breach reads the bucket the miner traded today
        self._latch_soft_breaches(evaluation_hotkeys, current_time_ms)
        timer.lap("latch_soft_breaches")
        self._record_bucket_stats(evaluation_hotkeys, current_time_ms)
        timer.lap("bucket_stats")
        self._refresh_rank_cache(rank_hotkeys, ledgers, rank_positions, accounts, asset_selections, current_time_ms)
        timer.lap("rank_cache")

        eliminations: dict[str, EliminationReason] = {}
        demotions: dict[str, MinerBucket] = {}
        promotions = []
        for hotkey, state in self.miner_states.items():
            if hotkey not in evaluation_hotkeys:
                if state.current_bucket == MinerBucket.UNKNOWN:
                    logger.warning(f"[CHALLENGE] {hotkey} bucket unknown")
                continue

            if reason := self._check_time(state, current_time_ms):
                eliminations[hotkey] = reason
                continue

            if state.current_phase.is_pro:
                # Rule 1: Daily loss limit — equity cannot drop below the day's opening equity
                if reason := self._check_intraday_drawdown(state):
                    eliminations[hotkey] = reason
                    continue

                # Rule 2: EOD Trailing loss limit — the end-of-day equity cannot drop below the
                # highest EOD equity. Checked once per UTC day, not in real time.
                if reason := self._check_eod_drawdown(state):
                    eliminations[hotkey] = reason
                    continue
            elif state.drawdown_criteria == DrawdownCriteria.STATIC:
                # Static rules for subaccounts registered after the effective time (Hyperscaled excluded)
                # Rule 1: Static drawdown — equity (including unrealized PnL) cannot drop more than 5% below starting balance
                if reason := self._check_static_drawdown(state):
                    eliminations[hotkey] = reason
                    continue

                # Rule 2: Daily loss limit — equity cannot drop more than the flat static threshold
                # below today's opening equity
                if reason := self._check_intraday_drawdown(state):
                    if state.current_bucket.is_subaccount or current_time_ms > self.DRAWDOWN_ACTIVATION_MS:
                        eliminations[hotkey] = reason
                    continue
            else:
                # Trailing rules: subaccounts eliminate immediately; regular miners only after activation
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

            # The transition wind-down ends on the week boundary, regardless of performance
            if self._check_transition_expiry(state, current_time_ms):
                promotions.append(hotkey)
                continue

            _asset = asset_selections.get(hotkey)
            if _asset is None:
                logger.warning(f"[CHALLENGE] {hotkey} no asset selection, skipping evaluation")
                continue
            asset_class = MinerAssetClass(_asset)

            if self._check_demotion(state):
                demotions[hotkey] = MinerBucket.PROBATION
                continue

            returns_threshold = state.current_phase.returns_threshold(asset_class)
            if hotkey == "5EPeU7Y8bqokEVf31ZWPZkP3F7Kv1v3ALuhnpp5T5Fvfjp85_87": # remove once eliminated or promoted
                returns_threshold = 0.08
            if self._check_promotion(state, returns_threshold, current_time_ms):
                promotions.append(hotkey)
                continue

        timer.lap("evaluate")

        state_changed |= self.eliminate_hotkeys(eliminations, current_time_ms)
        state_changed |= self.demote_hotkeys(demotions, current_time_ms)
        state_changed |= self.promote_hotkeys(promotions, current_time_ms)

        if state_changed:
            self._sync_buckets_to_accounts(hotkeys=list({*eliminations, *demotions, *promotions}))
        timer.lap("apply_changes")

        self._save_to_disk()
        timer.lap("save_to_disk")

        counts = {}
        for state in self.miner_states.values():
            counts[state.current_bucket] = counts.get(state.current_bucket, 0) + 1
        snapshot = " | ".join(f"{b.value}={n}" for b, n in sorted(counts.items(), key=lambda x: x[0].value))
        logger.info(f"[CHALLENGE] snapshot: {snapshot} (total={len(self.miner_states)})")

        logger.info(f"[CHALLENGE] timings: {timer.summary()} "
                    f"(evaluated={len(evaluation_hotkeys)}, ranked={len(rank_hotkeys)})")

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
    def _drawdown_reason(phase: BucketPhase, rule: str) -> EliminationReason:
        """Map (bucket phase, drawdown rule) to the elimination reason for that account tier.
        Pro buckets only ever pair with INTRADAY and EOD; the static rules are non-pro only."""
        bucket = phase.bucket
        if bucket == MinerBucket.PRO_FUNDED:
            tier = "PRO_FUNDED_PERIOD"
        elif phase.is_pro:
            tier = "PRO_CHALLENGE_PERIOD"
        elif bucket in (MinerBucket.CHALLENGE, MinerBucket.SUBACCOUNT_CHALLENGE):
            tier = "CHALLENGE_PERIOD"
        else:
            tier = "FUNDED_PERIOD"
        return EliminationReason[f"FAILED_{tier}_{rule}_DRAWDOWN"]

    @classmethod
    def _check_intraday_drawdown(cls, state: MinerBucketState) -> EliminationReason | None:
        if state.drawdown.intraday_drawdown_pct > state.intraday_drawdown_threshold_pct:
            logger.warning(f"[CHALLENGE] BREACH intraday drawdown {state.intraday_drawdown_threshold_pct}%: {state}")
            return cls._drawdown_reason(state.current_phase, "INTRADAY")
        elif state.drawdown.intraday_drawdown_pct > state.intraday_drawdown_threshold_pct * 0.75:
            logger.info(f"[CHALLENGE] near intraday drawdown {state.intraday_drawdown_threshold_pct}%: {state}")
        return None

    @classmethod
    def _check_eod_drawdown(cls, state: MinerBucketState) -> EliminationReason | None:
        if state.drawdown.eod_drawdown_pct > state.eod_drawdown_threshold_pct:
            logger.warning(f"[CHALLENGE] BREACH EOD drawdown {state.eod_drawdown_threshold_pct}%: {state}")
            return cls._drawdown_reason(state.current_phase, "EOD")

        # Live equity already past the limit: tonight's latch breaches unless the miner recovers
        if state.drawdown.trailing_drawdown_pct > state.eod_drawdown_threshold_pct:
            logger.info(f"[CHALLENGE] near trailing EOD with current equity {state.eod_drawdown_threshold_pct}%: {state}")
        elif state.drawdown.eod_drawdown_pct > state.eod_drawdown_threshold_pct * 0.75:
            logger.info(f"[CHALLENGE] near EOD drawdown {state.eod_drawdown_threshold_pct}%: {state}")

        return None

    @classmethod
    def _check_static_drawdown(cls, state: MinerBucketState) -> EliminationReason | None:
        phase = state.current_phase
        if phase.is_pro:
            return None
        threshold_pct = ValiConfig.SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD * 100
        if state.drawdown.static_drawdown_pct > threshold_pct:
            logger.warning(f"[CHALLENGE] BREACH static drawdown {threshold_pct}%: {state}")
            return cls._drawdown_reason(phase, "STATIC")
        elif state.drawdown.static_drawdown_pct > threshold_pct * 0.75:
            logger.info(f"[CHALLENGE] near static drawdown {threshold_pct}%: {state}")
        return None

    @staticmethod
    def _check_transition_expiry(state: MinerBucketState, current_time_ms: int) -> bool:
        """True once the SUBACCOUNT_FUNDED transition has run past the first Monday 00:00 UTC
        after the miner entered it"""
        if not state.in_transition:
            return False
        expired = (TimeUtil.ms_at_start_of_week(current_time_ms)
                   > TimeUtil.ms_at_start_of_week(state.current_bucket_start_ms))
        if expired:
            logger.info(f"[CHALLENGE] transition window closed: {state}")
        return expired

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
        # The transition leaves on its week boundary (_check_transition_expiry), never on returns
        if state.in_transition:
            return False

        if state.current_bucket == MinerBucket.CHALLENGE:
            if current_time_ms - state.current_bucket_start_ms < ValiConfig.CHALLENGE_PERIOD_MINIMUM_MS:
                return False

        phase = state.current_phase
        if phase.next_phase is None:
            return False

        minimum_trading_days = phase.minimum_trading_days
        if minimum_trading_days is not None and state.pro_stats.trading_days < minimum_trading_days:
            return False

        calmar_threshold = phase.calmar_threshold
        if calmar_threshold is not None and state.pro_stats.calmar < calmar_threshold:
            return False

        consistency_threshold = phase.daily_consistency_threshold
        if consistency_threshold is not None and state.pro_stats.daily_consistency > consistency_threshold:
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

        failed_hotkeys = set()
        for hotkey, elimination_reason in eliminations.items():
            state = self.miner_states[hotkey]
            if not state.current_bucket.is_active:
                logger.warning(f"[CHALLENGE] attempted elimination in non-eligible bucket: {state}")

            logger.info(f"[CHALLENGE] eliminating reason={elimination_reason.value}: {state}")

            is_pro = state.current_phase.is_pro
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

            # Pro runs the pro rules even on a static subaccount, so static reporting is non-pro only
            is_static = state.drawdown_criteria == DrawdownCriteria.STATIC and not is_pro
            # intraday_drawdown_pct always holds the literal equity-vs-day-open number
            intraday_drawdown_pct = state.drawdown.intraday_drawdown_pct
            # eod_drawdown_pct holds the account's other rule: static equity-vs-starting-balance for
            # static accounts, or the EOD-vs-high-water-mark check otherwise
            eod_drawdown_pct = state.drawdown.static_drawdown_pct if is_static else state.drawdown.eod_drawdown_pct

            # One failure (e.g. a slash that outlasts the RPC timeout) must not abort the rest of the
            # batch. The failed hotkey stays in its bucket; if its elimination completes server-side,
            # sync_elimination_miners marks it ELIMINATED on the next pass, otherwise it is retried.
            try:
                self._elimination_client.append_elimination_row(
                    hotkey=hotkey,
                    reason=elimination_reason,
                    elimination_drawdown_pct=elimination_drawdown_pct,
                    intraday_drawdown_pct=intraday_drawdown_pct,
                    eod_drawdown_pct=eod_drawdown_pct,
                    elimination_time_ms=elimination_time_ms,
                    bucket_at_elimination=state.current_bucket,
                )
            except Exception as e:
                logger.error(f"[CHALLENGE] elimination of {hotkey} failed, retrying next pass: {e}", exc_info=True)
                failed_hotkeys.add(hotkey)

        state_changed = False
        with self._buckets_lock:
            for hotkey in eliminations:
                if hotkey in failed_hotkeys:
                    continue
                state_changed |= self.miner_states[hotkey].add_bucket_entry(MinerBucket.ELIMINATED, current_time_ms)

        self._reconcile_bucket_history([hk for hk in eliminations if hk not in failed_hotkeys], current_time_ms)
        return state_changed

    def _switch_account(self, hotkey: str, target: BucketPhase | MinerBucket, current_time_ms: int) -> bool:
        """Wind down the account a miner is leaving. Run whenever a bucket change also changes
        the account size, so the new account's performance is tracked from scratch. A bare
        MinerBucket target means the bucket out of transition.

        Returns False when the account size could not be pointed at the target bucket. Nothing is
        wound down in that case: the caller must leave the miner in their current bucket rather
        than strand them on the new one with the old size and no history.
        """
        if isinstance(target, MinerBucket):
            target = target.phase()
        # Point the subaccount at the size the target bucket trades before resetting the account
        if is_synthetic_hotkey(hotkey):
            success, message = self._entity_client.apply_bucket_account_size(
                hotkey, target.bucket, in_transition=target.in_transition)
            if not success:
                logger.error(
                    f"[CHALLENGE] {hotkey} bucket change to {target} ABORTED - "
                    f"account size not updated: {message}"
                )
                return False

        self._close_out_account(hotkey, current_time_ms)
        # Keep the closed positions and final stats of the account in its bucket segment before
        # anything below archives or resets them
        self._record_wound_down_account(hotkey, current_time_ms)
        # Track payout information already gathered before promotion since account will be wound down
        from_phase = self.miner_states[hotkey].current_phase
        if is_synthetic_hotkey(hotkey) and from_phase.is_subaccount_earning:
            self._entity_client.settle_wound_down_segment(hotkey, from_phase.status, current_time_ms)
        # Reset account fields (PnL, capital used, borrowed amount, interest)
        self._miner_account_client.reset_account(hotkey, target.bucket, in_transition=target.in_transition)
        # Archive all positions (disk move + memory removal)
        self._position_client.archive_positions_for_hotkey(hotkey, archive_all=True)
        # Wipe perf ledgers so the new account's performance is tracked from scratch
        self._perf_ledger_client.wipe_miners_perf_ledgers([hotkey])
        # Delete debt ledger to match new perf ledger checkpoints
        self._debt_ledger_client.delete_debt_ledger(hotkey)
        # Reset drawdown cache
        self._reset_drawdown_stats_cache(hotkey)
        # Reset pro stats so neither the ratcheted drawdown nor the latched breach days carry into
        # the new account. This never runs on the PRO_FUNDED hop, which keeps the same account and
        # so keeps both.
        self.miner_states[hotkey].pro_stats = ProStats()
        return True

    def _close_out_account(self, hotkey: str, current_time_ms: int) -> None:
        """Close every position and cancel every limit order, leaving the account and its history."""
        self._position_client.close_all_positions(
            hotkey=hotkey,
            close_time_ms=current_time_ms,
            order_source=OrderSource.SUBACCOUNT_PROMOTION
        )
        self._limit_order_client.cancel_limit_order(hotkey, None, "ALL", current_time_ms)

    def _leave_phase(self, hotkey: str, target: BucketPhase, current_time_ms: int) -> bool:
        """Wind down what the miner is leaving for target: a full account switch, or just closing
        the positions on a move into or out of an earning bucket that keeps the account. False when
        the switch failed."""
        if target.switches_account:
            return self._switch_account(hotkey, target, current_time_ms)
        if target.closes_positions_from(self.miner_states[hotkey].current_phase):
            self._close_out_account(hotkey, current_time_ms)
            self._record_segment_payout(hotkey, current_time_ms)
        return True

    def demote_hotkeys(self, demotions: dict[str, MinerBucket], current_time_ms) -> bool:
        """Demote miners to the given target bucket."""
        if demotions:
            logger.info(f"[CHALLENGE] demoting {len(demotions)} miners")

        state_changed = False
        for hotkey, target_bucket in demotions.items():
            logger.info(f"[CHALLENGE] demoting to {target_bucket.value}: {self.miner_states[hotkey]}")

            # Leave the miner where they are when the account switch fails; the next refresh retries
            if not self._leave_phase(hotkey, target_bucket.phase(), current_time_ms):
                continue

            with self._buckets_lock:
                state_changed |= self.miner_states[hotkey].add_bucket_entry(target_bucket, current_time_ms)
            self._reconcile_bucket_history([hotkey], current_time_ms)
        return state_changed

    def promote_hotkeys(self, hotkeys: list[str], current_time_ms: int) -> bool:
        """Promote miners to next tier."""
        if hotkeys:
            logger.info(f"[CHALLENGE] promoting {len(hotkeys)} miners")

        state_changed = False
        for hotkey in hotkeys:
            state = self.miner_states[hotkey]
            target = state.current_phase.next_phase
            if target is None:
                logger.warning(f"[CHALLENGE] no next bucket for promotion: {state}")
                continue

            logger.info(f"[CHALLENGE] promoting to {target}: {state}")

            # Leave the miner where they are when the account switch fails; the next refresh retries
            if not self._leave_phase(hotkey, target, current_time_ms):
                continue

            with self._buckets_lock:
                state_changed |= self.miner_states[hotkey].add_bucket_entry(
                    target.bucket, current_time_ms, in_transition=target.in_transition)
            self._reconcile_bucket_history([hotkey], current_time_ms)

        return state_changed

    def promote_subaccount(
        self, hotkey: str, current_time_ms: int, pro_account_size: float | None = None
    ) -> tuple[bool, str]:
        """Promote a subaccount one step up the pro track on the entity miner's request.

        The only moves allowed are the three hops in BucketPhase.promotion_target; a subaccount in
        any other bucket is rejected, as is an Instant Funded subaccount in SUBACCOUNT_FUNDED.
        pro_account_size is the size the entity asked for: entering the pro track needs one, and None
        keeps the size already recorded (see EntityManager.apply_bucket_account_size, which rejects a
        subaccount that has neither).

        Only a target with switches_account wipes trading state, which is the two hops landing on a
        pro account (PRO_CHALLENGE_DIRECT, and PRO_CHALLENGE_FUNDED):
        positions closed, limit orders cancelled, ledgers restarted. The SUBACCOUNT_FUNDED
        transition is a wind-down week on the standard account, so the hop into it keeps the
        positions, limit orders and ledgers as they are and only sweeps the entry orders (see
        admin_set_bucket).
        """
        state = self.miner_states.get(hotkey)
        if state is None:
            return False, f"{hotkey} not found in challenge period manager"

        current_phase = state.current_phase
        target = current_phase.promotion_target
        if target is None:
            return False, f"{hotkey} cannot be promoted out of {current_phase}"
        # Checked before sizing so a refused move has nothing to roll back
        can_set, reason = self.can_admin_set_bucket(hotkey, target.bucket, in_transition=target.in_transition)
        if not can_set:
            return False, reason

        logger.info(f"[CHALLENGE] promotion to {target} requested "
                    f"(pro_account_size={pro_account_size}): {state}")

        # Record the granted pro size first: the account switch reads it back to size the new account.
        # Every pro phase past the transition trades the pro size
        sizing_snapshot = self._entity_client.snapshot_bucket_account_size(hotkey)
        success, message = self._entity_client.apply_bucket_account_size(
            hotkey, target.bucket, pro_account_size, in_transition=target.in_transition
        )
        if not success:
            logger.warning(f"[CHALLENGE] promotion rejected for {hotkey}: {message}")
            return False, message

        try:
            success, message = self.admin_set_bucket(hotkey, target.bucket, current_time_ms,
                                                     in_transition=target.in_transition)
        except Exception:
            # admin_set_bucket can raise after the bucket entry has landed: its entry-order sweep
            # and its disk write both run past that point. The sizing belongs to whichever side of
            # that line the miner ended up on, so give it back only if the bucket did not move -
            # restoring it after a move would strand a pro bucket on the standard size.
            state = self.miner_states.get(hotkey)
            if state is None or state.current_phase != target:
                self._restore_bucket_account_size(hotkey, sizing_snapshot)
            else:
                logger.error(f"[CHALLENGE] {hotkey} reached {target} before failing; "
                             f"keeping the sizing that bucket trades")
            raise
        if not success:
            # Every False return is from before the bucket entry (can_admin_set_bucket, the account
            # switch), so the move definitely did not happen and the sizing always goes back.
            self._restore_bucket_account_size(hotkey, sizing_snapshot)
        return success, message

    def _restore_bucket_account_size(self, hotkey: str, sizing_snapshot: dict | None) -> None:
        """Put a committed apply_bucket_account_size back when the bucket move it was for did not
        happen. A no-op without a snapshot, and never allowed to mask the failure it is cleaning up."""
        if not sizing_snapshot:
            return
        try:
            restored, message = self._entity_client.restore_bucket_account_size(hotkey, sizing_snapshot)
            if not restored:
                logger.error(f"[CHALLENGE] could not roll back the account size for {hotkey}: {message}")
        except Exception as restore_error:
            logger.error(f"[CHALLENGE] could not roll back the account size for {hotkey}: {restore_error}")

    def can_admin_set_bucket(self, hotkey: str, bucket: MinerBucket, in_transition: bool = False) -> tuple[bool, str]:
        """Report whether admin_set_bucket would reject this move, without changing anything."""
        target = BucketPhase(bucket, in_transition)
        state = self.miner_states.get(hotkey)
        if state is None:
            return False, f"{hotkey} not found in challenge period manager"
        if state.current_phase == target:
            return False, f"{hotkey} is already in {target}"
        # Instant Funded accounts never enter the pro track through the transition
        if target.in_transition and self._entity_client.is_instant_funded(hotkey):
            return False, f"{hotkey} is Instant Funded and cannot move to {target}"
        return True, ""

    def admin_set_bucket(self, hotkey: str, bucket: MinerBucket, current_time_ms: int,
                         in_transition: bool = False) -> tuple[bool, str]:
        """Move a miner into an arbitrary bucket. Runs the same account switch as an organic
        promotion when the target changes the account size. in_transition only applies to
        SUBACCOUNT_FUNDED: set, it enters the transition week on the standard account."""
        can_set, reason = self.can_admin_set_bucket(hotkey, bucket, in_transition)
        if not can_set:
            return False, reason

        target = BucketPhase(bucket, in_transition)
        state = self.miner_states[hotkey]
        logger.info(f"[CHALLENGE] admin moving to {target}: {state}")

        if target.switches_account and not self._switch_account(hotkey, target, current_time_ms):
            return False, f"{hotkey} account size update failed, bucket unchanged"

        # PRO_FUNDED keeps the account, so it never runs the switch that sizes one. An organic
        # promotion arrives there already on the pro size, but an admin can move a subaccount
        # straight in from a standard bucket, and a pro bucket has to trade the pro size.
        if not target.switches_account and target.is_pro and is_synthetic_hotkey(hotkey):
            sized, message = self._entity_client.apply_bucket_account_size(hotkey, bucket)
            if not sized:
                logger.error(
                    f"[CHALLENGE] {hotkey} bucket change to {target} ABORTED - "
                    f"account size not updated: {message}"
                )
                return False, f"{hotkey} account size update failed, bucket unchanged: {message}"

        if not target.switches_account and target.closes_positions_from(state.current_phase):
            self._close_out_account(hotkey, current_time_ms)
            self._record_segment_payout(hotkey, current_time_ms)

        with self._buckets_lock:
            state.add_bucket_entry(bucket, current_time_ms, in_transition=target.in_transition)
        self._reconcile_bucket_history([hotkey], current_time_ms)

        self._sync_buckets_to_accounts(hotkeys=[hotkey])

        if target.in_transition:
            self._limit_order_client.cancel_entry_orders(
                hotkey, current_time_ms, OrderSource.PRO_TRANSITION_CANCELLED
            )

        self._save_to_disk()
        return True, f"{hotkey} moved to {target}"

    def revert_elimination(self, hotkey: str) -> bool:
        """Undo an elimination, leaving the account as though it had never happened.

        The ELIMINATED entry is always kept as the record that it happened, and the previous bucket
        is appended on top.

        For PRO_FUNDED it is additionally marked reverted. Bucket history is replayed
        point-in-time when the penalty and debt ledgers are rebuilt
        (PenaltyLedgerManager._get_status_for_checkpoint), so a reverted span that still governed
        its window would reclassify every checkpoint inside it as non-earning and off the pro
        soft-breach track - forfeiting escrow for an elimination that was undone. Marking it keeps
        both the audit trail and the original classification. PRO_FUNDED also keeps its end-of-day
        high water mark, so the trailing loss limit is not silently reset, and pro_stats carries
        the ratcheted drawdown and the latched soft-breach days so breached weeks stay breached.

        """
        miner_state = self.miner_states.get(hotkey)
        if not miner_state:
            return False

        with self._buckets_lock:
            if miner_state.current_bucket != MinerBucket.ELIMINATED:
                return False

            if len(miner_state.entries) < 2:
                return False

            now_ms = TimeUtil.now_in_millis()
            prev_phase = miner_state.entries[-2].phase
            restores_pro_account = prev_phase.bucket == MinerBucket.PRO_FUNDED

            # The ELIMINATED entry stays as the record that it happened, but is marked reverted so
            # replaying this history never stamps a checkpoint inside the span with it
            if restores_pro_account:
                miner_state.entries[-1].reverted_ms = now_ms
            miner_state.add_bucket_entry(prev_phase.bucket, now_ms, in_transition=prev_phase.in_transition)
            logger.info(
                f"[CHALLENGE] reverted elimination, restored to {miner_state.current_bucket.value}: {miner_state}"
            )

        self._reconcile_bucket_history([hotkey], now_ms)
        self._reset_drawdown_stats_cache(hotkey, preserve_eod=restores_pro_account)
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
        hotkeys_with_positions: set[str],
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

            if hotkey not in hotkeys_with_positions:
                # skip log: no positions mean new account or recently promoted, no ledger
                continue

            now_ms = current_time_ms if current_time_ms is not None else TimeUtil.now_in_millis()
            current_day_open_ms = TimeUtil.get_start_of_day_ms(now_ms)
            # Held so a manual set_miner_drawdown_stats cannot land between reading and writing eod_hwm
            with self._buckets_lock:
                existing = self.miner_states[hotkey].drawdown
                existing.current_equity = current_equity
                existing.current_balance = current_balance

                # EOD fields are locked in for today once the snapshot has been captured
                if existing.last_eod_checked_ms == current_day_open_ms:
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
                    existing.current_equity = current_equity
                    existing.current_balance = current_balance
                    existing.last_eod_equity = last_eod_equity
                    existing.daily_open_equity = daily_open_equity
                    existing.eod_hwm = max(existing.eod_hwm, last_eod_equity, eod_hwm)
                    existing.last_eod_checked_ms = last_eod_checked_ms

    def _refresh_pro_stats(self, hotkeys: list[str], ledgers: dict[str, PerfLedger],
                           accounts: dict[str, MinerAccount]) -> None:
        for hotkey in hotkeys:
            state = self.miner_states[hotkey]
            if not state.current_phase.is_pro_track:
                continue

            # Worst live equity against the end-of-day high-water mark, in mdd ratio form.
            # eod_hwm only advances at UTC midnight, so an intraday peak never deepens it.
            max_drawdown = state.pro_stats.max_drawdown
            if state.drawdown.eod_hwm > 0:
                max_drawdown = min(max_drawdown, 1.0, state.drawdown.current_equity / state.drawdown.eod_hwm)
            else:
                logger.warning(f"[CHALLENGE] {hotkey} non-positive eod_hwm, holding ratchet: {state}")
            state.pro_stats.max_drawdown = max_drawdown

            ledger = ledgers.get(hotkey)
            if ledger is None:
                logger.warning(f"[CHALLENGE] {hotkey} missing ledger, skipping pro stats")
                continue

            account = accounts.get(hotkey)
            if account is None or account.account_size <= 0:
                logger.warning(f"[CHALLENGE] {hotkey} invalid account, skipping pro stats")
                continue

            log_returns = LedgerUtils.daily_return_log(ledger)
            state.pro_stats = ProStats(
                calmar=Metrics.all_time_calmar(
                    LedgerUtils.realized_return(ledger, account.account_size), max_drawdown),
                daily_consistency=Metrics.return_consistency(log_returns),
                max_drawdown=max_drawdown,
                trading_days=len(log_returns),
                # Recomputed metrics, but the latch is a history: carry it
                soft_breach_weeks=state.pro_stats.soft_breach_weeks,
            )

    def _latch_soft_breaches(self, hotkeys: list[str], current_time_ms: int) -> None:
        """Record today as breached for every miner currently breaking a pro rule.

        Runs on the same cadence as the rest of the refresh loop, so a breach is caught the moment
        it appears rather than only when the next 12h checkpoint closes. Latched days are read back
        by PenaltyLedgerManager to withhold the whole payout week.
        """
        for hotkey in hotkeys:
            state = self.miner_states.get(hotkey)
            if state is None or not state.current_bucket.soft_breach_applies:
                continue
            if state.latch_soft_breach(current_time_ms) and state.soft_breach:
                logger.warning(
                    f"[CHALLENGE] SOFT BREACH latched for {TimeUtil.millis_to_formatted_date_str(current_time_ms)}: "
                    f"{state}"
                )

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
                if asset_selections.get(hotkey) == asset_class
            }
            sorted_scores = sorted(miner_scores.items(), key=lambda item: item[1], reverse=True)
            for i, (hotkey, _) in enumerate(sorted_scores):
                self.miner_states[hotkey].rank = i + 1

    def _reset_drawdown_stats_cache(self, hotkey: str, preserve_eod: bool = False) -> None:
        """Reset a hotkey's drawdown stats cache to neutral default values.

        preserve_eod keeps the end-of-day high water mark and the last EOD reading. Use it whenever
        the account itself survives the reset (reverting an elimination), since eod_hwm is the
        denominator of the trailing loss limit and clearing it hands the miner a fresh 8% of room.
        The live equity fields are always cleared; the next refresh repopulates them.
        """
        state = self.miner_states[hotkey]
        if not preserve_eod:
            state.drawdown = DrawdownStats()
            return

        previous = state.drawdown
        state.drawdown = DrawdownStats(
            eod_hwm=previous.eod_hwm,
            last_eod_equity=previous.last_eod_equity,
            last_eod_checked_ms=previous.last_eod_checked_ms,
        )

    # ==================== Bucket History ====================

    def _reconcile_bucket_history(self, hotkeys: list[str], current_time_ms: int | None = None) -> None:
        """Open and close bucket segments to match each hotkey's bucket entries. Never raises: the
        history is a record of the bucket moves, and must not be able to block one."""
        if not self._bucket_history.enabled or not hotkeys:
            return
        now_ms = current_time_ms if current_time_ms is not None else TimeUtil.now_in_millis()
        changed = []
        for hotkey in hotkeys:
            state = self.miner_states.get(hotkey)
            if state is None:
                continue
            try:
                if self._bucket_history.reconcile(hotkey, list(state.entries), now_ms,
                                                  on_open=self._on_segment_open,
                                                  on_close=self._on_segment_close, save=False):
                    changed.append(hotkey)
            except Exception as e:
                logger.error(f"[BUCKET_HISTORY] reconcile failed for {hotkey}: {e}", exc_info=True)
        if changed:
            try:
                self._bucket_history.save_to_disk(changed)
            except Exception as e:
                logger.error(f"[BUCKET_HISTORY] save failed for {len(changed)} hotkeys: {e}", exc_info=True)

    def _get_account(self, hotkey: str) -> MinerAccount | None:
        try:
            return self._miner_account_client.get_accounts([hotkey]).get(hotkey)
        except Exception as e:
            logger.warning(f"[BUCKET_HISTORY] could not read the account of {hotkey}: {e}")
            return None

    def _on_segment_open(self, segment: BucketSegment) -> None:
        account = self._get_account(segment.hotkey)
        if account is None:
            return
        try:
            segment.account_size = account.get_account_size()
            segment.start_balance = account.balance
        except Exception as e:
            logger.warning(f"[BUCKET_HISTORY] could not read the opening account of {segment.hotkey}: {e}")

    def _on_segment_close(self, segment: BucketSegment) -> None:
        """Copy what the segment traded into its directory as the stint ends.

        After an account switch the positions are already copied and archived and the final stats
        already taken (_record_wound_down_account), so only the snapshots are left to copy. On a hop
        that keeps the account the positions are still live: every position the segment traded is
        copied as it stands now, and the open ones carry on into the next segment.
        """
        hotkey = segment.hotkey
        try:
            positions = [
                p for p in self._position_client.get_positions_for_one_hotkey(hotkey)
                if p.open_ms < segment.end_ms and (p.is_open_position or p.close_ms >= segment.start_ms)
            ]
            self._bucket_history.write_positions(segment, positions)
        except Exception as e:
            logger.error(f"[BUCKET_HISTORY] could not copy the positions of {hotkey} into {segment.dir_name}: {e}")
        try:
            self._bucket_history.write_snapshots(segment, [
                s.to_dict() for s in read_all_snapshots(hotkey, self.running_unit_tests)
                if segment.contains(s.snapshot_ms)
            ])
        except Exception as e:
            logger.error(f"[BUCKET_HISTORY] could not copy the snapshots of {hotkey} into {segment.dir_name}: {e}")
        if segment.final_drawdown is None:
            segment.update_from_final_state(**self._final_account_state(hotkey))

    def _final_account_state(self, hotkey: str) -> dict:
        """The account's closing figures, as recorded on a segment or an in-segment account switch."""
        state = self.miner_states[hotkey]
        account = self._get_account(hotkey)
        return {
            "end_balance": getattr(account, "balance", None),
            "end_equity": getattr(account, "equity", None),
            "drawdown": state.drawdown.to_dict(),
            "pro_stats": state.pro_stats.to_dict(),
        }

    def _record_wound_down_account(self, hotkey: str, current_time_ms: int) -> None:
        """Called from _switch_account once the positions are closed, before they are archived and
        the account is reset: copies the closed positions into the segment being left and records
        the account's final stats on it. The bucket entry added after the switch closes it."""
        if not self._bucket_history.enabled:
            return
        try:
            segment = self._bucket_history.get_live_segment(hotkey)
            if segment is None:
                self._reconcile_bucket_history([hotkey], current_time_ms)
                segment = self._bucket_history.get_live_segment(hotkey)
            if segment is None:
                return
            self._bucket_history.write_positions(segment, self._position_client.get_positions_for_one_hotkey(hotkey))
            segment.update_from_final_state(**self._final_account_state(hotkey))
            self._bucket_history.save_segment(segment)
        except Exception as e:
            logger.error(f"[BUCKET_HISTORY] could not record the wound-down account of {hotkey}: {e}", exc_info=True)

    def _record_bucket_stats(self, hotkeys: list[str], current_time_ms: int) -> None:
        """Append the day's drawdown and ProStats row to each hotkey's live segment, once per UTC day."""
        if not self._bucket_history.enabled:
            return
        day_ms = TimeUtil.get_start_of_day_ms(current_time_ms)
        for hotkey in hotkeys:
            state = self.miner_states.get(hotkey)
            segment = self._bucket_history.get_live_segment(hotkey)
            if state is None or segment is None:
                continue
            pro_stats = state.pro_stats
            try:
                self._bucket_history.append_stats_row(segment, {
                    "time_ms": current_time_ms,
                    "bucket": state.current_phase.status,
                    **state.drawdown.to_dict(),
                    "calmar": pro_stats.calmar,
                    "max_drawdown": pro_stats.max_drawdown,
                    "daily_consistency": pro_stats.daily_consistency,
                    "trading_days": pro_stats.trading_days,
                    "soft_breach": state.is_soft_breach_latched(day_ms),
                }, day_ms)
            except Exception as e:
                logger.error(f"[BUCKET_HISTORY] could not write the stats row of {hotkey}: {e}")

    def get_bucket_statistics(self, hotkey: str, bucket: str | None = None, start_ms: int | None = None,
                              include: list[str] | None = None, limit: int | None = None) -> dict | None:
        """A miner's bucket segments, oldest first, with the sections asked for.

        bucket filters by bucket (any status BucketPhase.from_status reads, legacy names included),
        start_ms picks one segment. A live segment's positions and snapshots are read from the live
        state (the positions of its account, so a PRO_FUNDED segment includes those carried over);
        a closed one's from its segment directory. limit keeps the last snapshot and
        stats rows. Returns None when the miner has no segments.
        """
        include = set(include or ("summary",))
        segments = [s for s in self._bucket_history.get_segments(hotkey) if s.reverted_ms is None]
        if not segments:
            return None
        if bucket:
            bucket_value = BucketPhase.from_status(bucket).bucket.value
            segments = [s for s in segments if s.bucket == bucket_value]
        if start_ms is not None:
            segments = [s for s in segments if s.start_ms == start_ms]

        out = []
        for segment in segments:
            row = segment.to_dict()
            row["live"] = segment.is_live
            if "positions" in include:
                if segment.is_live:
                    # Live positions only ever hold the current account: earlier accounts are archived
                    row["positions"] = [
                        p.to_dict() for p in self._position_client.get_positions_for_one_hotkey(hotkey, sort_positions=True)
                    ]
                else:
                    row["positions"] = self._bucket_history.read_positions(segment)
            if "snapshots" in include:
                if segment.is_live:
                    snapshots = [s.to_dict() for s in read_all_snapshots(hotkey, self.running_unit_tests)
                                 if segment.contains(s.snapshot_ms)]
                    row["snapshots"] = snapshots[-limit:] if limit else snapshots
                else:
                    row["snapshots"] = self._bucket_history.read_snapshots(segment, limit)
            if "stats" in include:
                row["stats"] = self._bucket_history.read_stats(segment, limit)
            out.append(row)
        return {"hotkey": hotkey, "segments": out}

    def get_bucket_history_checkpoint_dict(self) -> dict:
        return self._bucket_history.to_checkpoint_dict()

    def sync_bucket_history(self, checkpoint_dict: dict) -> dict:
        return self._bucket_history.sync_from_checkpoint(checkpoint_dict)

    def get_drawdown_snapshot_fields(self, hotkeys: list[str] | None = None) -> dict[str, dict]:
        """The drawdown fields stamped onto account snapshots, per hotkey."""
        targets = self.miner_states.keys() if hotkeys is None else hotkeys
        out = {}
        for hotkey in targets:
            state = self.miner_states.get(hotkey)
            if state is None:
                continue
            out[hotkey] = {
                "intraday_drawdown_pct": state.drawdown.intraday_drawdown_pct,
                "eod_drawdown_pct": state.drawdown.eod_drawdown_pct,
                "eod_hwm": state.drawdown.eod_hwm,
            }
        return out

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
        # Follow the synced bucket entries with the segments they describe
        self._reconcile_bucket_history(list(self.miner_states))

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
        # The development hotkey has positions but is never a miner; prune any state it picked up
        hotkeys.discard(ValiConfig.DEVELOPMENT_HOTKEY)

        hotkeys_prune = []
        for hotkey, state in self.miner_states.items():
            if hotkey not in hotkeys:
                bucket = state.current_bucket
                if bucket.is_pro or bucket in (MinerBucket.ENTITY, MinerBucket.ELIMINATED, MinerBucket.SUBACCOUNT_CHALLENGE, MinerBucket.SUBACCOUNT_FUNDED):
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
        hotkey_to_bucket: dict[str, BucketPhase] = {}
        for hotkey in target_hotkeys:
            state = self.miner_states.get(hotkey)
            if state is None:
                continue
            hotkey_to_bucket[hotkey] = state.current_phase
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
        intraday_drawdown_threshold: float | None = None,
        eod_hwm_threshold: float | None = None,
        in_transition: bool = False,
    ) -> bool:
        """
        Set or update a miner's bucket information, replace_top to override most recent entry.
        drawdown_criteria, intraday_drawdown_threshold and eod_hwm_threshold are set only on first creation
        and ignored for existing states. in_transition marks a SUBACCOUNT_FUNDED entry as the
        transition week.

        Only persists to disk on first creation - both are write-once, and updates to
        an existing state are already covered by the caller's own batched save (see refresh()).

        Returns:
            True if this is a new miner, False if updating existing
        """
        with self._buckets_lock:
            if hotkey not in self.miner_states:
                self.miner_states[hotkey] = MinerBucketState(hotkey, [BucketEntry(bucket, start_time_ms, in_transition=in_transition)], drawdown_criteria=drawdown_criteria,
                                                          intraday_drawdown_threshold_override=intraday_drawdown_threshold,
                                                          eod_hwm_threshold_override=eod_hwm_threshold)
                is_new = True
                self._save_to_disk()
            else:
                self.miner_states[hotkey].add_bucket_entry(bucket, start_time_ms, replace_top=replace_top,
                                                           in_transition=in_transition)
                is_new = False
        self._reconcile_bucket_history([hotkey])
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

    def set_miner_drawdown_stats(
        self, hotkey: str, updates: dict, force: bool = False, now_ms: int | None = None
    ) -> Tuple[bool, str, dict | None]:
        """Overwrite drawdown fields on a miner's live stats and persist to disk.

        force is required for a change that deepens a drawdown
        past its threshold, or that deepens a pro account's ratcheted max_drawdown.
        """
        today_ms = TimeUtil.get_start_of_day_ms(now_ms if now_ms is not None else TimeUtil.now_in_millis())
        with self._buckets_lock:
            state = self.miner_states.get(hotkey)
            if not state:
                return False, f"{hotkey} not found in challenge period manager", None
            old = state.drawdown
            new = replace(old, **updates)

            checked_ms = new.last_eod_checked_ms
            if 'last_eod_checked_ms' in updates and (
                    TimeUtil.get_start_of_day_ms(checked_ms) != checked_ms or checked_ms > today_ms):
                return False, f"last_eod_checked_ms must be a UTC day boundary no later than today ({today_ms})", None
            if {'daily_open_equity', 'last_eod_equity'} & updates.keys() and checked_ms != today_ms:
                return False, ("daily_open_equity and last_eod_equity only hold for today: wait for today's "
                               f"EOD snapshot or set last_eod_checked_ms to {today_ms}"), None

            eod_hwm_floor = max(1.0, new.last_eod_equity, new.daily_open_equity or 0.0)
            if new.eod_hwm < eod_hwm_floor:
                return False, f"eod_hwm {new.eod_hwm} is below its floor {eod_hwm_floor} (1.0, last_eod_equity, daily_open_equity)", None

            if not force:
                rules = (
                    ('intraday', old.intraday_drawdown_pct, new.intraday_drawdown_pct, state.intraday_drawdown_threshold_pct),
                    ('eod', old.eod_drawdown_pct, new.eod_drawdown_pct, state.eod_drawdown_threshold_pct),
                    ('trailing', old.trailing_drawdown_pct, new.trailing_drawdown_pct, state.eod_drawdown_threshold_pct),
                )
                for rule, before, after, threshold in rules:
                    if after > threshold and after > before:
                        return False, (f"{rule} drawdown would go from {before:.2f}% to {after:.2f}%, past the "
                                       f"{threshold:.2f}% threshold; pass force to apply"), None
                if state.current_phase.is_pro_track and new.current_equity / new.eod_hwm < state.pro_stats.max_drawdown:
                    return False, ("eod_hwm would permanently deepen pro max_drawdown "
                                   f"({state.pro_stats.max_drawdown:.4f} -> {new.current_equity / new.eod_hwm:.4f}); pass force to apply"), None

            changes = ", ".join(f"{name} {getattr(old, name)} -> {value}" for name, value in updates.items())
            for name, value in updates.items():
                setattr(old, name, value)
            logger.warning(f"[CHALLENGE] {hotkey} drawdown stats manually set (force={force}): {changes}")
            self._save_to_disk()
            result = old.to_dict()
        return True, f"drawdown stats updated for {hotkey}", result

    def get_soft_breach_weeks(self, hotkey: str) -> dict[int, list[int]] | None:
        """Week start -> latched calmar breach days. None when the miner is unknown."""
        with self._buckets_lock:
            state = self.miner_states.get(hotkey)
            if state is None:
                return None
            return {week: list(days) for week, days in state.pro_stats.soft_breach_weeks.items()}

    def admin_set_soft_breach_week(self, hotkey: str, week_start_ms: int,
                                   days: list[int] | None) -> Tuple[bool, str]:
        """Set the breached days of one payout week, or drop the week when days is None. An empty
        list marks the week checked and clean. Takes effect on the next full penalty ledger rebuild."""
        if week_start_ms != TimeUtil.ms_at_start_of_week(week_start_ms):
            return False, f"week_start_ms must be Monday 00:00 UTC; got {week_start_ms}"
        week_end_ms = week_start_ms + 7 * ValiConfig.DAILY_MS
        for day in days or []:
            if day != TimeUtil.get_start_of_day_ms(day) or not week_start_ms <= day < week_end_ms:
                return False, f"{day} is not a UTC day start inside the week of {week_start_ms}"
        with self._buckets_lock:
            state = self.miner_states.get(hotkey)
            if state is None:
                return False, f"{hotkey} not found in challenge period manager"
            weeks = state.pro_stats.soft_breach_weeks
            before = weeks.get(week_start_ms)
            if days is None:
                weeks.pop(week_start_ms, None)
            else:
                weeks[week_start_ms] = sorted(set(days))
            logger.warning(f"[CHALLENGE] {hotkey} soft breach week {week_start_ms} manually set: "
                           f"{before} -> {weeks.get(week_start_ms)}")
            self._save_to_disk()
        return True, f"soft breach week {week_start_ms} updated for {hotkey}"

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

    def get_miner_buckets(self, hotkeys: list[str], timestamp_ms: int | None = None) -> dict[str, MinerBucket | None]:
        """Get buckets for multiple miners in one call."""
        return {hotkey: self.get_miner_bucket(hotkey, timestamp_ms) for hotkey in hotkeys}

    def get_miner_phase(self, hotkey, timestamp_ms: int | None = None) -> BucketPhase | None:
        """Get the bucket and SUBACCOUNT_FUNDED transition flag of a miner, optionally at a
        specific timestamp."""
        if hotkey not in self.miner_states:
            return None
        return self.miner_states[hotkey].phase(timestamp_ms)

    def get_miner_phases(self, hotkeys: list[str], timestamp_ms: int | None = None) -> dict[str, BucketPhase | None]:
        """Get bucket phases for multiple miners in one call."""
        return {hotkey: self.get_miner_phase(hotkey, timestamp_ms) for hotkey in hotkeys}

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

    def get_testing_miners(self) -> dict[str, int]:
        """CHALLENGE and subaccount challenge miners as {hotkey: start_time}."""
        return {hotkey: state.current_bucket_start_ms for hotkey, state in self.miner_states.items()
                if state.current_bucket == MinerBucket.CHALLENGE or state.current_phase.is_subaccount_challenge}

    def get_success_miners(self) -> dict[str, int]:
        """MAINCOMP and subaccount funded miners (including the SUBACCOUNT_FUNDED transition)
        as {hotkey: start_time}."""
        return {hotkey: state.current_bucket_start_ms for hotkey, state in self.miner_states.items()
                if state.current_bucket == MinerBucket.MAINCOMP or state.current_phase.is_subaccount_funded}

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

    def get_bucket_history(self, hotkey) -> list[dict] | None:
        """
        Full ordered history of bucket transitions for a miner, e.g.
        [{"bucket": "CHALLENGE", "start_time_ms": ...}, {"bucket": "MAINCOMP", "start_time_ms": ...}, ...]
        """
        state = self.miner_states.get(hotkey)
        if not state:
            return None
        return [entry.to_dict() for entry in state.entries]

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

    def get_pro_stats(self, synthetic_hotkey: str) -> dict | None:
        """
        Return pro promotion criteria for a synthetic hotkey for dashboard display.

        Returns None if the hotkey is not a pro account or has not been evaluated yet.
        """
        state = self.miner_states.get(synthetic_hotkey)
        if not state or not state.current_phase.is_pro_track:
            return None

        phase = state.current_phase
        asset_class = self._asset_selection_client.get_asset_selection(synthetic_hotkey)
        returns_threshold = phase.returns_threshold(asset_class) if not phase.in_transition else None
        return {
            **state.pro_stats.to_dict(),
            "in_transition": phase.in_transition,
            "calmar_threshold": phase.calmar_threshold,
            "daily_consistency_threshold": phase.daily_consistency_threshold,
            "minimum_trading_days": phase.minimum_trading_days,
            "returns_threshold": returns_threshold,
            "soft_breach_applies": phase.soft_breach_applies,
            "soft_breach": state.soft_breach,
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
        Expected format: {hk: {"entries": [...], "drawdown": {...}, "drawdown_criteria": ..., "intraday_drawdown_threshold_override": ..., "rank": ...}}
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
