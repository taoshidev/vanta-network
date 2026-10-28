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
from vali_objects.enums.account_type_enum import AccountType
from vali_objects.enums.drawdown_criteria_enum import DrawdownCriteria
from vali_objects.enums.miner_bucket_enum import BucketEntry, LEGACY_TRANSITION_BUCKET, MinerBucket
from vali_objects.plagiarism.plagiarism_client import PlagiarismClient
from vali_objects.miner_account.miner_account_client import MinerAccountClient
from vali_objects.miner_account.miner_account_manager import MinerAccount
from shared_objects.rpc.common_data_client import CommonDataClient
from entity_management.entity_utils import is_created_through_challenge, is_synthetic_hotkey
from entity_management.entity_client import EntityClient
from vali_objects.challenge_period.bucket_history import BucketHistoryStore, BucketSegment
from vali_objects.miner_account.account_snapshot import read_all_snapshots


@dataclass
class DrawdownStats:
    current_equity: float = 1.0
    current_balance: float = 1.0
    daily_open_equity: float | None = 1.0
    # Set only on accounts whose daily loss limit is measured from the day-open balance; None uses daily_open_equity
    daily_open_balance: float | None = None
    eod_hwm: float = 1.0
    last_eod_equity: float = 1.0
    last_eod_checked_ms: int | None = None
    # Auto-promotion candidates only: the worst live equity against eod_hwm, in mdd ratio form, ratcheted
    # every refresh as in the pro challenge, and when that tracking began
    max_drawdown: float = 1.0
    max_drawdown_since_ms: int | None = None

    @property
    def intraday_drawdown_pct(self) -> float:
        day_open = self.daily_open_balance if self.daily_open_balance is not None else self.daily_open_equity
        if not day_open:
            return 0.0
        return (1.0 - self.current_equity / day_open) * 100.0

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
class AutoPromotionStats:
    """An auto-promotion candidate's track record, measured as in the pro challenge and recomputed every
    refresh. max_drawdown is the worse of the live ratchet (DrawdownStats.max_drawdown) and the perf
    ledger replayed over the time before that tracking began."""
    calmar: float
    daily_consistency: float
    max_drawdown: float
    trading_days: int


@dataclass
class AutoPromotion:
    """How a standard funded subaccount was promoted automatically, and what it qualified on. Kept for the
    rest of its pro journey. Every field has a default so a record written by another version still loads.
    multiplier is the pro size over the standard size, which is the path's multiple unless the size was
    already set by the mothership."""
    FAST_TRACK = "fast_track"
    STEADY_TRACK = "steady_track"

    path: str = ""
    multiplier: float = 0.0
    pro_account_size: float = 0.0
    promoted_ms: int = 0
    equity_return: float = 0.0
    balance_return: float = 0.0
    calmar: float = 0.0
    daily_consistency: float = 1.0
    trading_days: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> 'AutoPromotion | None':
        if not d:
            return None
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys and v is not None})


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
    auto_promotion: AutoPromotion | None = None

    def __post_init__(self):
        if not self.entries:
            raise ValueError("BucketInfo must be initialized with BucketEntries")
        self.entries = sorted(self.entries, key=lambda x: x.start_time_ms)

    def add_bucket_entry(self, bucket: MinerBucket, time_ms: int, *, replace_top: bool = False) -> bool:
        entry = BucketEntry(bucket, time_ms)
        if self.current_bucket == bucket:
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
        if time_ms is None:
            return self.entries[-1].bucket
        for entry in reversed(self.entries):
            if entry.is_reverted:
                continue
            if entry.start_time_ms <= time_ms:
                return entry.bucket
        return MinerBucket.UNKNOWN

    def to_checkpoint_dict(self) -> dict:
        """Serialize full state (entries + drawdown + drawdown_criteria + drawdown threshold overrides +
        rank + pro_stats + auto_promotion) for on-disk checkpoint. The latched
        soft-breach days ride along inside pro_stats."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "drawdown": self.drawdown.to_dict(),
            "drawdown_criteria": self.drawdown_criteria.value,
            "intraday_drawdown_threshold_override": self.intraday_drawdown_threshold_override,
            "eod_hwm_threshold_override": self.eod_hwm_threshold_override,
            "rank": self.rank,
            "pro_stats": self.pro_stats.to_dict(),
            "auto_promotion": self.auto_promotion.to_dict() if self.auto_promotion else None,
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
            auto_promotion=AutoPromotion.from_dict(data.get("auto_promotion")),
        )

    def __str__(self) -> str:
        from datetime import datetime
        start = datetime.fromtimestamp(self.current_bucket_start_ms / 1000).strftime("%Y-%m-%d %H:%M")
        dd = self.drawdown
        criteria_str = self.drawdown_criteria.value
        return (
            f"{self.hotkey} {self.current_bucket.value} {start} rank={self.rank} criteria={criteria_str} idt_override={self.intraday_drawdown_threshold_override} "
            f"eod_override={self.eod_hwm_threshold_override} "
            f"equity={dd.current_equity:.4f} balance={dd.current_balance:.4f} daily_open={f'{dd.daily_open_equity:.4f}' if dd.daily_open_equity is not None else 'None'} "
            f"daily_open_balance={f'{dd.daily_open_balance:.4f}' if dd.daily_open_balance is not None else 'None'} | "
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
    def current_bucket_start_ms(self) -> int:
        return self.entries[-1].start_time_ms

    @property
    def current_bucket_entry(self) -> BucketEntry:
        return self.entries[-1]

    def _threshold_time_for(self, entry: BucketEntry) -> int | None:
        """For SUBACCOUNT_FUNDED, returns the SUBACCOUNT_CHALLENGE entry's start time
        so versioned thresholds (V0/V1) are selected based on when the miner registered,
        not when they were promoted. For all other buckets, returns the entry's own start."""
        if entry.bucket == MinerBucket.SUBACCOUNT_FUNDED:
            challenge_entry = next((e for e in self.entries if e.bucket == MinerBucket.SUBACCOUNT_CHALLENGE), None)
            return challenge_entry.start_time_ms if challenge_entry else None
        return entry.start_time_ms

    @property
    def _threshold_time_ms(self) -> int | None:
        return self._threshold_time_for(self.current_bucket_entry)

    def _intraday_threshold_for(self, bucket: MinerBucket, threshold_time_ms: int | None) -> float:
        """An intraday drawdown threshold chosen at creation applies in every bucket but PRO_FUNDED. Otherwise
        static accounts use a flat intraday threshold regardless of bucket or registration time, and pro buckets
        keep their own threshold even when the subaccount was created static."""
        if self.intraday_drawdown_threshold_override is not None and bucket != MinerBucket.PRO_FUNDED:
            return self.intraday_drawdown_threshold_override
        if self.drawdown_criteria == DrawdownCriteria.STATIC and not bucket.is_pro:
            return ValiConfig.SUBACCOUNT_STATIC_INTRADAY_DRAWDOWN_THRESHOLD
        return bucket.intraday_drawdown_threshold(threshold_time_ms)

    @property
    def intraday_drawdown_threshold(self):
        if self.current_bucket == MinerBucket.ELIMINATED:
            if len(self.entries) >= 2:
                prev_entry = self.entries[-2]
                return self._intraday_threshold_for(prev_entry.bucket, self._threshold_time_for(prev_entry))
            else:
                # Dummy threshold value for eliminated accounts to not raise Error
                return self._intraday_threshold_for(MinerBucket.SUBACCOUNT_CHALLENGE, None)
        return self._intraday_threshold_for(self.current_bucket, self._threshold_time_ms)

    def _eod_threshold_for(self, bucket: MinerBucket, threshold_time_ms: int | None) -> float:
        """An EOD threshold chosen at creation applies in every bucket but PRO_FUNDED."""
        if self.eod_hwm_threshold_override is not None and bucket != MinerBucket.PRO_FUNDED:
            return self.eod_hwm_threshold_override
        return bucket.eod_drawdown_threshold(threshold_time_ms)

    @property
    def intraday_drawdown_threshold_pct(self):
        return self.intraday_drawdown_threshold * 100

    @property
    def eod_drawdown_threshold(self):
        if self.current_bucket == MinerBucket.ELIMINATED:
            if len(self.entries) >= 2:
                prev_entry = self.entries[-2]
                return self._eod_threshold_for(prev_entry.bucket, self._threshold_time_for(prev_entry))
            else:
                # Dummy threshold value for eliminated accounts to not raise Error
                return self._eod_threshold_for(MinerBucket.SUBACCOUNT_CHALLENGE, None)
        return self._eod_threshold_for(self.current_bucket, self._threshold_time_ms)

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
    # The fast track's equity must hold on every refresh for this long, so a bad price tick cannot promote
    AUTO_PROMOTION_FAST_TRACK_HOLD_MS = 3 * 60 * 1000
    # How often a payout week that has ended but is not final yet is retried for freezing
    CLOSED_WEEKS_RETRY_MS = 60 * 60 * 1000

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
        # Transitions read from bucket entries written before the merge, moved onto the accounts on refresh
        self._legacy_transition_starts: dict[str, int] = {}
        self.miner_states: dict[str, MinerBucketState] = self._read_states_from_disk()
        # Per-bucket record of every stint, kept after the live state is wound down. Off while
        # backtesting, which never persists anything (see _save_to_disk)
        self._bucket_history = BucketHistoryStore(
            running_unit_tests=running_unit_tests, enabled=not is_backtesting or running_unit_tests)

        # Cached scores for MinerStatisticsManager
        self._cached_asset_softmaxed_scores: dict[MinerAssetClass, dict[str, float]] = {}
        self._cached_asset_competitiveness: dict[MinerAssetClass, float] = {}

        # Auto-promotion: subaccounts' creation buckets (fixed once created), the candidates seen on the
        # last refresh, the stats of those whose ledger covers the time before live tracking, since when
        # each candidate's equity has held the fast track's return, the payout week a failed move was last
        # logged in, the ledgers logged incomplete, and whether this validator is the mothership
        self._initial_buckets: dict[str, str | None] = {}
        self._auto_promotion_candidates: set[str] = set()
        self._auto_promotion_stats: dict[str, AutoPromotionStats] = {}
        self._fast_track_since_ms: dict[str, int] = {}
        self._move_failure_logged: dict[str, int] = {}
        self._incomplete_ledgers_logged: set[str] = set()
        self._mothership: bool | None = None
        # One lock per miner, held across a promotion of it (an auto-promotion's from its decision on)
        self._promotion_locks: dict[str, threading.RLock] = {}
        self._promotion_locks_guard = threading.Lock()
        # Closed payout weeks: the week each live segment is frozen through, and the last attempt per miner
        self._closed_weeks_checked: dict[str, int] = {}
        self._closed_weeks_attempt_ms: dict[str, int] = {}

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

        self._apply_legacy_transitions()
        accounts = self._miner_account_client.get_accounts(evaluation_hotkeys)
        transitions = self._miner_account_client.get_transition_starts(evaluation_hotkeys)
        ledgers = self._perf_ledger_client.filtered_ledger_for_scoring(evaluation_hotkeys)
        rank_positions, _ = self._position_client.filtered_positions_for_scoring(hotkeys=rank_hotkeys)
        timer.lap("fetch_eval_data")
        self._refresh_drawdown_cache(evaluation_hotkeys, accounts, ledgers, set(hk_to_first_order_time), current_time_ms)
        timer.lap("drawdown_cache")
        self._refresh_pro_stats(evaluation_hotkeys, ledgers, accounts, transitions)
        timer.lap("pro_stats")
        # Before the eliminations, so a failure here must not stop them: nothing is evaluated this refresh
        try:
            self._refresh_auto_promotion_stats(evaluation_hotkeys, ledgers, accounts, current_time_ms)
        except Exception as e:
            logger.error(f"[CHALLENGE] auto-promotion stats failed, skipping auto-promotion this refresh: {e}",
                         exc_info=True)
            self._auto_promotion_stats = {}
            # A refresh that could not check the fast track's equity starts its hold over
            self._fast_track_since_ms.clear()
        timer.lap("auto_promotion_stats")
        # Latch before any bucket move below: soft_breach reads the bucket the miner traded today
        self._latch_soft_breaches(evaluation_hotkeys, current_time_ms)
        timer.lap("latch_soft_breaches")
        self._record_bucket_stats(evaluation_hotkeys, current_time_ms)
        timer.lap("bucket_stats")
        self._freeze_closed_weeks(evaluation_hotkeys, current_time_ms)
        timer.lap("closed_weeks")
        self._refresh_rank_cache(rank_hotkeys, ledgers, rank_positions, accounts, asset_selections, current_time_ms)
        timer.lap("rank_cache")
        try:
            flat_transitions = self._flat_auto_promoted_transitions(transitions)
        except Exception as e:
            logger.error(f"[CHALLENGE] open position lookup failed, auto-promoted transitions keep their "
                         f"schedule this refresh: {e}", exc_info=True)
            flat_transitions = set()
        timer.lap("flat_transitions")

        eliminations: dict[str, EliminationReason] = {}
        demotions: dict[str, MinerBucket] = {}
        promotions: dict[str, MinerBucket] = {}  # the bucket each was evaluated in
        for hotkey, state in self.miner_states.items():
            if hotkey not in evaluation_hotkeys:
                if state.current_bucket == MinerBucket.UNKNOWN:
                    logger.warning(f"[CHALLENGE] {hotkey} bucket unknown")
                continue

            if reason := self._check_time(state, current_time_ms):
                eliminations[hotkey] = reason
                continue

            if state.current_bucket.is_pro:
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

            # The transition wind-down ends on schedule (see _check_transition_expiry), regardless of performance
            if self._check_transition_expiry(state, transitions.get(hotkey), current_time_ms,
                                             hotkey in flat_transitions):
                promotions[hotkey] = state.current_bucket
                continue

            _asset = asset_selections.get(hotkey)
            if _asset is None:
                logger.warning(f"[CHALLENGE] {hotkey} no asset selection, skipping evaluation")
                continue
            asset_class = MinerAssetClass(_asset)

            if self._check_demotion(state):
                demotions[hotkey] = MinerBucket.PROBATION
                continue

            returns_threshold = state.current_bucket.returns_threshold(asset_class)
            if hotkey == "5EPeU7Y8bqokEVf31ZWPZkP3F7Kv1v3ALuhnpp5T5Fvfjp85_87": # remove once eliminated or promoted
                returns_threshold = 0.08
            if self._check_promotion(state, returns_threshold, current_time_ms, hotkey in transitions):
                promotions[hotkey] = state.current_bucket
                continue

        timer.lap("evaluate")

        state_changed |= self.eliminate_hotkeys(eliminations, current_time_ms)
        state_changed |= self.demote_hotkeys(demotions, current_time_ms)
        state_changed |= self.promote_hotkeys(promotions, current_time_ms)

        if state_changed:
            self._sync_buckets_to_accounts(hotkeys=list({*eliminations, *demotions, *promotions}))
        timer.lap("apply_changes")

        # Last, so a failure here cannot hold up the moves above; each move syncs and saves itself
        try:
            auto_promotions = self._auto_promote(accounts, set(eliminations), current_time_ms)
        except Exception as e:
            logger.error(f"[CHALLENGE] auto-promotion pass failed, retrying next refresh: {e}", exc_info=True)
            auto_promotions = []
        timer.lap("auto_promote")

        self._save_to_disk()
        timer.lap("save_to_disk")

        counts = {}
        for state in self.miner_states.values():
            counts[state.current_bucket] = counts.get(state.current_bucket, 0) + 1
        snapshot = " | ".join(f"{b.value}={n}" for b, n in sorted(counts.items(), key=lambda x: x[0].value))
        logger.info(f"[CHALLENGE] snapshot: {snapshot} (total={len(self.miner_states)})")

        logger.info(f"[CHALLENGE] timings: {timer.summary()} "
                    f"(evaluated={len(evaluation_hotkeys)}, ranked={len(rank_hotkeys)})")

        return self._to_slack_message([*promotions, *auto_promotions])

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
    def _drawdown_reason(bucket: MinerBucket, rule: str) -> EliminationReason:
        """Map (bucket, drawdown rule) to the elimination reason for that account tier.
        Pro buckets only ever pair with INTRADAY and EOD; the static rules are non-pro only."""
        if bucket == MinerBucket.PRO_FUNDED:
            tier = "PRO_FUNDED_PERIOD"
        elif bucket.is_pro:
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
            return cls._drawdown_reason(state.current_bucket, "INTRADAY")
        elif state.drawdown.intraday_drawdown_pct > state.intraday_drawdown_threshold_pct * 0.75:
            logger.info(f"[CHALLENGE] near intraday drawdown {state.intraday_drawdown_threshold_pct}%: {state}")
        return None

    @classmethod
    def _check_eod_drawdown(cls, state: MinerBucketState) -> EliminationReason | None:
        if state.drawdown.eod_drawdown_pct > state.eod_drawdown_threshold_pct:
            logger.warning(f"[CHALLENGE] BREACH EOD drawdown {state.eod_drawdown_threshold_pct}%: {state}")
            return cls._drawdown_reason(state.current_bucket, "EOD")

        # Live equity already past the limit: tonight's latch breaches unless the miner recovers
        if state.drawdown.trailing_drawdown_pct > state.eod_drawdown_threshold_pct:
            logger.info(f"[CHALLENGE] near trailing EOD with current equity {state.eod_drawdown_threshold_pct}%: {state}")
        elif state.drawdown.eod_drawdown_pct > state.eod_drawdown_threshold_pct * 0.75:
            logger.info(f"[CHALLENGE] near EOD drawdown {state.eod_drawdown_threshold_pct}%: {state}")

        return None

    @classmethod
    def _check_static_drawdown(cls, state: MinerBucketState) -> EliminationReason | None:
        bucket = state.current_bucket
        if bucket.is_pro:
            return None
        threshold_pct = ValiConfig.SUBACCOUNT_STATIC_DRAWDOWN_THRESHOLD * 100
        if state.drawdown.static_drawdown_pct > threshold_pct:
            logger.warning(f"[CHALLENGE] BREACH static drawdown {threshold_pct}%: {state}")
            return cls._drawdown_reason(bucket, "STATIC")
        elif state.drawdown.static_drawdown_pct > threshold_pct * 0.75:
            logger.info(f"[CHALLENGE] near static drawdown {threshold_pct}%: {state}")
        return None

    def _flat_auto_promoted_transitions(self, transitions: dict[str, int]) -> set[str]:
        """Auto-promoted subaccounts in the SUBACCOUNT_FUNDED transition with no open position. The
        transition cannot open one, so nothing is left to wind down."""
        hotkeys = [hotkey for hotkey in transitions
                   if hotkey in self.miner_states and self.miner_states[hotkey].auto_promotion is not None]
        if not hotkeys:
            return set()
        open_positions = self._position_client.get_positions_for_hotkeys(hotkeys, only_open_positions=True)
        return {hotkey for hotkey in hotkeys if not open_positions.get(hotkey)}

    @staticmethod
    def _check_transition_expiry(state: MinerBucketState, transition_start_ms: int | None, current_time_ms: int,
                                 flat: bool = False) -> bool:
        """True once the SUBACCOUNT_FUNDED transition (started at transition_start_ms, None when not in it)
        has run its course: after an auto-promotion, once the account is flat or AUTO_PROMOTION_TRANSITION_DAYS
        have passed, else past the first Monday 00:00 UTC after the miner entered it"""
        if transition_start_ms is None:
            return False
        if state.auto_promotion is not None and flat:
            logger.info(f"[CHALLENGE] transition ended early, no open positions: {state}")
            return True
        if state.auto_promotion is not None:
            promoted_ms = state.auto_promotion.promoted_ms or transition_start_ms
            expired = current_time_ms >= promoted_ms + ValiConfig.AUTO_PROMOTION_TRANSITION_DAYS * ValiConfig.DAILY_MS
        else:
            expired = (TimeUtil.ms_at_start_of_week(current_time_ms)
                       > TimeUtil.ms_at_start_of_week(transition_start_ms))
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
    def _check_promotion(state: MinerBucketState, returns_threshold: float, current_time_ms: int,
                         in_transition: bool = False) -> bool:
        # The transition leaves on its week boundary (_check_transition_expiry), never on returns
        if in_transition:
            return False

        if state.current_bucket == MinerBucket.CHALLENGE:
            if current_time_ms - state.current_bucket_start_ms < ValiConfig.CHALLENGE_PERIOD_MINIMUM_MS:
                return False

        bucket = state.current_bucket
        if bucket.next_bucket is None:
            return False

        minimum_trading_days = bucket.minimum_trading_days
        if minimum_trading_days is not None and state.pro_stats.trading_days < minimum_trading_days:
            return False

        calmar_threshold = bucket.calmar_threshold
        if calmar_threshold is not None and state.pro_stats.calmar < calmar_threshold:
            return False

        consistency_threshold = bucket.daily_consistency_threshold
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

            is_pro = state.current_bucket.is_pro
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
            # intraday_drawdown_pct always holds the equity-vs-day-open number (day-open balance or equity)
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

    def _switch_account(self, hotkey: str, target: MinerBucket, current_time_ms: int) -> bool:
        """Wind down the account a miner is leaving. Run whenever a bucket change also changes
        the account size, so the new account's performance is tracked from scratch.

        Returns False when the account size could not be pointed at the target bucket. Nothing is
        wound down in that case: the caller must leave the miner in their current bucket rather
        than strand them on the new one with the old size and no history.
        """
        # Point the subaccount at the size the target bucket trades before resetting the account. An
        # auto-promoted subaccount goes onto its pro account free of the promotion fee.
        if is_synthetic_hotkey(hotkey):
            state = self.miner_states.get(hotkey)
            waive_fee = state is not None and state.auto_promotion is not None
            success, message = self._entity_client.apply_bucket_account_size(
                hotkey, target, waive_promotion_fee=waive_fee)
            if not success:
                logger.error(
                    f"[CHALLENGE] {hotkey} bucket change to {target.value} ABORTED - "
                    f"account size not updated: {message}"
                )
                return False

        self._close_out_account(hotkey, current_time_ms)
        # Keep the closed positions and final stats of the account in its bucket segment before
        # anything below archives or resets them
        self._record_wound_down_account(hotkey, current_time_ms)
        self._record_segment_payout(hotkey, current_time_ms)
        # Reset account fields (PnL, capital used, borrowed amount, interest), ending any transition
        self._miner_account_client.reset_account(hotkey, target)
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
            # 1ms before the bucket move, so the close falls inside the segment being left and is paid by its record
            close_time_ms=current_time_ms - 1,
            order_source=OrderSource.SUBACCOUNT_PROMOTION
        )
        self._limit_order_client.cancel_limit_order(hotkey, None, "ALL", current_time_ms)

    def _leave_bucket(self, hotkey: str, target: MinerBucket, current_time_ms: int) -> bool:
        """Wind down what the miner is leaving for target: a full account switch, or just closing
        the positions on a move into or out of an earning bucket that keeps the account. False when
        the switch failed."""
        if target.switches_account:
            return self._switch_account(hotkey, target, current_time_ms)
        if target.closes_positions_from(self.miner_states[hotkey].current_bucket):
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
            if not self._leave_bucket(hotkey, target_bucket, current_time_ms):
                continue

            with self._buckets_lock:
                state_changed |= self.miner_states[hotkey].add_bucket_entry(target_bucket, current_time_ms)
            self._reconcile_bucket_history([hotkey], current_time_ms)
        return state_changed

    def promote_hotkeys(self, promotions: dict[str, MinerBucket], current_time_ms: int) -> bool:
        """Promote miners to next tier. promotions maps each miner to the bucket it was evaluated in;
        one that has left it since, e.g. through an entity promotion, is not promoted again. A miner in
        its transition week moves to PRO_CHALLENGE_FUNDED."""
        if promotions:
            logger.info(f"[CHALLENGE] promoting {len(promotions)} miners")

        state_changed = False
        for hotkey, evaluated_bucket in promotions.items():
            # The lock an entity promotion holds, so one cannot land between the check and the move
            with self._promotion_lock(hotkey):
                state = self.miner_states.get(hotkey)
                if state is None or state.current_bucket != evaluated_bucket:
                    logger.info(f"[CHALLENGE] {hotkey} left {evaluated_bucket.value} after it was evaluated, "
                                f"not promoting: {state}")
                    continue
                if self._transition_start(hotkey) is not None:
                    target = MinerBucket.PRO_CHALLENGE_FUNDED
                else:
                    target = evaluated_bucket.next_bucket
                if target is None:
                    logger.warning(f"[CHALLENGE] no next bucket for promotion: {state}")
                    continue

                logger.info(f"[CHALLENGE] promoting to {target.value}: {state}")

                # Leave the miner where they are when the account switch fails; the next refresh retries
                if not self._leave_bucket(hotkey, target, current_time_ms):
                    continue

                with self._buckets_lock:
                    state_changed |= self.miner_states[hotkey].add_bucket_entry(target, current_time_ms)
                self._reconcile_bucket_history([hotkey], current_time_ms)

        return state_changed

    @staticmethod
    def _promotion_target(bucket: MinerBucket, in_transition: bool) -> tuple[MinerBucket, bool] | None:
        """(bucket, in_transition) a self-service promotion moves to: SUBACCOUNT_FUNDED enters its
        transition week, and the transition leaves for PRO_CHALLENGE_FUNDED."""
        if in_transition:
            return MinerBucket.PRO_CHALLENGE_FUNDED, False
        if bucket == MinerBucket.SUBACCOUNT_FUNDED:
            return MinerBucket.SUBACCOUNT_FUNDED, True
        target = bucket.promotion_target
        return (target, False) if target is not None else None

    @staticmethod
    def _describe(bucket: MinerBucket, in_transition: bool) -> str:
        return f"{bucket.value} (in transition)" if in_transition else bucket.value

    def promote_subaccount(
        self, hotkey: str, current_time_ms: int, pro_account_size: float | None = None,
        from_bucket: MinerBucket | None = None, from_in_transition: bool = False,
    ) -> tuple[bool, str]:
        """Promote a subaccount one step up the pro track on the entity miner's request, or by
        auto-promotion (_auto_promote).

        The only moves allowed are the three hops in _promotion_target; a subaccount in
        any other bucket is rejected, as is an Instant Funded subaccount in SUBACCOUNT_FUNDED.
        pro_account_size is the size the entity asked for: entering the pro track needs one, and None
        keeps the size already recorded (see EntityManager.apply_bucket_account_size, which rejects a
        subaccount that has neither). from_bucket and from_in_transition, when given, reject the call
        unless the subaccount is still there, so a caller that decided on one hop never makes the next.
        A subaccount promoted automatically may leave the SUBACCOUNT_FUNDED transition early, but only
        on the size the network granted and free of the promotion fee: a request that sends a
        pro_account_size is rejected.

        Only a target with switches_account wipes trading state, which is the two hops landing on a
        pro account (PRO_CHALLENGE_DIRECT, and PRO_CHALLENGE_FUNDED):
        positions closed, limit orders cancelled, ledgers restarted. The SUBACCOUNT_FUNDED
        transition is a wind-down week on the standard account, so the hop into it keeps the
        positions, limit orders and ledgers as they are and only sweeps the entry orders (see
        admin_set_bucket).

        Runs under the subaccount's promotion lock, so two promotions of it never interleave their
        sizing and bucket moves. Without from_bucket the state at the time of the call is used, so a
        duplicate request that waited on the lock is rejected instead of making the next hop.
        """
        if from_bucket is None and hotkey in self.miner_states:
            from_bucket = self.miner_states[hotkey].current_bucket
            from_in_transition = self._transition_start(hotkey) is not None
        with self._promotion_lock(hotkey):
            return self._promote_subaccount(hotkey, current_time_ms, pro_account_size, from_bucket, from_in_transition)

    def _promote_subaccount(self, hotkey: str, current_time_ms: int, pro_account_size: float | None,
                            from_bucket: MinerBucket | None, from_in_transition: bool) -> tuple[bool, str]:
        """promote_subaccount, run under the subaccount's promotion lock."""
        state = self.miner_states.get(hotkey)
        if state is None:
            return False, f"{hotkey} not found in challenge period manager"

        bucket = state.current_bucket
        in_transition = self._transition_start(hotkey) is not None
        if from_bucket is not None and (bucket, in_transition) != (from_bucket, from_in_transition):
            return False, (f"{hotkey} is in {self._describe(bucket, in_transition)}, "
                           f"no longer {self._describe(from_bucket, from_in_transition)}")
        auto_promoted = in_transition and state.auto_promotion is not None
        if auto_promoted and pro_account_size is not None:
            return False, (f"{hotkey} was promoted automatically: its pro account size is set by the network "
                           f"and cannot be changed")
        target = self._promotion_target(bucket, in_transition)
        if target is None:
            return False, f"{hotkey} cannot be promoted out of {self._describe(bucket, in_transition)}"
        target_bucket, target_in_transition = target
        # Checked before sizing so a refused move has nothing to roll back
        can_set, reason = self.can_admin_set_bucket(hotkey, target_bucket, in_transition=target_in_transition)
        if not can_set:
            return False, reason

        logger.info(f"[CHALLENGE] promotion to {self._describe(*target)} requested "
                    f"(pro_account_size={pro_account_size}): {state}")

        # Record the granted pro size first: the account switch reads it back to size the new account.
        # Every pro bucket past the transition trades the pro size
        sizing_snapshot = self._entity_client.snapshot_bucket_account_size(hotkey)
        success, message = self._entity_client.apply_bucket_account_size(
            hotkey, target_bucket, pro_account_size, in_transition=target_in_transition,
            waive_promotion_fee=auto_promoted
        )
        if not success:
            logger.warning(f"[CHALLENGE] promotion rejected for {hotkey}: {message}")
            return False, message

        try:
            success, message = self.admin_set_bucket(hotkey, target_bucket, current_time_ms,
                                                     in_transition=target_in_transition)
        except Exception:
            # admin_set_bucket can raise after the move has landed: its entry-order sweep
            # and its disk write both run past that point. The sizing belongs to whichever side of
            # that line the miner ended up on, so give it back only if the bucket did not move -
            # restoring it after a move would strand a pro bucket on the standard size.
            state = self.miner_states.get(hotkey)
            if state is None or (state.current_bucket, self._transition_start(hotkey) is not None) != target:
                self._restore_bucket_account_size(hotkey, sizing_snapshot)
            else:
                logger.error(f"[CHALLENGE] {hotkey} reached {self._describe(*target)} before failing; "
                             f"keeping the sizing that bucket trades")
            raise
        if not success:
            # Every False return is from before the move (can_admin_set_bucket, the account
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
        in_transition = in_transition and bucket == MinerBucket.SUBACCOUNT_FUNDED
        state = self.miner_states.get(hotkey)
        if state is None:
            return False, f"{hotkey} not found in challenge period manager"
        if (state.current_bucket, self._transition_start(hotkey) is not None) == (bucket, in_transition):
            return False, f"{hotkey} is already in {self._describe(bucket, in_transition)}"
        # Instant Funded accounts never enter the pro track through the transition
        if in_transition and self._entity_client.is_instant_funded(hotkey):
            return False, f"{hotkey} is Instant Funded and cannot move to {self._describe(bucket, in_transition)}"
        return True, ""

    def admin_set_bucket(self, hotkey: str, bucket: MinerBucket, current_time_ms: int,
                         in_transition: bool = False) -> tuple[bool, str]:
        """Move a miner into an arbitrary bucket. Runs the same account switch as an organic
        promotion when the target changes the account size. in_transition only applies to
        SUBACCOUNT_FUNDED: set, it enters the transition week on the standard account."""
        can_set, reason = self.can_admin_set_bucket(hotkey, bucket, in_transition)
        if not can_set:
            return False, reason

        in_transition = in_transition and bucket == MinerBucket.SUBACCOUNT_FUNDED
        target = self._describe(bucket, in_transition)
        state = self.miner_states[hotkey]
        was_in_transition = self._transition_start(hotkey) is not None
        logger.info(f"[CHALLENGE] admin moving to {target}: {state}")

        # Entering the transition keeps the standard account
        switches_account = bucket.switches_account and not in_transition
        if switches_account and not self._switch_account(hotkey, bucket, current_time_ms):
            return False, f"{hotkey} account size update failed, bucket unchanged"

        # PRO_FUNDED keeps the account, so it never runs the switch that sizes one. An organic
        # promotion arrives there already on the pro size, but an admin can move a subaccount
        # straight in from a standard bucket, and a pro bucket has to trade the pro size.
        if not switches_account and bucket.is_pro and is_synthetic_hotkey(hotkey):
            sized, message = self._entity_client.apply_bucket_account_size(hotkey, bucket)
            if not sized:
                logger.error(
                    f"[CHALLENGE] {hotkey} bucket change to {target} ABORTED - "
                    f"account size not updated: {message}"
                )
                return False, f"{hotkey} account size update failed, bucket unchanged: {message}"

        if not switches_account and bucket.closes_positions_from(state.current_bucket):
            self._close_out_account(hotkey, current_time_ms)
            self._record_segment_payout(hotkey, current_time_ms)

        # The account holds the transition; set before the bucket entry so the order gate never lags it
        if bucket == MinerBucket.SUBACCOUNT_FUNDED and in_transition != was_in_transition:
            self._miner_account_client.set_transition(hotkey, current_time_ms if in_transition else None)

        with self._buckets_lock:
            state.add_bucket_entry(bucket, current_time_ms)
        self._reconcile_bucket_history([hotkey], current_time_ms)

        self._sync_buckets_to_accounts(hotkeys=[hotkey])

        if in_transition and not was_in_transition:
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
            prev_bucket = miner_state.entries[-2].bucket
            restores_pro_account = prev_bucket == MinerBucket.PRO_FUNDED

            # The ELIMINATED entry stays as the record that it happened, but is marked reverted so
            # replaying this history never stamps a checkpoint inside the span with it
            if restores_pro_account:
                miner_state.entries[-1].reverted_ms = now_ms
            miner_state.add_bucket_entry(prev_bucket, now_ms)
            logger.info(
                f"[CHALLENGE] reverted elimination, restored to {miner_state.current_bucket.value}: {miner_state}"
            )

        self._sync_buckets_to_accounts(hotkeys=[hotkey])
        # The account kept its transition through the elimination; it restarts with the restored bucket
        if self._transition_start(hotkey) is not None:
            self._miner_account_client.set_transition(hotkey, now_ms)
        self._reconcile_bucket_history([hotkey], now_ms)
        self._reset_drawdown_stats_cache(hotkey, preserve_eod=restores_pro_account)
        self._save_to_disk()
        return True

    # ==================== Drawdown/Rank Refresh methods ====================

    @staticmethod
    def _parse_eod_checkpoints(ledger: PerfLedger, now_ms: int, account_size: float | None = None
                               ) -> tuple[float, float | None, float, int | None, float | None]:
        """
        Parse midnight checkpoints from a ledger.
        Returns (last_eod, daily_open_equity, eod_hwm, last_eod_checked_ms, daily_open_balance).
        """
        midnight_cps = [cp for cp in ledger.cps if cp.last_update_ms % 86400000 == 0 and cp.equity_ret > 0]
        last_eod = midnight_cps[-1].equity_ret if midnight_cps else 1.0
        last_eod_checked_ms = midnight_cps[-1].last_update_ms if midnight_cps else None
        today_midnight_ms = (now_ms // 86400000) * 86400000
        today_open_cp = next((cp for cp in midnight_cps if cp.last_update_ms == today_midnight_ms), None)
        daily_open_equity = today_open_cp.equity_ret if today_open_cp else (None if last_eod else 1.0)
        eod_hwm = max(max(cp.equity_ret for cp in midnight_cps), 1.0) if midnight_cps else 1.0
        daily_open_balance = (today_open_cp.equity_ret - today_open_cp.unrealized_pnl / account_size
                              if today_open_cp and account_size else None)
        return last_eod, daily_open_equity, eod_hwm, last_eod_checked_ms, daily_open_balance

    @staticmethod
    def _daily_open_balance(account: MinerAccount, now_ms: int) -> float | None:
        """Today's opening balance over account size, from the daily open snapshot. None when no snapshot
        holds today's open, which leaves the day on the day-open equity."""
        snapshot = account.daily_open_snapshot
        if (snapshot and snapshot.account_size
                and TimeUtil.get_start_of_day_ms(snapshot.snapshot_ms) == TimeUtil.get_start_of_day_ms(now_ms)):
            return snapshot.balance / snapshot.account_size
        return None

    def _daily_loss_from_balance(self, hotkey: str, account: MinerAccount) -> bool:
        """Standard subaccounts, created through SUBACCOUNT_CHALLENGE and not HL-linked, measure the daily
        loss limit from the day-open balance while in the standard buckets. Every other account keeps the
        day-open equity."""
        if not self.miner_states[hotkey].current_bucket.daily_loss_from_balance or account.is_hl_linked:
            return False
        info = self._entity_client.get_subaccount_info_for_synthetic(hotkey)
        return bool(info) and is_created_through_challenge(info.get("initial_bucket"))

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
            try:
                from_balance = self._daily_loss_from_balance(hotkey, account)
            except Exception as e:
                # Keep the account on the reference it has rather than abort every miner's drawdown check
                from_balance = self.miner_states[hotkey].drawdown.daily_open_balance is not None
                logger.warning(f"[CHALLENGE] {hotkey} entity lookup failed, keeping its daily loss reference: {e}")
            # Held so a manual set_miner_drawdown_stats cannot land between reading and writing eod_hwm
            with self._buckets_lock:
                existing = self.miner_states[hotkey].drawdown
                existing.current_equity = current_equity
                existing.current_balance = current_balance
                if not from_balance:
                    existing.daily_open_balance = None

                # EOD fields are locked in for today once the snapshot has been captured
                if existing.last_eod_checked_ms == current_day_open_ms:
                    continue

                # Use daily open snapshot from miner account as default and fallback to perf ledgers
                snapshot = account.daily_open_snapshot
                if snapshot and TimeUtil.get_start_of_day_ms(snapshot.snapshot_ms) == current_day_open_ms:
                    last_eod_equity = daily_open_equity = snapshot.equity_return
                    eod_hwm = max(existing.eod_hwm, daily_open_equity)
                    last_eod_checked_ms = current_day_open_ms
                    daily_open_balance = self._daily_open_balance(account, now_ms)
                else:
                    ledger = ledgers.get(hotkey)
                    if ledger is None:
                        logger.warning(f"[CHALLENGE] {hotkey} missing ledger, skipping drawdown cache")
                        continue
                    last_eod_equity, daily_open_equity, eod_hwm, last_eod_checked_ms, daily_open_balance = (
                        self._parse_eod_checkpoints(ledger, now_ms, account.account_size))

                if last_eod_checked_ms == current_day_open_ms:
                    existing.current_equity = current_equity
                    existing.current_balance = current_balance
                    existing.last_eod_equity = last_eod_equity
                    existing.daily_open_equity = daily_open_equity
                    if from_balance:
                        existing.daily_open_balance = daily_open_balance
                        if daily_open_balance is None:
                            logger.warning(f"[CHALLENGE] {hotkey} has no day-open balance, today's daily loss "
                                           f"limit is measured from the day-open equity: {self.miner_states[hotkey]}")
                    existing.eod_hwm = max(existing.eod_hwm, last_eod_equity, eod_hwm)
                    existing.last_eod_checked_ms = last_eod_checked_ms

    def _refresh_pro_stats(self, hotkeys: list[str], ledgers: dict[str, PerfLedger],
                           accounts: dict[str, MinerAccount], transitions: dict[str, int]) -> None:
        for hotkey in hotkeys:
            state = self.miner_states[hotkey]
            # The pro track: the pro buckets, and the transition week leading to them
            if not (state.current_bucket.is_pro_track or hotkey in transitions):
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

    def _refresh_auto_promotion_stats(self, hotkeys: list[str], ledgers: dict[str, PerfLedger],
                                      accounts: dict[str, MinerAccount], current_time_ms: int) -> None:
        """Track every auto-promotion candidate's drawdown and measure its track record as in the pro
        challenge.

        From the first refresh that sees a candidate, its worst live equity against eod_hwm is ratcheted
        in DrawdownStats, as the pro challenge ratchets its own. The time before has no live record, so it
        is replayed from the perf ledger's per-checkpoint lowest equity, and the worse of the two is the
        drawdown. A candidate whose ledger cannot replay that time gets no stats, so it is not evaluated.

        Also times how long each candidate's equity has held the fast track's return: any refresh below
        it starts the hold over.
        """
        candidates = set()
        stats = {}
        transitions = self._miner_account_client.get_transition_starts(hotkeys)
        for hotkey in hotkeys:
            state = self.miner_states[hotkey]
            account = accounts.get(hotkey)
            if (state.current_bucket != MinerBucket.SUBACCOUNT_FUNDED or hotkey in transitions
                    or not self._is_auto_promotion_candidate(hotkey, account)):
                continue
            candidates.add(hotkey)

            drawdown = state.drawdown
            with self._buckets_lock:
                if drawdown.max_drawdown_since_ms is None:
                    drawdown.max_drawdown_since_ms = current_time_ms
                if drawdown.eod_hwm > 0:
                    drawdown.max_drawdown = min(drawdown.max_drawdown, 1.0,
                                                drawdown.current_equity / drawdown.eod_hwm)
            if drawdown.current_equity - 1.0 >= ValiConfig.AUTO_PROMOTION_FAST_TRACK_EQUITY_RETURN:
                self._fast_track_since_ms.setdefault(hotkey, current_time_ms)
            else:
                self._fast_track_since_ms.pop(hotkey, None)

            ledger = ledgers.get(hotkey)
            if ledger is None or account.account_size <= 0:
                continue
            history = LedgerUtils.eod_hwm_drawdown(ledger, before_ms=drawdown.max_drawdown_since_ms)
            if history is None:
                if hotkey not in self._incomplete_ledgers_logged:
                    self._incomplete_ledgers_logged.add(hotkey)
                    logger.warning(f"[CHALLENGE] {hotkey} ledger does not record every checkpoint's lowest equity; "
                                   f"not evaluated for auto-promotion until the ledger is regenerated")
                continue

            max_drawdown = min(drawdown.max_drawdown, history)
            log_returns = LedgerUtils.daily_return_log(ledger)
            stats[hotkey] = AutoPromotionStats(
                calmar=Metrics.all_time_calmar(
                    LedgerUtils.realized_return(ledger, account.account_size), max_drawdown),
                daily_consistency=Metrics.return_consistency(log_returns),
                max_drawdown=max_drawdown,
                trading_days=len(log_returns),
            )

        self._auto_promotion_candidates = candidates
        self._auto_promotion_stats = stats
        # Forget accounts that are no longer candidates
        for cache in (self._fast_track_since_ms, self._move_failure_logged):
            for hotkey in cache.keys() - candidates:
                del cache[hotkey]
        self._incomplete_ledgers_logged &= candidates
        with self._buckets_lock:
            known = set(self.miner_states)
        for hotkey in self._initial_buckets.keys() - known:
            del self._initial_buckets[hotkey]

    def _is_auto_promotion_candidate(self, hotkey: str, account: MinerAccount | None) -> bool:
        """A standard subaccount that can be auto-promoted into its SUBACCOUNT_FUNDED transition: created through
        SUBACCOUNT_CHALLENGE (not Instant Funded) and not HL-linked. A failed lookup reads as not
        a candidate until the next refresh."""
        try:
            if account is None or account.is_hl_linked:
                return False
            if hotkey not in self._initial_buckets:
                info = self._entity_client.get_subaccount_info_for_synthetic(hotkey)
                if not info:
                    return False
                self._initial_buckets[hotkey] = info.get("initial_bucket")
            return is_created_through_challenge(self._initial_buckets[hotkey])
        except Exception as e:
            logger.warning(f"[CHALLENGE] {hotkey} auto-promotion candidacy lookup failed, skipping this refresh: {e}")
            return False

    def _auto_promote(self, accounts: dict[str, MinerAccount], eliminating: set[str],
                      current_time_ms: int) -> list[str]:
        """Move every auto-promotion candidate that qualifies into its SUBACCOUNT_FUNDED transition. Only
        candidates with stats are evaluated (their drawdown covers the whole funded account), never one
        this refresh is eliminating, even if that elimination failed, and a subaccount is promoted
        automatically at most once. Returns the hotkeys moved."""
        promoted = []
        for hotkey, stats in list(self._auto_promotion_stats.items()):
            state = self.miner_states.get(hotkey)
            account = accounts.get(hotkey)
            if (hotkey in eliminating or state is None or state.current_bucket != MinerBucket.SUBACCOUNT_FUNDED
                    or state.auto_promotion or account is None or account.account_size <= 0):
                continue
            since_ms = self._fast_track_since_ms.get(hotkey)
            held = since_ms is not None and current_time_ms - since_ms >= self.AUTO_PROMOTION_FAST_TRACK_HOLD_MS
            qualified = self._auto_promotion_path(stats, held, state.drawdown.current_return)
            if qualified is None:
                continue
            with self._promotion_lock(hotkey):
                if self._move_to_auto_promotion(hotkey, account, stats, qualified, current_time_ms):
                    promoted.append(hotkey)
        return promoted

    def _move_to_auto_promotion(self, hotkey: str, account: MinerAccount, stats: AutoPromotionStats,
                                qualified: tuple[str, int], current_time_ms: int) -> bool:
        """Move one qualifying candidate into its SUBACCOUNT_FUNDED transition. Called with its promotion lock held.
        True when it moved.

        Only the mothership chooses the pro size (the path's multiple of the standard account), and its
        sizing reaches the other validators by broadcast, or by the nightly sync when a broadcast is
        missed. Every other validator moves only once that size is on its record, and follows it.
        """
        # Checked again under the lock: the entity may have promoted it since the decision
        state = self.miner_states.get(hotkey)
        if (state is None or state.current_bucket != MinerBucket.SUBACCOUNT_FUNDED or state.auto_promotion
                or self._transition_start(hotkey) is not None):
            return False
        path, multiple = qualified
        try:
            info = self._entity_client.get_subaccount_info_for_synthetic(hotkey) or {}
            if info.get("account_type") == AccountType.PRO.value and info.get("pro_account_size"):
                pro_account_size, requested_size = info["pro_account_size"], None
            elif self._is_mothership():
                pro_account_size = requested_size = multiple * account.account_size
            else:
                return False  # waiting for the mothership's pro size
        except Exception as e:
            self._log_move_failure(hotkey, current_time_ms, f"subaccount lookup failed: {e}")
            return False

        record = AutoPromotion(path=path, multiplier=pro_account_size / account.account_size,
                               pro_account_size=pro_account_size, promoted_ms=current_time_ms,
                               equity_return=state.drawdown.current_equity - 1.0,
                               balance_return=state.drawdown.current_balance - 1.0, calmar=stats.calmar,
                               daily_consistency=stats.daily_consistency, trading_days=stats.trading_days)
        # Recorded before the move so it is saved to disk with it
        with self._buckets_lock:
            state.auto_promotion = record
        try:
            moved, message = self.promote_subaccount(hotkey, current_time_ms, requested_size,
                                                     from_bucket=MinerBucket.SUBACCOUNT_FUNDED)
        except Exception as e:
            # It can raise after its own transition has started (see promote_subaccount); one started
            # by anyone else (the entity's own request) carries another time
            moved = self._transition_start(hotkey) == current_time_ms
            message = f"{type(e).__name__}: {e}"
        if not moved:
            with self._buckets_lock:
                state.auto_promotion = None
            self._log_move_failure(hotkey, current_time_ms, message)
            return False

        # A nightly sync that replaced the state during the move has put the mothership's view in place,
        # with its own record or none: that one stands
        self._save_to_disk()
        logger.info(f"[CHALLENGE] AUTO-PROMOTED {hotkey} on the {path} at {record.multiplier:g}x "
                    f"(pro size ${pro_account_size:,.0f}): {record}")
        return True

    def _promotion_lock(self, hotkey: str) -> threading.RLock:
        with self._promotion_locks_guard:
            return self._promotion_locks.setdefault(hotkey, threading.RLock())

    def _is_mothership(self) -> bool:
        """Whether this validator is the mothership, read once from the entity server."""
        if self._mothership is None:
            self._mothership = bool(self._entity_client.is_mothership())
        return self._mothership

    def _log_move_failure(self, hotkey: str, current_time_ms: int, message: str) -> None:
        """A failed move is retried every refresh but logged once a payout week."""
        week_start_ms = TimeUtil.ms_at_start_of_week(current_time_ms)
        if self._move_failure_logged.get(hotkey) != week_start_ms:
            self._move_failure_logged[hotkey] = week_start_ms
            logger.error(f"[CHALLENGE] auto-promotion of {hotkey} failed, retrying on later refreshes: {message}")

    @staticmethod
    def _auto_promotion_path(stats: AutoPromotionStats, fast_track_held: bool,
                             current_return: float) -> tuple[str, int] | None:
        """The path a candidate qualifies on and its multiple, the larger when it qualifies on both, or
        None. fast_track_held is whether its equity has held the fast track's return for
        AUTO_PROMOTION_FAST_TRACK_HOLD_MS; current_return is DrawdownStats.current_return."""
        qualified = []
        if fast_track_held:
            qualified.append((AutoPromotion.FAST_TRACK, ValiConfig.AUTO_PROMOTION_FAST_TRACK_MULTIPLIER))
        if (stats.trading_days >= ValiConfig.AUTO_PROMOTION_STEADY_TRACK_MINIMUM_DAYS
                and current_return > ValiConfig.AUTO_PROMOTION_STEADY_TRACK_MINIMUM_RETURN
                and stats.daily_consistency <= ValiConfig.AUTO_PROMOTION_STEADY_TRACK_MAX_DAILY_CONSISTENCY):
            tiers = [m for calmar, m in ValiConfig.AUTO_PROMOTION_STEADY_TRACK_CALMAR_MULTIPLIERS if stats.calmar >= calmar]
            if tiers:
                qualified.append((AutoPromotion.STEADY_TRACK, max(tiers)))
        return max(qualified, key=lambda q: q[1], default=None)

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
        try:
            transitions = self._miner_account_client.get_transition_starts(hotkeys)
        except Exception as e:
            # Segments keep the transition they recorded
            logger.warning(f"[BUCKET_HISTORY] could not read the transitions: {e}")
            transitions = {}
        changed = []
        for hotkey in hotkeys:
            state = self.miner_states.get(hotkey)
            if state is None:
                continue
            try:
                if self._bucket_history.reconcile(hotkey, list(state.entries), now_ms,
                                                  on_open=self._on_segment_open,
                                                  on_close=self._on_segment_close, save=False,
                                                  transition_start_ms=transitions.get(hotkey)):
                    changed.append(hotkey)
            except Exception as e:
                logger.error(f"[BUCKET_HISTORY] reconcile failed for {hotkey}: {e}", exc_info=True)
        if changed:
            try:
                self._bucket_history.save_to_disk(changed)
            except Exception as e:
                logger.error(f"[BUCKET_HISTORY] save failed for {len(changed)} hotkeys: {e}", exc_info=True)

    def _transition_start(self, hotkey: str) -> int | None:
        """Start of the miner's SUBACCOUNT_FUNDED transition week, None when not in it."""
        return self._miner_account_client.get_transition_starts([hotkey]).get(hotkey)

    def _apply_legacy_transitions(self) -> None:
        """Move transitions read from legacy bucket entries onto the accounts, which hold them now."""
        moved = []
        for hotkey, start_ms in list(self._legacy_transition_starts.items()):
            try:
                state = self.miner_states.get(hotkey)
                if state is not None and state.current_bucket == MinerBucket.SUBACCOUNT_FUNDED:
                    if not self._miner_account_client.set_transition(hotkey, start_ms):
                        continue  # no account yet, retried next refresh
                    logger.info(f"[CHALLENGE] moved the legacy transition of {hotkey} onto its account")
                    moved.append(hotkey)
                del self._legacy_transition_starts[hotkey]
            except Exception as e:
                logger.warning(f"[CHALLENGE] could not move the legacy transition of {hotkey}: {e}")
        self._reconcile_bucket_history(moved)

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

    def _record_segment_payout(self, hotkey: str, current_time_ms: int) -> None:
        """Pin what the earning segment being left is owed, once its positions are closed. The
        payout paths read the record from now on instead of recomputing the segment. Never raises:
        a failure must not block the bucket move."""
        if not (self._bucket_history.enabled and is_synthetic_hotkey(hotkey)
                and self.miner_states[hotkey].current_bucket.is_subaccount_earning):
            return
        try:
            segment = self._bucket_history.get_live_segment(hotkey)
            if segment is None:
                self._reconcile_bucket_history([hotkey], current_time_ms)
                segment = self._bucket_history.get_live_segment(hotkey)
            if segment is None:
                return
            live = self._position_client.get_positions_for_one_hotkey(hotkey)
            if self._bucket_history.write_positions(segment, live):
                positions = self._bucket_history.read_positions(segment)
            else:
                positions = [p.to_dict() for p in live]
            self._store_payout_record(segment, positions, current_time_ms)
        except Exception as e:
            logger.error(f"[BUCKET_HISTORY] could not record the payout of {hotkey}: {e}", exc_info=True)

    def _store_payout_record(self, segment: BucketSegment, positions: list[dict], end_ms: int) -> bool:
        record = self._entity_client.calculate_segment_payout_record(
            segment.hotkey, segment.to_dict(), positions, end_ms)
        if record is None:
            logger.error(f"[BUCKET_HISTORY] no payout record could be calculated for {segment.hotkey} {segment.dir_name}")
            return False
        segment.payout_record = record
        self._bucket_history.save_segment(segment)
        logger.info(f"[BUCKET_HISTORY] payout record for {segment.hotkey} {segment.dir_name}: ${record['payout_usd']:.2f}")
        return True

    def _freeze_closed_weeks(self, hotkeys: list[str], current_time_ms: int) -> None:
        """Freeze the settlement of each payout week a live earning segment has closed, once the
        week is final. A week not final yet is retried every CLOSED_WEEKS_RETRY_MS."""
        if not self._bucket_history.enabled:
            return
        week_start_ms = TimeUtil.ms_at_start_of_week(current_time_ms)
        for hotkey in hotkeys:
            state = self.miner_states.get(hotkey)
            if (state is None or not is_synthetic_hotkey(hotkey) or not state.current_bucket.is_subaccount_earning
                    or self._closed_weeks_checked.get(hotkey, 0) >= week_start_ms
                    or current_time_ms - self._closed_weeks_attempt_ms.get(hotkey, 0) < self.CLOSED_WEEKS_RETRY_MS):
                continue
            segment = self._bucket_history.get_live_segment(hotkey)
            if segment is None:
                continue
            self._closed_weeks_attempt_ms[hotkey] = current_time_ms
            try:
                rows = self._entity_client.calculate_segment_closed_weeks(hotkey, segment.to_dict(), current_time_ms)
            except Exception as e:
                logger.error(f"[BUCKET_HISTORY] could not calculate the closed weeks of {hotkey}: {e}")
                continue
            if rows is None:
                continue
            self._closed_weeks_checked[hotkey] = week_start_ms
            if rows:
                segment.closed_weeks = (segment.closed_weeks or []) + rows
                self._bucket_history.save_segment(segment)
                logger.info(f"[BUCKET_HISTORY] froze {len(rows)} closed payout week rows of {hotkey} "
                            f"{segment.dir_name} through {rows[-1]['end_ms']}")

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
            if segment.payout_scale is None and is_synthetic_hotkey(hotkey):
                # Captured here rather than when the segment opens: a segment can open inside an
                # entity server call (subaccount creation), which must not be called back into
                try:
                    segment.payout_scale = self._entity_client.get_payout_scale(hotkey)
                    self._bucket_history.save_segment(segment)
                except Exception as e:
                    logger.warning(f"[BUCKET_HISTORY] could not read the payout scale of {hotkey}: {e}")
            pro_stats = state.pro_stats
            try:
                self._bucket_history.append_stats_row(segment, {
                    "time_ms": current_time_ms,
                    "bucket": state.current_bucket.value,
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

        bucket filters by bucket (any status MinerBucket.from_status reads, legacy names included),
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
            bucket_value = MinerBucket.from_status(bucket).value
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

    def get_bucket_segments(self, hotkey: str) -> list[dict]:
        """A miner's segment summaries, oldest first, without reverted ones."""
        return [s.to_dict() for s in self._bucket_history.get_segments(hotkey) if s.reverted_ms is None]

    def get_payout_records(self, hotkey: str) -> list[dict]:
        return [{"bucket": s.bucket, "start_ms": s.start_ms, "end_ms": s.end_ms, "payout_record": s.payout_record}
                for s in self._bucket_history.get_segments(hotkey)
                if s.reverted_ms is None and s.payout_record is not None]

    def admin_update_payout_record(self, hotkey: str, segment_start_ms: int, payout_usd: float | None = None,
                                   remove: bool = False, recompute: bool = False) -> Tuple[bool, str, dict | None]:
        """Correct a segment's payout record: amend its amount, drop it, or rebuild it from the
        positions in the segment directory."""
        segment = self._bucket_history.get_segment(hotkey, segment_start_ms)
        if segment is None or segment.reverted_ms is not None:
            return False, f"{hotkey} has no segment starting at {segment_start_ms}", None
        if remove:
            segment.payout_record = None
        elif recompute:
            if segment.end_ms is None or not MinerBucket.from_status(segment.bucket).is_subaccount_earning:
                return False, f"{segment.dir_name} is not a closed earning segment", None
            if not self._store_payout_record(segment, self._bucket_history.read_positions(segment), segment.end_ms):
                return False, f"payout record for {segment.dir_name} could not be calculated", None
        else:
            record = segment.payout_record
            if record is None:
                return False, f"{segment.dir_name} has no payout record to amend", None
            delta = payout_usd - record['payout_usd']
            if record['settlements']:
                record['settlements'][-1]['payout'] += delta
            record['payout_usd'] = payout_usd
            record['amended_ms'] = TimeUtil.now_in_millis()
        logger.warning(f"[BUCKET_HISTORY] {hotkey} {segment.dir_name} payout record manually "
                       f"{'removed' if remove else 'recomputed' if recompute else 'amended'}: {segment.payout_record}")
        self._bucket_history.save_segment(segment)
        return True, f"payout record for {segment.dir_name} updated", segment.payout_record

    def admin_drop_closed_weeks(self, hotkey: str, week_starts_ms: list[int] | None = None) -> Tuple[bool, str, list[int]]:
        """Unfreeze closed payout weeks, given by their Monday 00:00 UTC (None: every one), so they
        are computed again. Each frozen week resumes from the one before it, so every later frozen
        week of the same segment goes too. The live segment's are frozen again on the next refresh;
        a payout record already pinned keeps its rows until it is recomputed.
        Returns (success, message, the week starts dropped)."""
        segments = [s for s in self._bucket_history.get_segments(hotkey) if s.reverted_ms is None]
        if not segments:
            return False, f"{hotkey} has no bucket segments", []
        wanted = None if week_starts_ms is None else set(week_starts_ms)
        dropped = []
        for segment in segments:
            rows = segment.closed_weeks or []
            cut = next((i for i, row in enumerate(rows)
                        if wanted is None or TimeUtil.ms_at_start_of_week(row['start_ms']) in wanted), None)
            if cut is None:
                continue
            dropped += [row['start_ms'] for row in rows[cut:]]
            segment.closed_weeks = rows[:cut] or None
        if not dropped:
            return False, f"{hotkey} has no frozen weeks matching {week_starts_ms}", []
        self._closed_weeks_checked.pop(hotkey, None)
        self._closed_weeks_attempt_ms.pop(hotkey, None)
        self._bucket_history.save_to_disk([hotkey])
        logger.warning(f"[BUCKET_HISTORY] {hotkey} closed payout weeks manually unfrozen: {dropped}")
        return True, f"unfroze {len(dropped)} closed payout week rows of {hotkey}", dropped

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
        # Parsed before the swap, so data that fails to parse leaves the current states in place
        parsed = self.parse_checkpoint_dict(miner_states_data)
        with self._buckets_lock:
            self.miner_states.clear()
            self.miner_states.update(parsed)
            self._legacy_transition_starts.update(self._read_legacy_transitions(miner_states_data))
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
        intraday_drawdown_threshold: float | None = None,
        eod_hwm_threshold: float | None = None,
    ) -> bool:
        """
        Set or update a miner's bucket information, replace_top to override most recent entry.
        drawdown_criteria, intraday_drawdown_threshold and eod_hwm_threshold are set only on first creation
        and ignored for existing states.

        Only persists to disk on first creation - both are write-once, and updates to
        an existing state are already covered by the caller's own batched save (see refresh()).

        Returns:
            True if this is a new miner, False if updating existing
        """
        with self._buckets_lock:
            if hotkey not in self.miner_states:
                self.miner_states[hotkey] = MinerBucketState(hotkey, [BucketEntry(bucket, start_time_ms)], drawdown_criteria=drawdown_criteria,
                                                          intraday_drawdown_threshold_override=intraday_drawdown_threshold,
                                                          eod_hwm_threshold_override=eod_hwm_threshold)
                is_new = True
                self._save_to_disk()
            else:
                self.miner_states[hotkey].add_bucket_entry(bucket, start_time_ms, replace_top=replace_top)
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
        past its threshold, or that deepens the ratcheted max_drawdown of a pro account or of an
        auto-promotion candidate.
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
            if {'daily_open_equity', 'daily_open_balance', 'last_eod_equity'} & updates.keys() and checked_ms != today_ms:
                return False, ("daily_open_equity, daily_open_balance and last_eod_equity only hold for today: "
                               f"wait for today's EOD snapshot or set last_eod_checked_ms to {today_ms}"), None
            if 'daily_open_balance' in updates and old.daily_open_balance is None:
                return False, f"{hotkey} measures its daily loss limit from daily_open_equity, not daily_open_balance", None

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
                if state.current_bucket.is_pro_track or self._transition_start(hotkey) is not None:
                    ratchet, name = state.pro_stats.max_drawdown, "pro max_drawdown"
                elif hotkey in self._auto_promotion_candidates:
                    ratchet, name = old.max_drawdown, "the auto-promotion max_drawdown"
                else:
                    ratchet = None
                if ratchet is not None and new.current_equity / new.eod_hwm < ratchet:
                    return False, (f"eod_hwm would permanently deepen {name} "
                                   f"({ratchet:.4f} -> {new.current_equity / new.eod_hwm:.4f}); pass force to apply"), None

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
                if state.current_bucket == MinerBucket.CHALLENGE or state.current_bucket.is_subaccount_challenge}

    def get_success_miners(self) -> dict[str, int]:
        """MAINCOMP and subaccount funded miners (including the SUBACCOUNT_FUNDED transition)
        as {hotkey: start_time}."""
        return {hotkey: state.current_bucket_start_ms for hotkey, state in self.miner_states.items()
                if state.current_bucket == MinerBucket.MAINCOMP or state.current_bucket.is_subaccount_funded}

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
        if not state:
            return None
        bucket = state.current_bucket
        in_transition = self._transition_start(synthetic_hotkey) is not None
        if not (bucket.is_pro_track or in_transition):
            return None

        asset_class = self._asset_selection_client.get_asset_selection(synthetic_hotkey)
        returns_threshold = bucket.returns_threshold(asset_class) if not in_transition else None
        return {
            **state.pro_stats.to_dict(),
            "in_transition": in_transition,
            "calmar_threshold": bucket.calmar_threshold,
            "daily_consistency_threshold": bucket.daily_consistency_threshold,
            "minimum_trading_days": bucket.minimum_trading_days,
            "returns_threshold": returns_threshold,
            "soft_breach_applies": bucket.soft_breach_applies,
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
            self._legacy_transition_starts.update(self._read_legacy_transitions(disk_data))

        return miner_states

    @staticmethod
    def _read_legacy_transitions(json_dict) -> dict[str, int]:
        """hotkey -> start of the transition for states whose last entry is PRO_CHALLENGE_TRANSITION,
        written before the transition moved onto the account."""
        starts = {}
        for hotkey, info in (json_dict or {}).items():
            entries = info.get("entries") if isinstance(info, dict) else None
            if not entries:
                continue
            last = max(entries, key=lambda e: e.get("start_time_ms") or e.get("bucket_start_time") or 0)
            if last.get("bucket") == LEGACY_TRANSITION_BUCKET:
                starts[hotkey] = last.get("start_time_ms") or last.get("bucket_start_time")
        return starts

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
