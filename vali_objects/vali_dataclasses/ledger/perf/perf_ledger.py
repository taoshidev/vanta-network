from typing import Optional
from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
import logging
from shared_objects.log import logger

# Bump whenever checkpoint semantics change. Ledgers persisted with another version are rebuilt
# from position history instead of being extended.
PERF_LEDGER_VERSION = 2


class PerfCheckpoint:
    """
    PnL snapshot of a miner's portfolio at a checkpoint boundary.

    A checkpoint covers (last_update_ms - accum_ms, last_update_ms]. Completed checkpoints end on a
    12-hour UTC boundary with accum_ms == target_cp_duration_ms; the trailing checkpoint ends at the
    ledger's update time and is recomputed on every update until its boundary passes.
    """
    def __init__(
        self,
        last_update_ms: int,
        accum_ms: int = 0,
        realized_pnl: float = 0.0,                 # Realized PnL booked during this checkpoint (USD)
        unrealized_pnl: float = 0.0,               # Unrealized PnL of open positions at last_update_ms (USD)
        fees_usd: float = 0.0,                     # Fees charged during this checkpoint (USD)
        equity_ret: float = 1.0,                   # (account_size + cumulative realized - cumulative fees + unrealized) / account_size
        **kwargs  # Fields from older ledger versions are dropped
    ):
        self.last_update_ms = int(last_update_ms)
        self.accum_ms = int(accum_ms)
        self.realized_pnl = float(realized_pnl)
        self.unrealized_pnl = float(unrealized_pnl)
        self.fees_usd = float(fees_usd)
        self.equity_ret = float(equity_ret)

    def __eq__(self, other):
        if not isinstance(other, PerfCheckpoint):
            return False
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        return dict(self.__dict__)

    @property
    def lowerbound_time_created_ms(self):
        return self.last_update_ms - self.accum_ms


class PerfLedger():
    """Checkpoints from the miner's first order onward. History is never truncated."""
    def __init__(self, initialization_time_ms: int=0,
                 target_cp_duration_ms:int=ValiConfig.TARGET_CHECKPOINT_DURATION_MS,
                 cps: list[PerfCheckpoint]=None,
                 version: int = PERF_LEDGER_VERSION):
        if cps is None:
            cps = []
        self.target_cp_duration_ms = int(target_cp_duration_ms)
        self.initialization_time_ms = int(initialization_time_ms)
        self.cps = cps
        self.version = int(version)

    def to_dict(self):
        return {
            "initialization_time_ms": self.initialization_time_ms,
            "target_cp_duration_ms": self.target_cp_duration_ms,
            "cps": [cp.to_dict() for cp in self.cps],
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, x):
        assert isinstance(x, dict), x
        x = dict(x)
        cps = []
        prev_cumulative_fees = 0.0
        for cp in x['cps']:
            cp = dict(cp)
            if 'fees_usd' not in cp and 'cumulative_fees_usd' in cp:
                # Older checkpoints stored a running fee total; convert to the per-checkpoint amount
                cp['fees_usd'] = cp['cumulative_fees_usd'] - prev_cumulative_fees
                prev_cumulative_fees = cp['cumulative_fees_usd']
            cps.append(PerfCheckpoint(**cp))
        x['cps'] = cps
        # Ledgers serialized before versioning are version 1
        x.setdefault('version', 1)
        for legacy_key in ('tp_id', 'max_return', 'last_known_prices', 'target_ledger_window_ms'):
            x.pop(legacy_key, None)
        return cls(**x)

    @property
    def last_update_ms(self):
        if len(self.cps) == 0:  # important to return 0 as default value. Otherwise update flow wont trigger after init.
            return 0
        return self.cps[-1].last_update_ms

    @property
    def cumulative_realized_pnl(self) -> float:
        # Realized PnL since the first order, in USD, before fees
        return sum(cp.realized_pnl for cp in self.cps)

    @property
    def cumulative_fees_usd(self) -> float:
        # Fees paid since the first order, in USD
        return sum(cp.fees_usd for cp in self.cps)

    @property
    def realized_pnl_net_usd(self):
        # Cumulative realized PnL less all fees paid, in USD. Excludes unrealized marks.
        return self.cumulative_realized_pnl - self.cumulative_fees_usd

    @property
    def max_equity_ret(self) -> float:
        # Highest equity return reached at any checkpoint, floored at the starting equity of 1.0
        return max([1.0] + [cp.equity_ret for cp in self.cps])

    def equity_peaks(self) -> list[tuple[float, float]]:
        """(peak equity return so far, equity / peak) at each checkpoint. The peak starts at the
        initial equity of 1.0, so a drawdown of 1.0 means the checkpoint is at its peak."""
        running_max = 1.0
        out = []
        for cp in self.cps:
            running_max = max(running_max, cp.equity_ret)
            out.append((running_max, cp.equity_ret / running_max))
        return out

    @property
    def start_time_ms(self):
        if len(self.cps) == 0:
            return 0
        elif self.initialization_time_ms != 0:
            return self.initialization_time_ms
        else:
            return self.cps[0].lowerbound_time_created_ms

    def is_complete(self, cp: PerfCheckpoint) -> bool:
        return cp.accum_ms == self.target_cp_duration_ms

    def trim_checkpoints(self, cutoff_ms: int):
        # Drop every checkpoint whose window ends at or after cutoff_ms so it is recomputed.
        self.cps = [cp for cp in self.cps
                    if cp.lowerbound_time_created_ms + self.target_cp_duration_ms < cutoff_ms]

    def count_events(self):
        return len(self.cps)

    def get_total_ledger_duration_ms(self):
        return sum(cp.accum_ms for cp in self.cps)

    def get_checkpoint_at_time(self, timestamp_ms: int, target_cp_duration_ms: int) -> Optional[PerfCheckpoint]:
        """
        Get the checkpoint at a specific timestamp (efficient O(1) lookup).

        Uses index calculation instead of scanning since checkpoints are evenly-spaced
        and contiguous.

        Args:
            timestamp_ms: Exact timestamp to query (should match last_update_ms)
            target_cp_duration_ms: Target checkpoint duration in milliseconds

        Returns:
            Checkpoint at the exact timestamp, or None if not found

        Raises:
            ValueError: If checkpoint exists at calculated index but timestamp doesn't match (data corruption)
        """
        if not self.cps:
            return None

        first_checkpoint_ms = self.cps[0].last_update_ms
        if timestamp_ms < first_checkpoint_ms:
            return None

        time_diff = timestamp_ms - first_checkpoint_ms
        if time_diff % target_cp_duration_ms != 0:
            return None

        index = time_diff // target_cp_duration_ms
        if index >= len(self.cps):
            return None

        checkpoint = self.cps[index]
        if checkpoint.last_update_ms != timestamp_ms:
            raise ValueError(
                f"Data corruption detected for portfolio: "
                f"checkpoint at index {index} has last_update_ms {checkpoint.last_update_ms} "
                f"({TimeUtil.millis_to_formatted_date_str(checkpoint.last_update_ms)}), "
                f"but expected {timestamp_ms} "
                f"({TimeUtil.millis_to_formatted_date_str(timestamp_ms)}). "
                f"Checkpoints are not properly contiguous."
            )

        return checkpoint


if __name__ == "__main__":
    # Import here to avoid circular imports
    from vali_objects.position_management.position_utils.position_source import PositionSourceManager, PositionSource
    from vali_objects.vali_dataclasses.ledger.perf.perf_ledger_manager import PerfLedgerManager
    from vali_objects.position_management.position_manager_client import PositionManagerClient

    logger.setLevel(logging.INFO)

    # Configuration flags
    use_database_positions = True  # NEW: Enable database position loading
    use_test_positions = False      # NEW: Enable test position loading
    crypto_only = False # Whether to process only crypto trade pairs
    test_single_hotkey = '5FRWVox3FD5Jc2VnS7FUCCf8UJgLKfGdEnMAN7nU3LrdMWHu'  # Set to a specific hotkey string to test single hotkey, or None for all
    regenerate_all = False  # Whether to regenerate all ledgers from scratch

    # Time range for database queries (if using database positions)
    end_time_ms = None# 1736035200000    # Jan 5, 2025

    # Validate configuration
    if use_database_positions and use_test_positions:
        raise ValueError("Cannot use both database and test positions. Choose one.")

    # Initialize components
    all_miners_dir = ValiBkpUtils.get_miner_dir(running_unit_tests=False)
    all_hotkeys_on_disk = CacheController.get_directory_names(all_miners_dir)

    # Determine which hotkeys to process
    if test_single_hotkey:
        hotkeys_to_process = [test_single_hotkey]
    else:
        hotkeys_to_process = all_hotkeys_on_disk

    # Load positions from alternative sources if configured
    hk_to_positions = {}
    if use_database_positions or use_test_positions:
        # Determine source type
        if use_database_positions:
            source_type = PositionSource.DATABASE
            logger.info("Using database as position source")
        else:  # use_test_positions
            source_type = PositionSource.TEST
            logger.info("Using test data as position source")

        # Load positions
        position_source_manager = PositionSourceManager(source_type)
        hk_to_positions = position_source_manager.load_positions(
            end_time_ms=end_time_ms if use_database_positions else None,
            hotkeys=hotkeys_to_process if use_database_positions else None)

        # Update hotkeys to process based on loaded positions
        if hk_to_positions:
            hotkeys_to_process = list(hk_to_positions.keys())
            logger.info(f"Loaded positions for {len(hotkeys_to_process)} miners from {source_type.value}")

    # Save loaded positions if using alternative source
    if hk_to_positions:
        position_manager_client = PositionManagerClient(connect_immediately=False)
        position_count = 0
        for hk, positions in hk_to_positions.items():
            for pos in positions:
                if crypto_only and not pos.trade_pair.is_crypto:
                    continue
                position_manager_client.save_miner_position(pos)
                position_count += 1
        logger.info(f"Saved {position_count} positions to position manager")

    # PerfLedgerManager creates its own MetagraphClient and PositionManagerClient internally
    perf_ledger_manager = PerfLedgerManager(running_unit_tests=False, enable_rss=False)
    if test_single_hotkey:
        logger.info(f"Running single-hotkey test for: {test_single_hotkey}")
        perf_ledger_manager.update(testing_one_hotkey=test_single_hotkey, t_ms=TimeUtil.now_in_millis())
    else:
        logger.info("Running standard sequential update for all hotkeys")
        perf_ledger_manager.update(regenerate_all_ledgers=regenerate_all)
