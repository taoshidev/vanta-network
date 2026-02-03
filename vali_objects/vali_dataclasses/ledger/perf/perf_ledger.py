import math

from vali_objects.enums.misc import TradePairReturnStatus
from typing import Dict, Tuple, Optional
import bittensor as bt
from shared_objects.sn8_multiprocessing import ParallelizationMode, get_spark_session, get_multiprocessing_pool
from time_util.time_util import MS_IN_8_HOURS, MS_IN_24_HOURS
from shared_objects.cache_controller import CacheController
from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.position import Position
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils

TP_ID_PORTFOLIO = 'portfolio'


class FeeCache():
    def __init__(self):
        self.spread_fee: float = 1.0
        self.spread_fee_last_order_processed_ms: int = 0

        self.carry_fee: float = 1.0  # product of all individual interval fees.
        self.carry_fee_next_increase_time_ms: int = 0  # Compute fees based off the prior interval

    def get_spread_fee(self, position: Position, current_time_ms: int) -> (float, bool):
        if position.orders[-1].processed_ms == self.spread_fee_last_order_processed_ms:
            return self.spread_fee, False

        if position.is_closed_position:
            current_time_ms = min(current_time_ms, position.close_ms)

        self.spread_fee = position.get_spread_fee(current_time_ms)
        self.spread_fee_last_order_processed_ms = position.orders[-1].processed_ms
        return self.spread_fee, True

    def get_carry_fee(self, current_time_ms, position: Position) -> (float, bool):
        # Calculate the number of times a new day occurred (UTC). If a position is opened at 23:59:58 and this function is
        # called at 00:00:02, the carry fee will be calculated as if a day has passed. Another example: if a position is
        # opened at 23:59:58 and this function is called at 23:59:59, the carry fee will be calculated as 0 days have passed
        if position.is_closed_position:
            current_time_ms = min(current_time_ms, position.close_ms)
        # cache hit?
        if position.trade_pair.is_crypto:
            start_time_cache_hit = self.carry_fee_next_increase_time_ms - MS_IN_8_HOURS
        elif position.trade_pair.is_forex or position.trade_pair.is_indices or position.trade_pair.is_equities:
            start_time_cache_hit = self.carry_fee_next_increase_time_ms - MS_IN_24_HOURS
        else:
            raise Exception(f"Unknown trade pair type: {position.trade_pair}")
        if start_time_cache_hit <= current_time_ms < self.carry_fee_next_increase_time_ms:
            return self.carry_fee, False

        # cache miss
        carry_fee, next_update_time_ms = position.get_carry_fee(current_time_ms)
        assert next_update_time_ms > current_time_ms, [TimeUtil.millis_to_verbose_formatted_date_str(x) for x in (self.carry_fee_next_increase_time_ms, next_update_time_ms, current_time_ms)] + [carry_fee, position] + [self.carry_fee_next_increase_time_ms, next_update_time_ms, current_time_ms]

        assert carry_fee >= 0, (carry_fee, next_update_time_ms, position)
        self.carry_fee = carry_fee
        self.carry_fee_next_increase_time_ms = next_update_time_ms
        return self.carry_fee, True

class PerfCheckpoint:
    def __init__(
        self,
        last_update_ms: int,
        prev_portfolio_ret: float,
        prev_portfolio_realized_pnl: float = 0.0,
        prev_portfolio_unrealized_pnl: float = 0.0,
        prev_portfolio_spread_fee: float = 1.0,
        prev_portfolio_carry_fee: float = 1.0,
        accum_ms: int = 0,
        open_ms: int = 0,
        n_updates: int = 0,
        gain: float = 0.0,
        loss: float = 0.0,
        spread_fee_loss: float = 0.0,
        carry_fee_loss: float = 0.0,
        mdd: float = 1.0,
        mpv: float = 0.0,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
        **kwargs  # Support extra fields like BaseModel's extra="allow"
    ):
        # Type coercion to match BaseModel behavior (handles numpy types and ensures correct types)
        self.last_update_ms = int(last_update_ms)
        self.prev_portfolio_ret = float(prev_portfolio_ret)
        self.prev_portfolio_realized_pnl = float(prev_portfolio_realized_pnl)
        self.prev_portfolio_unrealized_pnl = float(prev_portfolio_unrealized_pnl)
        self.prev_portfolio_spread_fee = float(prev_portfolio_spread_fee)
        self.prev_portfolio_carry_fee = float(prev_portfolio_carry_fee)
        self.accum_ms = int(accum_ms)
        self.open_ms = int(open_ms)
        self.n_updates = int(n_updates)
        self.gain = float(gain)
        self.loss = float(loss)
        self.spread_fee_loss = float(spread_fee_loss)
        self.carry_fee_loss = float(carry_fee_loss)
        self.mdd = float(mdd)
        self.mpv = float(mpv)
        self.realized_pnl = float(realized_pnl)
        self.unrealized_pnl = float(unrealized_pnl)

        # Store any extra fields (equivalent to model_config extra="allow")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __eq__(self, other):
        """Equality comparison (replaces BaseModel's automatic __eq__)"""
        if not isinstance(other, PerfCheckpoint):
            return False
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.to_dict())

    def to_dict(self):
        # Convert any numpy types to Python types for JSON serialization
        result = {}
        for key, value in self.__dict__.items():
            # Handle numpy int64, float64, etc.
            if hasattr(value, 'item'):  # numpy types have .item() method
                result[key] = value.item()
            else:
                result[key] = value
        return result

    @property
    def lowerbound_time_created_ms(self):
        # accum_ms boundary alignment makes this a lowerbound for the first cp.
        return self.last_update_ms - self.accum_ms


class PerfLedger():
    def __init__(self, initialization_time_ms: int=0, max_return:float=1.0,
                 target_cp_duration_ms:int=ValiConfig.TARGET_CHECKPOINT_DURATION_MS,
                 target_ledger_window_ms=ValiConfig.TARGET_LEDGER_WINDOW_MS, cps: list[PerfCheckpoint]=None,
                 tp_id: str=TP_ID_PORTFOLIO, last_known_prices: Dict[str, Tuple[float, int]]=None):
        if cps is None:
            cps = []
        if last_known_prices is None:
            last_known_prices = {}
        self.max_return = float(max_return)
        self.target_cp_duration_ms = int(target_cp_duration_ms)
        self.target_ledger_window_ms = target_ledger_window_ms
        self.initialization_time_ms = int(initialization_time_ms)
        self.tp_id = str(tp_id)
        self.cps = cps
        # Price continuity tracking - maps trade pair to (price, timestamp_ms)
        self.last_known_prices = last_known_prices
        if last_known_prices and self.tp_id != TP_ID_PORTFOLIO:
            raise ValueError(f"last_known_prices should only be set for portfolio ledgers, but got tp_id: {self.tp_id}")

    def to_dict(self):
        return {
            "initialization_time_ms": self.initialization_time_ms,
            "max_return": self.max_return,
            "target_cp_duration_ms": self.target_cp_duration_ms,
            "target_ledger_window_ms": self.target_ledger_window_ms,
            "cps": [cp.to_dict() for cp in self.cps],
            "last_known_prices": self.last_known_prices
        }

    @classmethod
    def from_dict(cls, x):
        assert isinstance(x, dict), x
        x['cps'] = [PerfCheckpoint(**cp) for cp in x['cps']]
        # Handle missing last_known_prices for backward compatibility
        if 'last_known_prices' not in x:
            x['last_known_prices'] = {}
        instance = cls(**x)
        return instance

    @property
    def mdd(self):
        return min(cp.mdd for cp in self.cps) if self.cps else 1.0

    @property
    def total_open_ms(self):
        if len(self.cps) == 0:
            return 0
        return sum(cp.open_ms for cp in self.cps)

    @property
    def last_update_ms(self):
        if len(self.cps) == 0:  # important to return 0 as default value. Otherwise update flow wont trigger after init.
            return 0
        return self.cps[-1].last_update_ms

    @property
    def prev_portfolio_ret(self):
        if len(self.cps) == 0:
            return 1.0  # Initial value
        return self.cps[-1].prev_portfolio_ret

    @property
    def start_time_ms(self):
        if len(self.cps) == 0:
            return 0
        elif self.initialization_time_ms != 0:  # 0 default value for old ledgers that haven't rebuilt as of this update.
            return self.initialization_time_ms
        else:
            return self.cps[0].lowerbound_time_created_ms  # legacy calculation that will stop being used in ~24 hrs

    def init_max_portfolio_value(self):
        if self.cps:
            self.max_return = max(x.mpv for x in self.cps)
        # Initial portfolio value is 1.0
        self.max_return = max(self.max_return, 1.0)


    def init_with_first_order(self, order_processed_ms: int, point_in_time_dd: float, current_portfolio_value: float,
                              current_portfolio_fee_spread:float, current_portfolio_carry:float,
                              hotkey: str=None):
        # figure out how many ms we want to initalize the checkpoint with so that once self.target_cp_duration_ms is
        # reached, the CP ends at 00:00:00 UTC or 12:00:00 UTC (12 hr cp case). This may change based on self.target_cp_duration_ms
        # |----x------midday-----------| -> accum_ms_for_utc_alignment = (distance between start of day and x) = x - start_of_day_ms
        # |-----------midday-----x-----| -> accum_ms_for_utc_alignment = (distance between midday and x) = x - midday_ms
        # By calculating the initial accum_ms this way, the co will always end at middday or 00:00:00 the next day.

        assert order_processed_ms != 0, "order_processed_ms cannot be 0. This is likely a bug in the code."
        datetime_representation = TimeUtil.millis_to_datetime(order_processed_ms)
        assert self.target_cp_duration_ms == 43200000, f'self.target_cp_duration_ms is not 12 hours {self.target_cp_duration_ms}'
        midday = datetime_representation.replace(hour=12, minute=0, second=0, microsecond=0)
        midday_ms = int(midday.timestamp() * 1000)
        if order_processed_ms < midday_ms:
            start_of_day = datetime_representation.replace(hour=0, minute=0, second=0, microsecond=0)
            start_of_day_ms = int(start_of_day.timestamp() * 1000)
            accum_ms_for_utc_alignment = order_processed_ms - start_of_day_ms
        else:
            accum_ms_for_utc_alignment = order_processed_ms - midday_ms

        # Start with open_ms equal to accum_ms (assuming positions are open from the start)
        new_cp = PerfCheckpoint(last_update_ms=order_processed_ms, prev_portfolio_ret=current_portfolio_value,
                                mdd=point_in_time_dd, prev_portfolio_spread_fee=current_portfolio_fee_spread,
                                prev_portfolio_carry_fee=current_portfolio_carry, accum_ms=accum_ms_for_utc_alignment,
                                mpv=1.0)
        self.cps.append(new_cp)



    def compute_delta_between_ticks(self, cur: float, prev: float):
        return math.log(cur / prev)

    def purge_old_cps(self):
        while self.get_total_ledger_duration_ms() > self.target_ledger_window_ms:
            bt.logging.trace(
                f"Purging old perf cp {self.cps[0]}. Total ledger duration: {self.get_total_ledger_duration_ms()}. Target ledger window: {self.target_ledger_window_ms}")
            self.cps = self.cps[1:]  # Drop the first cp (oldest)

    def trim_checkpoints(self, cutoff_ms: int):
        new_cps = []
        any_changes = False
        for cp in self.cps:
            if cp.lowerbound_time_created_ms + self.target_cp_duration_ms >= cutoff_ms:
                any_changes = True
                continue
            new_cps.append(cp)
        if any_changes:
            self.cps = new_cps
            self.init_max_portfolio_value()

    def update_pl(self, current_portfolio_value: float, now_ms: int, miner_hotkey: str, any_open: TradePairReturnStatus,
                  current_portfolio_fee_spread: float, current_portfolio_carry: float, current_realized_pnl_usd: float, current_unrealized_pnl_usd: float,
                  tp_debug=None, debug_dict=None):
        # Skip gap validation during void filling, shortcuts, or when no debug info
        # The absence of tp_debug typically means this is a high-level update that may span time
        skip_gap_check = (not tp_debug or '_shortcut' in tp_debug or 'void' in tp_debug)

        # If we have checkpoints, verify continuous updates (unless explicitly skipping)
        if len(self.cps) > 0 and not skip_gap_check:
            time_gap = now_ms - self.last_update_ms

            # Allow up to 1 minute gap (plus small buffer for processing)
            max_allowed_gap = 61000  # 61 seconds

            assert time_gap <= max_allowed_gap, (
                f"Large gap in update_pl for {tp_debug or 'portfolio'}: {time_gap/1000:.1f}s. "
                f"Last: {TimeUtil.millis_to_formatted_date_str(self.last_update_ms)}, "
                f"Now: {TimeUtil.millis_to_formatted_date_str(now_ms)}"
            )

        if len(self.cps) == 0:
            self.init_with_first_order(now_ms, point_in_time_dd=1.0, current_portfolio_value=1.0,
                                           current_portfolio_fee_spread=1.0, current_portfolio_carry=1.0)
        prev_max_return = self.max_return
        last_portfolio_return = self.cps[-1].prev_portfolio_ret
        prev_mdd = CacheController.calculate_drawdown(last_portfolio_return, prev_max_return)
        self.max_return = max(self.max_return, current_portfolio_value)
        point_in_time_dd = CacheController.calculate_drawdown(current_portfolio_value, self.max_return)
        if not point_in_time_dd:
            time_formatted = TimeUtil.millis_to_verbose_formatted_date_str(now_ms)
            raise Exception(f'point_in_time_dd is {point_in_time_dd} at time {time_formatted}. '
                            f'any_open: {any_open}, prev_portfolio_value {self.cps[-1].prev_portfolio_ret}, '
                            f'current_portfolio_value: {current_portfolio_value}, self.max_return: {self.max_return}, debug_dict: {debug_dict}')

        if len(self.cps) == 0:
            self.init_with_first_order(now_ms, point_in_time_dd, current_portfolio_value, current_portfolio_fee_spread,
                                       current_portfolio_carry)
            return

        time_since_last_update_ms = now_ms - self.cps[-1].last_update_ms
        assert time_since_last_update_ms >= 0, self.cps

        if time_since_last_update_ms + self.cps[-1].accum_ms > self.target_cp_duration_ms:
            # Need to fill void - complete current checkpoint and create new ones

            # Validate that we're working with 12-hour checkpoints
            if self.target_cp_duration_ms != 43200000:  # 12 hours in milliseconds
                raise Exception(f"Checkpoint boundary alignment only supports 12-hour checkpoints, "
                                f"but target_cp_duration_ms is {self.target_cp_duration_ms} ms "
                                f"({self.target_cp_duration_ms / 3600000:.1f} hours)")

            # Step 1: Complete the current checkpoint by aligning to 12-hour boundary
            # Find the next 12-hour boundary
            next_boundary = TimeUtil.align_to_12hour_checkpoint_boundary(self.cps[-1].last_update_ms)
            if next_boundary > now_ms:
                raise Exception(
                    f"Cannot align checkpoint: next boundary {next_boundary} ({TimeUtil.millis_to_formatted_date_str(next_boundary)}) "
                    f"exceeds current time {now_ms} ({TimeUtil.millis_to_formatted_date_str(now_ms)})")

            # Update the current checkpoint to end at the boundary
            delta_to_boundary = self.target_cp_duration_ms - self.cps[-1].accum_ms
            self.cps[-1].last_update_ms = next_boundary
            self.cps[-1].accum_ms = self.target_cp_duration_ms

            # Complete the current checkpoint using last_portfolio_return (no change in value during void)
            # The current checkpoint should be filled to the boundary but without value changes
            # Only the final checkpoint after void filling gets the new portfolio value
            if any_open > TradePairReturnStatus.TP_MARKET_NOT_OPEN:
                self.cps[-1].open_ms += delta_to_boundary

            # Step 2: Create full 12-hour checkpoints for the void period
            current_boundary = next_boundary
            # During void periods, portfolio value remains constant at last_portfolio_return
            # Do NOT update last_portfolio_return to current_portfolio_value yet

            while now_ms - current_boundary > self.target_cp_duration_ms:
                current_boundary += self.target_cp_duration_ms
                new_cp = PerfCheckpoint(
                    last_update_ms=current_boundary,
                    prev_portfolio_ret=last_portfolio_return,  # Keep constant during void
                    prev_portfolio_realized_pnl=self.cps[-1].prev_portfolio_realized_pnl,
                    prev_portfolio_unrealized_pnl=self.cps[-1].prev_portfolio_unrealized_pnl,
                    prev_portfolio_spread_fee=self.cps[-1].prev_portfolio_spread_fee,
                    prev_portfolio_carry_fee=self.cps[-1].prev_portfolio_carry_fee,
                    accum_ms=self.target_cp_duration_ms,
                    open_ms=0,  # No market data for void periods
                    mdd=prev_mdd,
                    mpv=last_portfolio_return
                )
                assert new_cp.last_update_ms % self.target_cp_duration_ms == 0, f"Checkpoint not aligned: {new_cp.last_update_ms}"
                self.cps.append(new_cp)

            # Step 3: Create final partial checkpoint from last boundary to now
            time_since_boundary = now_ms - current_boundary
            assert 0 <= time_since_boundary <= self.target_cp_duration_ms

            final_open_ms = time_since_boundary if any_open > TradePairReturnStatus.TP_MARKET_NOT_OPEN else 0
            # Calculate MDD for this checkpoint period based on the change from boundary to now
            # MDD should be the worst decline within this checkpoint period

            new_cp = PerfCheckpoint(
                last_update_ms=now_ms,
                prev_portfolio_ret=last_portfolio_return, # old for now, update below
                prev_portfolio_realized_pnl=self.cps[-1].prev_portfolio_realized_pnl,
                prev_portfolio_unrealized_pnl=self.cps[-1].prev_portfolio_unrealized_pnl,
                prev_portfolio_spread_fee=self.cps[-1].prev_portfolio_spread_fee,  # old for now update below
                prev_portfolio_carry_fee=self.cps[-1].prev_portfolio_carry_fee,    # old for now update below
                carry_fee_loss=0, # 0 for now, update below
                spread_fee_loss=0, # 0 for now, update below
                n_updates = 0, # 0 for now, update below
                gain=0,  # 0 for now, update below
                loss=0,  # 0 for now, update below
                mdd=prev_mdd,  # old for now update below
                mpv=last_portfolio_return, # old for now, update below
                accum_ms=time_since_boundary,
                open_ms=final_open_ms,
            )
            self.cps.append(new_cp)
        else:
            # Nominal update. No void to fill
            current_cp = self.cps[-1]
            # Calculate time since this checkpoint's last update
            time_to_accumulate = now_ms - current_cp.last_update_ms
            if time_to_accumulate < 0:
                bt.logging.error(f"Negative accumulated time: {time_to_accumulate} for miner {miner_hotkey}."
                                 f" start_time_ms: {self.start_time_ms}, now_ms: {now_ms}")
                time_to_accumulate = 0

            current_cp.accum_ms += time_to_accumulate
            # Update open_ms only when market is actually open
            if any_open > TradePairReturnStatus.TP_MARKET_NOT_OPEN:
                current_cp.open_ms += time_to_accumulate


        current_cp = self.cps[-1]  # Get the current checkpoint after updates
        current_cp.mdd = min(current_cp.mdd, point_in_time_dd)
        # Update gains/losses based on portfolio value change
        n_updates = 1
        delta_return = self.compute_delta_between_ticks(current_portfolio_value, current_cp.prev_portfolio_ret)

        if delta_return > 0:
            current_cp.gain += delta_return
        elif delta_return < 0:
            current_cp.loss += delta_return
        else:
            n_updates = 0

        # realized_pnl stores delta (sum of realized gains/losses during checkpoint period)
        delta_realized = current_realized_pnl_usd - current_cp.prev_portfolio_realized_pnl
        current_cp.realized_pnl += delta_realized
        # unrealized_pnl stores snapshot (current unrealized PnL at checkpoint end)
        current_cp.unrealized_pnl = current_unrealized_pnl_usd

        # Update fee losses
        if current_cp.prev_portfolio_carry_fee != current_portfolio_carry:
            current_cp.carry_fee_loss += self.compute_delta_between_ticks(current_portfolio_carry,
                                                                          current_cp.prev_portfolio_carry_fee)
        if current_cp.prev_portfolio_spread_fee != current_portfolio_fee_spread:
            current_cp.spread_fee_loss += self.compute_delta_between_ticks(current_portfolio_fee_spread,
                                                                           current_cp.prev_portfolio_spread_fee)

        # Update portfolio values
        current_cp.prev_portfolio_ret = current_portfolio_value
        current_cp.prev_portfolio_realized_pnl = current_realized_pnl_usd
        current_cp.prev_portfolio_unrealized_pnl = current_unrealized_pnl_usd
        current_cp.last_update_ms = now_ms
        current_cp.prev_portfolio_spread_fee = current_portfolio_fee_spread
        current_cp.prev_portfolio_carry_fee = current_portfolio_carry
        current_cp.mpv = max(current_cp.mpv, current_portfolio_value)
        current_cp.n_updates += n_updates


    def count_events(self):
        # Return the number of events currently stored
        return len(self.cps)

    def get_product_of_gains(self):
        cumulative_gains = sum(cp.gain for cp in self.cps)
        return math.exp(cumulative_gains)

    def get_product_of_loss(self):
        cumulative_loss = sum(cp.loss for cp in self.cps)
        return math.exp(cumulative_loss)

    def get_total_product(self):
        cumulative_gains = sum(cp.gain for cp in self.cps)
        cumulative_loss = sum(cp.loss for cp in self.cps)
        return math.exp(cumulative_gains + cumulative_loss)

    def get_total_ledger_duration_ms(self):
        return sum(cp.accum_ms for cp in self.cps)

    def get_checkpoint_at_time(self, timestamp_ms: int, target_cp_duration_ms: int) -> Optional[PerfCheckpoint]:
        """
        Get the checkpoint at a specific timestamp (efficient O(1) lookup).

        Uses index calculation instead of scanning since checkpoints are evenly-spaced
        and contiguous (enforced by strict checkpoint validation).

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

        # Calculate expected index based on first checkpoint and duration
        first_checkpoint_ms = self.cps[0].last_update_ms

        # Check if timestamp is before first checkpoint
        if timestamp_ms < first_checkpoint_ms:
            return None

        # Calculate index (checkpoints are evenly spaced by target_cp_duration_ms)
        time_diff = timestamp_ms - first_checkpoint_ms
        if time_diff % target_cp_duration_ms != 0:
            # Timestamp doesn't align with checkpoint boundaries
            return None

        index = time_diff // target_cp_duration_ms

        # Check if index is within bounds
        if index >= len(self.cps):
            return None

        # Validate the checkpoint at this index has the expected timestamp
        checkpoint = self.cps[index]
        if checkpoint.last_update_ms != timestamp_ms:
            from time_util.time_util import TimeUtil
            raise ValueError(
                f"Data corruption detected for {self.tp_id}: "
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

    bt.logging.enable_info()

    # Configuration flags
    use_database_positions = True  # NEW: Enable database position loading
    use_test_positions = False      # NEW: Enable test position loading
    crypto_only = False # Whether to process only crypto trade pairs
    parallel_mode = ParallelizationMode.SERIAL  # 1 for pyspark, 2 for multiprocessing
    top_n_miners = 4
    test_single_hotkey = '5FRWVox3FD5Jc2VnS7FUCCf8UJgLKfGdEnMAN7nU3LrdMWHu'  # Set to a specific hotkey string to test single hotkey, or None for all
    regenerate_all = False  # Whether to regenerate all ledgers from scratch
    build_portfolio_ledgers_only = False  # Whether to build only the portfolio ledgers or per trade pair

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
            bt.logging.info("Using database as position source")
        else:  # use_test_positions
            source_type = PositionSource.TEST
            bt.logging.info("Using test data as position source")

        # Load positions
        position_source_manager = PositionSourceManager(source_type)
        hk_to_positions = position_source_manager.load_positions(
            end_time_ms=end_time_ms if use_database_positions else None,
            hotkeys=hotkeys_to_process if use_database_positions else None)

        # Update hotkeys to process based on loaded positions
        if hk_to_positions:
            hotkeys_to_process = list(hk_to_positions.keys())
            bt.logging.info(f"Loaded positions for {len(hotkeys_to_process)} miners from {source_type.value}")

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
        bt.logging.info(f"Saved {position_count} positions to position manager")

    # PerfLedgerManager creates its own MetagraphClient and PositionManagerClient internally
    perf_ledger_manager = PerfLedgerManager(running_unit_tests=False,
                                            enable_rss=False, parallel_mode=parallel_mode,
                                            build_portfolio_ledgers_only=build_portfolio_ledgers_only)


    if parallel_mode == ParallelizationMode.SERIAL:
        # Use serial update like validators do
        if test_single_hotkey:
            bt.logging.info(f"Running single-hotkey test for: {test_single_hotkey}")
            perf_ledger_manager.update(testing_one_hotkey=test_single_hotkey, t_ms=TimeUtil.now_in_millis())
        else:
            bt.logging.info("Running standard sequential update for all hotkeys")
            perf_ledger_manager.update(regenerate_all_ledgers=regenerate_all)
    else:
        # Get positions and existing ledgers
        hotkey_to_positions, _ = perf_ledger_manager.get_positions_perf_ledger(testing_one_hotkey=test_single_hotkey)

        existing_perf_ledgers = {} if regenerate_all else perf_ledger_manager.get_perf_ledgers(portfolio_only=False, from_disk=True)

        # Run the parallel update
        spark, should_close = get_spark_session(parallel_mode)
        pool = get_multiprocessing_pool(parallel_mode)
        assert pool, parallel_mode
        updated_perf_ledgers = perf_ledger_manager.update_perf_ledgers_parallel(spark, pool, hotkey_to_positions,
                                    existing_perf_ledgers, parallel_mode=parallel_mode, top_n_miners=top_n_miners)

        PerfLedgerManager.print_bundles(updated_perf_ledgers)
