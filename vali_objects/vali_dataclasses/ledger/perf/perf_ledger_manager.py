import os
import time
import traceback
from bisect import bisect_right
from collections import defaultdict
from copy import deepcopy
from typing import List

from data_generator.hyperliquid_data_service import HyperliquidDataService
from data_generator.polygon_data_service import PolygonDataService
from shared_objects.cache_controller import CacheController
from shared_objects.rpc.shutdown_coordinator import ShutdownCoordinator
from time_util.time_util import TimeUtil, timeme
from vali_objects.enums.miner_bucket_enum import MinerBucket
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
from vali_objects.utils.elimination.elimination_client import EliminationClient
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.trade_pair import TradePairSource
from vali_objects.vali_config import RPCConnectionMode, ValiConfig
from vali_objects.vali_dataclasses.ledger.perf.perf_ledger import PerfLedger, PerfCheckpoint, PERF_LEDGER_VERSION

from vali_objects.vali_dataclasses.position import Position
from entity_management.entity_utils import is_synthetic_hotkey
from shared_objects.log import logger


class PerfLedgerManager(CacheController):
    MINUTE_MS = 60 * 1000
    CHECKPOINT_MS = ValiConfig.TARGET_CHECKPOINT_DURATION_MS
    MARK_SHORT_LOOKBACK_MS = 60 * 60 * 1000  # A mark older than this is only used when no newer candle exists
    MARK_LONG_LOOKBACK_MS = 4 * ValiConfig.DAILY_MS  # Spans weekends and holidays for non-24/7 markets
    POLYGON_MAX_CANDLE_LIMIT = 49999
    UPDATE_LOOKBACK_MS = 600000  # 10 minutes ago. Want to give Polygon time to create candles on the backend.

    def __init__(self, connection_mode: "RPCConnectionMode" = RPCConnectionMode.RPC,
                 running_unit_tests=False,
                 enable_rss=True, is_backtesting=False, secrets=None):
        super().__init__(running_unit_tests=running_unit_tests, is_backtesting=is_backtesting, connection_mode=connection_mode)

        self.connection_mode = connection_mode
        self.perf_ledger_hks_to_invalidate = {}
        # Hotkeys whose invalidation the in-progress update is applying (read by save_perf_ledgers)
        self.hks_attempting_invalidations = []
        self.running_unit_tests = running_unit_tests
        self.enable_rss = enable_rss

        self.hotkey_to_perf_bundle = {}
        self._frozen_ledgers: dict[str, PerfLedger] = {}

        self._position_manager_client = PositionManagerClient(
            connect_immediately=False
        )

        # Create own EliminationClient (forward compatibility - no parameter passing)
        self._elimination_client = EliminationClient(
            port=ValiConfig.RPC_ELIMINATION_PORT,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        # Lazy import to avoid circular dependency
        from vali_objects.miner_account.miner_account_client import MinerAccountClient
        self._miner_account_client = MinerAccountClient(
            port=ValiConfig.RPC_MINERACCOUNT_PORT,
            connect_immediately=False,
            connection_mode=connection_mode
        )

        self.pds = None  # Load it later once the process starts so ipc works.
        self.hds = None  # HyperliquidDataService, lazily created for HL candle fetching.

        # Create own LivePriceFetcherClient (forward compatibility - no parameter passing)
        self._live_price_client = LivePriceFetcherClient(running_unit_tests=running_unit_tests)

        # Mark prices memoized per update round, keyed by (trade_pair_id, t_ms)
        self._mark_cache: dict[tuple[str, int], float | None] = {}

        # Every update, pick a hotkey to rebuild in case polygon candle data changed.
        self.random_security_screenings = set()
        self.n_api_calls = 0
        self.now_ms = 0  # Ledger update time. Candles closing after this are never used.
        self.hk_to_last_order_processed_ms = {}
        if self.is_backtesting:
            logger.debug("[PERF_LEDGER] Skipping disk load (backtesting)")
        else:
            logger.info("[PERF_LEDGER] Loading initial performance ledgers from disk...")
            initial_perf_ledgers = self.get_perf_ledgers(from_disk=True)
            logger.info(f"[PERF_LEDGER] Loaded {len(initial_perf_ledgers)} performance ledger bundles from disk")
            for k, v in initial_perf_ledgers.items():
                self.hotkey_to_perf_bundle[k] = v
            initial_frozen_ledgers = self.get_frozen_ledgers(from_disk=True)
            logger.info(f"[PERF_LEDGER] Loaded {len(initial_frozen_ledgers)} frozen performance ledgers from disk")
            for k, v in initial_frozen_ledgers.items():
                self._frozen_ledgers[k] = v
        if secrets:
            self.secrets = secrets
        else:
            self.secrets = ValiUtils.get_secrets(running_unit_tests=self.running_unit_tests)

    def clear_all_ledger_data(self):
        # Clear in-memory and on-disk ledgers. Only for unit tests.
        assert self.running_unit_tests, 'this is only valid for unit tests'
        self.hotkey_to_perf_bundle.clear()
        self._frozen_ledgers.clear()
        self.clear_perf_ledgers_from_disk()  # Also clears in-memory
        self.clear_frozen_ledgers_from_disk()
        self.perf_ledger_hks_to_invalidate.clear()  # Clear invalidation list for test isolation

    def re_init_perf_ledger_data(self):
        """
        Reinitialize perf ledger data by reloading from disk.
        This is useful after clear_all_ledger_data() + save_perf_ledgers() to ensure
        all internal state (caches, counters, etc.) is properly reset.
        Only for unit tests.
        """
        assert self.running_unit_tests, 'this is only valid for unit tests'

        # Reload ledgers from disk into memory cache
        ledgers_from_disk = self.get_perf_ledgers(from_disk=True)
        self.hotkey_to_perf_bundle.clear()
        for hk, bundle in ledgers_from_disk.items():
            self.hotkey_to_perf_bundle[hk] = bundle

        # Reload frozen ledgers
        frozen_from_disk = self.get_frozen_ledgers(from_disk=True)
        self._frozen_ledgers.clear()
        for hk, ledger in frozen_from_disk.items():
            self._frozen_ledgers[hk] = ledger

        logger.info(f"Reinitialized {len(self.hotkey_to_perf_bundle)} perf ledgers and {len(self._frozen_ledgers)} frozen ledgers from disk")

    # ==================== Client Properties (forward compatibility) ====================

    @property
    def metagraph(self):
        """Get metagraph client (forward compatibility - created internally)."""
        return self._metagraph_client

    @metagraph.setter
    def metagraph(self, value):
        """
        Setter to handle base class CacheController assignment.
        We ignore the value since we use our internal _metagraph_client instead.
        """
        # CacheController.__init__ sets self.metagraph = metagraph (usually None)
        # We ignore this since we use _metagraph_client created in __init__
        pass

    def _is_shutdown(self):
        """Check if shutdown has been signaled via ShutdownCoordinator."""
        return ShutdownCoordinator.is_shutdown()

    @staticmethod
    def print_bundle(hk: str, ledger: PerfLedger):
        logger.info(f'Hotkey: {hk}. Max equity return: {ledger.max_equity_ret}. Initialization time: {TimeUtil.millis_to_timestamp(ledger.initialization_time_ms)}')
        logger.info('  --portfolio-- ')
        for idx, x in enumerate(ledger.cps):
            last_update_formatted = TimeUtil.millis_to_timestamp(x.last_update_ms)
            if 1:  # idx == 0 or idx == len(ledger.cps) - 1:
                logger.info(f'    {idx} {last_update_formatted} {x}')

    def get_perf_ledgers(self, from_disk=False) -> dict[str, PerfLedger]:
        ret = {}
        if from_disk:
            compressed_json_path = ValiBkpUtils.get_perf_ledgers_path(self.running_unit_tests)

            # Try compressed JSON first (primary format)
            if os.path.exists(compressed_json_path):
                data = ValiBkpUtils.read_compressed_json(compressed_json_path)
            # Fall back to migration from .pkl or .json
            elif ValiBkpUtils.migrate_perf_ledgers_to_compressed(self.running_unit_tests):
                # Migration succeeded, now read the newly created .json.gz file
                data = ValiBkpUtils.read_compressed_json(compressed_json_path)
            else:
                # No file exists to migrate
                return ret

            for hk, value in data.items():
                try:
                    if isinstance(value, dict):
                        if 'cps' in value:
                            # New flat format: value is a PerfLedger dict directly
                            ret[hk] = PerfLedger.from_dict(value)
                        elif 'portfolio' in value:
                            # Old V2 bundle format: extract portfolio ledger
                            ret[hk] = PerfLedger.from_dict(value['portfolio'])
                        # else: skip unrecognized format
                    elif isinstance(value, PerfLedger):
                        ret[hk] = value
                except Exception as e:
                    logger.error(f"Error reading perf ledger from disk for hotkey {hk}: {e}. Skipping; it will be rebuilt from position history.")
            return ret

        return dict(self.hotkey_to_perf_bundle)

    def get_frozen_ledgers(self, from_disk=False) -> dict[str, PerfLedger]:
        ret = {}
        if from_disk:
            compressed_json_path = ValiBkpUtils.get_frozen_perf_ledgers_path(self.running_unit_tests)

            if not os.path.exists(compressed_json_path):
                return ret

            data = ValiBkpUtils.read_compressed_json(compressed_json_path)

            for hk, value in data.items():
                try:
                    if isinstance(value, dict):
                        if 'cps' in value:
                            ret[hk] = PerfLedger.from_dict(value)
                        elif 'portfolio' in value:
                            ret[hk] = PerfLedger.from_dict(value['portfolio'])
                    elif isinstance(value, PerfLedger):
                        ret[hk] = value
                except Exception as e:
                    logger.error(f"Error reading frozen perf ledger from disk for hotkey {hk}: {e}. Skipping.")
            return ret

        return dict(self._frozen_ledgers)

    def get_returns(self, hotkey: str) -> float | None:
        """
        Calculate returns for a specific hotkey's portfolio.

        Args:
            hotkey: Miner hotkey

        Returns:
            Returns as float (e.g., 0.08 for 8%), or None if no data exists
        """
        if hotkey not in self.hotkey_to_perf_bundle:
            return None

        portfolio_ledger = self.hotkey_to_perf_bundle[hotkey]
        if not portfolio_ledger.cps:
            return None

        returns = sum([cp.realized_pnl for cp in portfolio_ledger.cps]) + portfolio_ledger.cps[-1].unrealized_pnl
        return returns

    def filtered_ledger_for_scoring(
            self,
            hotkeys: List[str] = None
    ) -> dict[str, PerfLedger]:
        """
        Filter the ledger for a set of hotkeys.
        """

        if hotkeys is None:
            hotkeys = self._metagraph_client.get_hotkeys()

        # Build filtered ledger for all miners with positions
        filtered_ledger = {}

        for hotkey, perf_ledger in self.get_perf_ledgers().items():
            if hotkey not in hotkeys:
                continue

            if hotkey in self.perf_ledger_hks_to_invalidate:
                logger.warning(f"Skipping hotkey {hotkey} in filtered_ledger_for_scoring due to invalidation.")
                continue

            if perf_ledger is None or len(perf_ledger.cps) == 0:
                continue

            filtered_ledger[hotkey] = perf_ledger

        return filtered_ledger

    def clear_perf_ledgers_from_disk(self):
        assert self.running_unit_tests, 'this is only valid for unit tests'
        self.hotkey_to_perf_bundle = {}

        # Clear compressed JSON file (current format)
        json_gz_path = ValiBkpUtils.get_perf_ledgers_path(self.running_unit_tests)
        if os.path.exists(json_gz_path):
            ValiBkpUtils.write_compressed_json(json_gz_path, {})

        # Clear .pkl file if it exists (from bug)
        pkl_path = ValiBkpUtils.get_perf_ledgers_path_pkl(self.running_unit_tests)
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

        # Clear legacy uncompressed JSON file if it exists
        legacy_json_path = ValiBkpUtils.get_perf_ledgers_path_legacy(self.running_unit_tests)
        if os.path.exists(legacy_json_path):
            os.remove(legacy_json_path)

        for k in list(self.hotkey_to_perf_bundle.keys()):
            del self.hotkey_to_perf_bundle[k]

    def clear_frozen_ledgers_from_disk(self):
        assert self.running_unit_tests, 'this is only valid for unit tests'
        self._frozen_ledgers = {}

        json_gz_path = ValiBkpUtils.get_frozen_perf_ledgers_path(self.running_unit_tests)
        if os.path.exists(json_gz_path):
            os.remove(json_gz_path)

    def sync_frozen_ledgers(self, frozen_ledgers_data: dict):
        file_path = ValiBkpUtils.get_frozen_perf_ledgers_path(self.running_unit_tests)
        ValiBkpUtils.write_compressed_json(file_path, frozen_ledgers_data)
        self._frozen_ledgers = self.get_frozen_ledgers(from_disk=True)
        logger.info(f"Synced {len(self._frozen_ledgers)} frozen perf ledgers from auto sync")

    # ==================== Mark prices ====================

    @staticmethod
    def _last_close_within(closes: list[tuple[int, float]], t_ms: int, lookback_ms: int) -> float | None:
        """closes is sorted by close time. Returns the latest close in (t_ms - lookback_ms, t_ms]."""
        idx = bisect_right(closes, (t_ms, float('inf'))) - 1
        if idx < 0 or closes[idx][0] <= t_ms - lookback_ms:
            return None
        return closes[idx][1]

    def _fetch_closes(self, trade_pair, start_ms: int, end_ms: int, candle_span_ms: int) -> list[tuple[int, float]]:
        """
        Fetch candles covering [start_ms, end_ms] and return sorted (close_time_ms, close) pairs.
        A candle's close time is its open timestamp plus its span, so its close is the last traded
        price at or before that time.
        """
        self.n_api_calls += 1
        if trade_pair.src == TradePairSource.HYPERLIQUID:
            if self.hds is None:
                self.hds = HyperliquidDataService(disable_ws=True, running_unit_tests=self.running_unit_tests)
            candles = self.hds.fetch_candle_range(trade_pair, start_ms, end_ms, min_interval_span_ms=candle_span_ms)
            closes = [(c.timestamp + c.span_ms, c.close) for c in candles]
        else:
            if self.running_unit_tests:
                # LivePriceFetcherClient supports RPC test data injection (set_test_candle_data())
                candles = self._live_price_client.unified_candle_fetcher(
                    trade_pair=trade_pair, start_date=start_ms, order_date=end_ms, timespan='minute')
            else:
                if self.pds is None:
                    self.pds = PolygonDataService(api_key=self.secrets["polygon_apikey"], disable_ws=True,
                                                  is_backtesting=self.is_backtesting,
                                                  running_unit_tests=self.running_unit_tests)
                candles = self.pds.unified_candle_fetcher(
                    trade_pair=trade_pair, start_timestamp_ms=start_ms, end_timestamp_ms=end_ms, timespan='minute')
            closes = [(c.timestamp + self.MINUTE_MS, c.close) for c in (candles or [])]
        # Never use a candle that had not closed by the ledger's update time
        return sorted(x for x in closes if x[0] <= self.now_ms)

    def get_marks(self, trade_pair, times_ms: list[int]) -> dict[int, float]:
        """
        Mark price for trade_pair at each time: the last traded price at or before that time.
        Times with no price available are omitted; callers keep the position's previous mark.
        Results are memoized for the update round, since every miner shares checkpoint boundaries
        and the update time.
        """
        tp_id = trade_pair.trade_pair_id
        missing = sorted(t for t in set(times_ms) if (tp_id, t) not in self._mark_cache)

        if trade_pair.src == TradePairSource.HYPERLIQUID:
            # HL keeps only the most recent 5000 candles per interval. 12h candles close exactly on
            # checkpoint boundaries and reach back years; the unaligned update time uses 1m candles.
            boundaries = [t for t in missing if t % self.CHECKPOINT_MS == 0]
            if boundaries:
                closes = self._fetch_closes(trade_pair, boundaries[0] - self.CHECKPOINT_MS, boundaries[-1],
                                            self.CHECKPOINT_MS)
                for t in boundaries:
                    self._mark_cache[(tp_id, t)] = self._last_close_within(closes, t, self.CHECKPOINT_MS)
            for t in missing:
                if t % self.CHECKPOINT_MS != 0:
                    closes = self._fetch_closes(trade_pair, t - self.MARK_SHORT_LOOKBACK_MS, t, self.MINUTE_MS)
                    self._mark_cache[(tp_id, t)] = self._last_close_within(closes, t, self.MARK_SHORT_LOOKBACK_MS)
        else:
            # Minute candles, batched so each request stays under Polygon's per-request candle limit
            max_chunk_span_ms = self.POLYGON_MAX_CANDLE_LIMIT * self.MINUTE_MS - self.MARK_SHORT_LOOKBACK_MS
            chunks = []
            for t in missing:
                if chunks and t - chunks[-1][0] <= max_chunk_span_ms:
                    chunks[-1].append(t)
                else:
                    chunks.append([t])
            for chunk in chunks:
                closes = self._fetch_closes(trade_pair, chunk[0] - self.MARK_SHORT_LOOKBACK_MS, chunk[-1], self.MINUTE_MS)
                for t in chunk:
                    self._mark_cache[(tp_id, t)] = self._last_close_within(closes, t, self.MARK_SHORT_LOOKBACK_MS)

        # No recent candle (market closed for a weekend or holiday, or a data gap): use the last close
        for t in missing:
            if self._mark_cache[(tp_id, t)] is None:
                closes = self._fetch_closes(trade_pair, t - self.MARK_LONG_LOOKBACK_MS, t, self.MINUTE_MS)
                self._mark_cache[(tp_id, t)] = self._last_close_within(closes, t, self.MARK_LONG_LOOKBACK_MS)

        return {t: self._mark_cache[(tp_id, t)] for t in times_ms
                if self._mark_cache.get((tp_id, t)) is not None}

    # ==================== Ledger construction ====================

    def checkpoint_times(self, ledger: PerfLedger, now_ms: int) -> list[int]:
        """End times of the checkpoints to (re)compute: each 12h boundary after the ledger's last
        completed checkpoint up to now_ms, followed by now_ms itself if it isn't a boundary."""
        d = ledger.target_cp_duration_ms
        if ledger.cps:
            start = ledger.last_update_ms
        else:
            start = ledger.initialization_time_ms - ledger.initialization_time_ms % d
        times = list(range(start + d, now_ms + 1, d))
        if now_ms > (times[-1] if times else start):
            times.append(now_ms)
        return times

    def compute_pnl_snapshots(self, positions: List[Position], times_ms: list[int]) -> list[tuple[float, float, float]]:
        """
        For each time t (ascending), return (cumulative realized PnL, unrealized PnL, cumulative fees)
        across all positions, using only orders processed at or before t. Open positions are marked
        to the last traded price at t.
        """
        positions = [p for p in positions if p.orders]
        order_times = {id(p): [o.processed_ms for o in p.orders] for p in positions}

        def orders_until(p: Position, t_ms: int) -> int:
            times = order_times[id(p)]
            if t_ms >= times[-1]:
                return len(times)
            return bisect_right(times, t_ms)

        def is_open_at(p: Position, t_ms: int) -> bool:
            n = orders_until(p, t_ms)
            return n > 0 and not (p.is_closed_position and n == len(p.orders))

        # Mark prices for every (trade pair, time) with a position open past its latest order
        tp_to_times: dict[str, set] = defaultdict(set)
        tp_by_id = {}
        for p in positions:
            for t in times_ms:
                if is_open_at(p, t) and p.orders[orders_until(p, t) - 1].processed_ms < t:
                    tp_to_times[p.trade_pair.trade_pair_id].add(t)
                    tp_by_id[p.trade_pair.trade_pair_id] = p.trade_pair
        marks = {tp_id: self.get_marks(tp_by_id[tp_id], sorted(ts)) for tp_id, ts in tp_to_times.items()}

        # Partial rebuilds, keyed by (position_uuid, n_orders included). Reused across times so an
        # open position keeps its latest mark when no new price is available.
        rebuilt: dict[tuple[str, int], Position] = {}

        fee_events = sorted((e for p in positions for e in p.fee_history), key=lambda e: e.time_ms)
        fee_cursor = 0
        cumulative_fees = 0.0

        snapshots = []
        for t in times_ms:
            realized = 0.0
            unrealized = 0.0
            for p in positions:
                if t < order_times[id(p)][0]:
                    continue
                n = orders_until(p, t)
                if p.is_closed_position and n == len(p.orders):
                    realized += p.realized_pnl
                    continue
                key = (p.position_uuid, n)
                hist = rebuilt.get(key)
                if hist is None:
                    hist = deepcopy(p)
                    hist.orders = p.orders[:n]
                    hist.rebuild_position_with_updated_orders()
                    rebuilt[key] = hist
                realized += hist.realized_pnl
                if hist.is_open_position:
                    price = marks.get(p.trade_pair.trade_pair_id, {}).get(t)
                    if price is not None and hist.orders[-1].processed_ms < t:
                        hist.set_returns(price, time_ms=t)
                    unrealized += hist.unrealized_pnl

            while fee_cursor < len(fee_events) and fee_events[fee_cursor].time_ms <= t:
                cumulative_fees += fee_events[fee_cursor].amount
                fee_cursor += 1

            snapshots.append((realized, unrealized, cumulative_fees))
        return snapshots

    def update_one_perf_ledger_bundle(self, hotkey_i: int, n_hotkeys: int, hotkey: str, positions: List[Position],
                                      now_ms: int,
                                      existing_perf_ledger_bundles: dict[str, PerfLedger],
                                      account_size: float = None) -> None:
        t0 = time.time()
        self.n_api_calls = 0
        self.now_ms = now_ms

        existing_ledger = existing_perf_ledger_bundles.get(hotkey)
        if isinstance(existing_ledger, dict) and 'portfolio' in existing_ledger:
            existing_ledger = existing_ledger['portfolio']
        if not isinstance(existing_ledger, PerfLedger) or existing_ledger.version != PERF_LEDGER_VERSION:
            existing_ledger = None

        if existing_ledger is not None and now_ms < existing_ledger.last_update_ms:
            now_formatted = TimeUtil.millis_to_formatted_date_str(now_ms)
            last_update_formatted = TimeUtil.millis_to_formatted_date_str(existing_ledger.last_update_ms)
            raise Exception(f'Trying to update in the past for {hotkey}. now {now_formatted} < last update {last_update_formatted}')

        positions = [p for p in positions if p.orders and p.orders[0].processed_ms <= now_ms]
        if not positions:
            return

        if existing_ledger is None:
            first_order_time_ms = min(p.orders[0].processed_ms for p in positions)
            portfolio_pl = PerfLedger(initialization_time_ms=first_order_time_ms)
            verbose = True
            logger.info(f"Creating new perf ledger for {hotkey} with init time: {TimeUtil.millis_to_formatted_date_str(first_order_time_ms)}")
        else:
            portfolio_pl = deepcopy(existing_ledger)
            verbose = False

        # The trailing checkpoint is provisional until its boundary passes; recompute it.
        if portfolio_pl.cps and not portfolio_pl.is_complete(portfolio_pl.cps[-1]):
            portfolio_pl.cps.pop()

        if not account_size or account_size <= 0:
            # Fall back to the account size snapshotted on the miner's most recent position, then the
            # capital floor, so every ledger has an equity curve
            sized = [p for p in positions if p.account_size and p.account_size > 0]
            account_size = (max(sized, key=lambda p: p.orders[0].processed_ms).account_size if sized
                            else ValiConfig.MIN_CAPITAL)

        times = self.checkpoint_times(portfolio_pl, now_ms)
        if times:
            prev_realized = portfolio_pl.cumulative_realized_pnl
            prev_fees = portfolio_pl.cumulative_fees_usd
            # Boundary the first checkpoint starts from
            prev_t = (times[0] - 1) // portfolio_pl.target_cp_duration_ms * portfolio_pl.target_cp_duration_ms
            snapshots = self.compute_pnl_snapshots(positions, times)
            for t, (realized, unrealized, fees) in zip(times, snapshots):
                equity_ret = (account_size + realized - fees + unrealized) / account_size
                portfolio_pl.cps.append(PerfCheckpoint(
                    last_update_ms=t,
                    accum_ms=t - prev_t,
                    realized_pnl=realized - prev_realized,
                    unrealized_pnl=unrealized,
                    fees_usd=fees - prev_fees,
                    equity_ret=equity_ret,
                ))
                prev_t = t
                prev_realized = realized
                prev_fees = fees

        self.hk_to_last_order_processed_ms[hotkey] = max(
            o.processed_ms for p in positions for o in p.orders if o.processed_ms <= now_ms)

        if verbose:
            last_cp = portfolio_pl.cps[-1] if portfolio_pl.cps else None
            logger.info(
                f"Done updating perf ledger for {hotkey} {hotkey_i + 1}/{n_hotkeys} in {time.time() - t0:.2f}s. "
                f"End time {TimeUtil.millis_to_formatted_date_str(now_ms)}. n_checkpoints {len(portfolio_pl.cps)}. "
                f"n_api_calls: {self.n_api_calls}. last cp {last_cp}")

        # Write candidate at the very end in case an exception leads to a partial update
        existing_perf_ledger_bundles[hotkey] = portfolio_pl

    def update_all_perf_ledgers(self, hotkey_to_positions: dict[str, List[Position]],
                                existing_perf_ledgers: dict[str, PerfLedger],
                                now_ms: int,
                                hotkey_to_account_size: dict = None) -> None | dict[str, PerfLedger]:
        t_init = time.time()
        self.now_ms = now_ms
        self._mark_cache.clear()

        n_hotkeys = len(hotkey_to_positions)
        for hotkey_i, (hotkey, positions) in enumerate(hotkey_to_positions.items()):
            try:
                # logger.info(f"Building perf ledger for {hotkey} ({hotkey_i + 1}/{n_hotkeys})")
                account_size = hotkey_to_account_size.get(hotkey) if hotkey_to_account_size else None
                self.update_one_perf_ledger_bundle(hotkey_i, n_hotkeys, hotkey, positions, now_ms, existing_perf_ledgers,
                                                   account_size=account_size)
            except Exception as e:
                logger.error(f"Error updating perf ledger for {hotkey}: {e}. Please alert a team member ASAP!")
                logger.error(traceback.format_exc())
                continue

        n_perf_ledgers = len(existing_perf_ledgers) if existing_perf_ledgers else 0
        n_hotkeys_with_positions = len(hotkey_to_positions) if hotkey_to_positions else 0
        logger.info(f"Done updating perf ledger for all hotkeys in {time.time() - t_init} s. n_perf_ledgers {n_perf_ledgers}. n_hotkeys_with_positions {n_hotkeys_with_positions}")
        if self._is_shutdown():
            return

        self.save_perf_ledgers(existing_perf_ledgers)
        if self._frozen_ledgers and not self.is_backtesting:
            self.save_frozen_ledgers_to_disk()
        return existing_perf_ledgers


    def get_positions_perf_ledger(self, testing_one_hotkey=None):
        #testing_one_hotkey = '5GzYKUYSD5d7TJfK4jsawtmS2bZDgFuUYw8kdLdnEDxSykTU'
        hotkeys_with_no_positions = set()
        if testing_one_hotkey:
            hotkey_to_positions = self._position_manager_client.get_positions_for_hotkeys(
                [testing_one_hotkey], sort_positions=True
            )
        else:
            # live_price_fetcher is now created in __init__ - no conditional needed
            hotkey_to_positions = self._position_manager_client.get_positions_for_all_miners(sort_positions=True, filter_eliminations=True)
            n_positions_total = 0
            n_hotkeys_total = len(hotkey_to_positions)
            # Keep only hotkeys with positions
            for k, positions in hotkey_to_positions.items():
                # Rebuild closed positions to ensure returns are accurate WRT latest fee structure and retro prices.
                for p in positions:
                    if p.is_closed_position:
                        p.rebuild_position_with_updated_orders()
                n_positions = len(positions)
                n_positions_total += n_positions
                if n_positions == 0:
                    hotkeys_with_no_positions.add(k)
            for k in hotkeys_with_no_positions:
                del hotkey_to_positions[k]
            logger.info(f'PERF LEDGERS TOTAL N POSITIONS IN MEMORY: {n_positions_total} TOTAL N HOTKEYS IN MEMORY: {n_hotkeys_total}')

        return hotkey_to_positions, hotkeys_with_no_positions

    def generate_perf_ledgers_for_analysis(self, hotkey_to_positions: dict[str, List[Position]], t_ms: int = None) -> dict[str, PerfLedger]:
        if t_ms is None:
            t_ms = TimeUtil.now_in_millis()  # Time to build the perf ledgers up to. Goes back 30 days from this time.
        existing_perf_ledgers = {}
        return self.update_all_perf_ledgers(hotkey_to_positions, existing_perf_ledgers, t_ms)

    @timeme
    def update(self, testing_one_hotkey=None, regenerate_all_ledgers=False, t_ms=None):
        # Use PerfLedgerManager's own metagraph client (forward compatibility)
        assert self.metagraph, "Metagraph must be loaded before updating perf ledgers"
        perf_ledger_bundles = self.get_perf_ledgers()
        if self.is_backtesting:
            if not t_ms:
                raise Exception("t_ms must be provided in backtesting mode")
            logger.info(f'Updating perf ledgers for backtesting at time {TimeUtil.millis_to_formatted_date_str(t_ms)}')
        if t_ms is None:
            t_ms = TimeUtil.now_in_millis() - self.UPDATE_LOOKBACK_MS

        hotkey_to_positions, hotkeys_with_no_positions = self.get_positions_perf_ledger(testing_one_hotkey=testing_one_hotkey)

        def sort_key(x):
            # Highest priority. Want to rebuild this hotkey first in case it has an incorrect dd from a Polygon bug
            #if x == "5Et6DsfKyfe2PBziKo48XNsTCWst92q8xWLdcFy6hig427qH":
            #    return float('inf')
            # Otherwise, sort by the last trade time
            return hotkey_to_positions[x][-1].orders[-1].processed_ms

        # Sort the keys with the custom sort key
        hotkeys_ordered_by_last_trade = sorted(hotkey_to_positions.keys(), key=sort_key, reverse=True)

        # Remove keys from perf ledgers if they aren't inx the metagraph anymore
        metagraph_hotkeys = set(self._metagraph_client.get_hotkeys())

        # Freeze funded subaccount perf ledgers (move to separate frozen storage)
        frozen_ledger_hotkeys = self._elimination_client.get_eliminated_hotkeys_by_bucket(
            [b for b in MinerBucket if b.is_subaccount_earning]
        )

        # Move frozen ledgers from active to frozen storage
        for hk in frozen_ledger_hotkeys:
            if hk in perf_ledger_bundles:
                ledger = perf_ledger_bundles[hk]
                if ledger.cps and ledger.cps[-1].accum_ms != ledger.target_cp_duration_ms:
                    ledger.cps.pop()
                if ledger.cps:
                    self._frozen_ledgers[hk] = ledger
                del perf_ledger_bundles[hk]
                logger.info(f"Moved ledger {hk} to frozen ledger storage")
            hotkey_to_positions.pop(hk, None)

        hotkeys_to_delete = set([x for x in hotkeys_with_no_positions if x in perf_ledger_bundles])
        rss_modified = False
        # Recently re-registered
        hotkeys_rrr = []
        deltas = []
        n_valid_times = 0
        total_n_times = 0
        for hotkey in hotkey_to_positions:
            corresponding_ledger_bundle = perf_ledger_bundles.get(hotkey)
            if corresponding_ledger_bundle is None:
                continue
            portfolio_ledger = corresponding_ledger_bundle
            first_order_time_ms = min(p.orders[0].processed_ms for p in hotkey_to_positions[hotkey])
            total_n_times += 1
            if portfolio_ledger.initialization_time_ms != first_order_time_ms:
                hotkeys_rrr.append(hotkey)
                deltas.append(portfolio_ledger.initialization_time_ms - first_order_time_ms)
            else:
                n_valid_times += 1

        if hotkeys_rrr:
            logger.warning(f'Removing recently re-registered hotkeys from perf ledgers. n_valid_times {n_valid_times} total_n_times {total_n_times}. pct valid {n_valid_times / total_n_times * 100:.2f}%')
            for x in list(zip(hotkeys_rrr, deltas)):
                logger.warning(x)
            hotkeys_to_delete.update(hotkeys_rrr)

        # Determine which hotkeys to remove from the perf ledger
        hotkeys_to_iterate = [x for x in hotkeys_ordered_by_last_trade if x in perf_ledger_bundles]
        for k in perf_ledger_bundles.keys():  # Some hotkeys may not be in the positions (old, bugged, etc.)
            if k not in hotkeys_to_iterate:
                hotkeys_to_iterate.append(k)

        for hotkey in hotkeys_to_iterate:
            if hotkey in frozen_ledger_hotkeys:
                continue
            if not is_synthetic_hotkey(hotkey) and hotkey not in metagraph_hotkeys:
                hotkeys_to_delete.add(hotkey)
            elif not len(hotkey_to_positions.get(hotkey, [])):
                hotkeys_to_delete.add(hotkey)
            elif self.enable_rss and not rss_modified and hotkey not in self.random_security_screenings:
                rss_modified = True
                self.random_security_screenings.add(hotkey)
                hotkeys_to_delete.add(hotkey)

        # Start over again
        if not rss_modified:
            self.random_security_screenings = set()

        # Regenerate checkpoints if a hotkey was modified during position sync
        self.hks_attempting_invalidations = list(self.perf_ledger_hks_to_invalidate.keys())
        if self.hks_attempting_invalidations:
            for hk, t in self.perf_ledger_hks_to_invalidate.items():
                hotkeys_to_delete.add(hk)
                logger.info(f"perf ledger marked for full rebuild for hk {hk} due to position sync at time {t}")

        for k in hotkeys_to_delete:
            if k in perf_ledger_bundles:
                del perf_ledger_bundles[k]

        self.hk_to_last_order_processed_ms = {k: v for k, v in self.hk_to_last_order_processed_ms.items() if k in perf_ledger_bundles}

        #hk_to_last_update_date = {k: TimeUtil.millis_to_formatted_date_str(v.last_update_ms)
        #                            if v.last_update_ms else 'N/A' for k, v in perf_ledgers.items()}

        logger.info(f"perf ledger PLM hotkeys to delete: {hotkeys_to_delete}. rss: {self.random_security_screenings}")

        if regenerate_all_ledgers or testing_one_hotkey:
            logger.info("Regenerating all perf ledgers")
            for k in list(perf_ledger_bundles.keys()):
                del perf_ledger_bundles[k]
        try:
            self.restore_out_of_sync_ledgers(perf_ledger_bundles, hotkey_to_positions)
            if regenerate_all_ledgers or testing_one_hotkey:
                logger.info(f"  After restore_out_of_sync_ledgers: {len(perf_ledger_bundles)} ledgers")
        except Exception as e:
            logger.warning(f"Couldn't restore out of sync ledgers: {e}. Continuing...")
            logger.warning(traceback.format_exc())

        # Time in the past to start updating the perf ledgers
        logger.info("Fetching miner account sizes...")
        hotkey_to_account_size = self._miner_account_client.get_all_miner_account_sizes()
        logger.info(f"Got {len(hotkey_to_account_size)} miner account sizes. Starting update_all_perf_ledgers for {len(hotkey_to_positions)} hotkeys.")
        self.update_all_perf_ledgers(hotkey_to_positions, perf_ledger_bundles, t_ms, hotkey_to_account_size=hotkey_to_account_size)

        # Clear invalidations after successful update. Prevent race condition by only clearing if we attempted invalidation for specific hk
        if self.hks_attempting_invalidations:
            for x in self.hks_attempting_invalidations:
                if x in self.perf_ledger_hks_to_invalidate:
                    del self.perf_ledger_hks_to_invalidate[x]

        if testing_one_hotkey and not self.running_unit_tests:
            self.debug_pl_plot(testing_one_hotkey)

    def save_perf_ledgers_to_disk(self, perf_ledgers: dict[str, PerfLedger]):
        file_path = ValiBkpUtils.get_perf_ledgers_path(self.running_unit_tests)

        # Convert PerfLedger objects to dictionaries for JSON serialization
        serializable_ledgers = {}
        for hotkey, ledger in perf_ledgers.items():
            if isinstance(ledger, PerfLedger):
                serializable_ledgers[hotkey] = ledger.to_dict()
            elif isinstance(ledger, dict):
                # Handle old bundle format or already-serialized dict
                if 'portfolio' in ledger:
                    pl = ledger['portfolio']
                    serializable_ledgers[hotkey] = pl.to_dict() if isinstance(pl, PerfLedger) else pl
                elif 'cps' in ledger:
                    serializable_ledgers[hotkey] = ledger
                else:
                    serializable_ledgers[hotkey] = ledger
            else:
                serializable_ledgers[hotkey] = ledger

        ValiBkpUtils.write_compressed_json(file_path, serializable_ledgers)

    def remove_hotkeys_from_frozen_ledgers(self, hotkeys: list[str]) -> None:
        removed = [hk for hk in hotkeys if hk in self._frozen_ledgers]
        for hk in removed:
            del self._frozen_ledgers[hk]
        if removed:
            self.save_frozen_ledgers_to_disk()
            logger.info(f"[PERF_LEDGER] Removed {len(removed)} hotkeys from frozen ledgers: {removed}")

    def save_frozen_ledgers_to_disk(self, frozen_ledgers: dict[str, PerfLedger] = None):
        if frozen_ledgers is None:
            frozen_ledgers = self._frozen_ledgers

        file_path = ValiBkpUtils.get_frozen_perf_ledgers_path(self.running_unit_tests)

        serializable_ledgers = {}
        for hotkey, ledger in frozen_ledgers.items():
            if isinstance(ledger, PerfLedger):
                serializable_ledgers[hotkey] = ledger.to_dict()
            elif isinstance(ledger, dict):
                if 'portfolio' in ledger:
                    pl = ledger['portfolio']
                    serializable_ledgers[hotkey] = pl.to_dict() if isinstance(pl, PerfLedger) else pl
                elif 'cps' in ledger:
                    serializable_ledgers[hotkey] = ledger
                else:
                    serializable_ledgers[hotkey] = ledger
            else:
                serializable_ledgers[hotkey] = ledger

        ValiBkpUtils.write_compressed_json(file_path, serializable_ledgers)

    def debug_pl_plot(self, testing_one_hotkey):
        portfolio_ledger = self.get_perf_ledgers()[testing_one_hotkey]
        print(f'Portfolio ledger attributes: initialization_time_ms {portfolio_ledger.initialization_time_ms},'
              f' max_equity_ret {portfolio_ledger.max_equity_ret}')
        times = []
        equity = []
        realized = []
        unrealized = []
        cumulative_net_realized = 0.0
        for i, x in enumerate(portfolio_ledger.cps):
            times.append(TimeUtil.millis_to_timestamp(x.last_update_ms))
            equity.append(x.equity_ret)
            cumulative_net_realized += x.realized_pnl - x.fees_usd
            realized.append(cumulative_net_realized)
            unrealized.append(x.unrealized_pnl)
            # every checkpoint but the trailing one ends on a 12 hour boundary
            if i != len(portfolio_ledger.cps) - 1:
                assert x.last_update_ms % portfolio_ledger.target_cp_duration_ms == 0, x.last_update_ms
            print(x, TimeUtil.millis_to_timestamp(x.last_update_ms))

        import matplotlib.pyplot as plt
        fig, (ax_equity, ax_pnl) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax_equity.plot(times, equity, color='red')
        ax_equity.set_title(f'Equity return vs Time for HK {testing_one_hotkey}')
        ax_pnl.plot(times, realized, label='Realized PnL net of fees (USD)')
        ax_pnl.plot(times, unrealized, label='Unrealized PnL (USD)')
        ax_pnl.set_xlabel('Time')
        ax_pnl.legend()
        plt.show()

    @timeme
    def save_perf_ledgers(self, perf_ledgers_copy: dict[str, PerfLedger]):
        # We may have items in perf_ledger_hks_to_invalidate added after the iteration began.
        # Let's nuke them to allow freed hotkeys to escape elimination.
        for hk, t in self.perf_ledger_hks_to_invalidate.items():
            if hk not in self.hks_attempting_invalidations:
                logger.warning(f"perf ledger invalidated for hk {hk} during update dat {self.perf_ledger_hks_to_invalidate[hk]}. Removing from perf ledgers.")
                perf_ledgers_copy.pop(hk, None)

        if not self.is_backtesting:
            self.save_perf_ledgers_to_disk(perf_ledgers_copy)

        for k in list(self.hotkey_to_perf_bundle.keys()):
            if k not in perf_ledgers_copy:
                del self.hotkey_to_perf_bundle[k]

        for k, v in perf_ledgers_copy.items():
            self.hotkey_to_perf_bundle[k] = v

    def restore_out_of_sync_ledgers(self, existing_bundles, hotkey_to_positions):
        # TODO: Write tests
        """
        Restore ledgers subject to race condition. Perf ledger fully update loop can take 30 min.
        An order can come in during update.

        We can only build perf ledgers between orders or after all orders
        """
        for hk, bundle in existing_bundles.items():
            last_acked_order_time_ms = self.hk_to_last_order_processed_ms.get(hk)
            if not last_acked_order_time_ms:
                continue
            # bundle is now a PerfLedger directly
            pl = bundle if isinstance(bundle, PerfLedger) else bundle.get('portfolio')
            if pl is None:
                continue
            ledger_last_update_time = pl.last_update_ms
            positions = hotkey_to_positions.get(hk)
            if positions is None:
                continue
            smallest_conflict_time_ms = float('inf')
            for p in positions:
                for o in p.orders:
                    # An order came in while the perf ledger was being updated. Trim the checkpoints to avoid a race condition.
                    if last_acked_order_time_ms < o.processed_ms < ledger_last_update_time:
                        smallest_conflict_time_ms = min(smallest_conflict_time_ms, o.processed_ms)
            if smallest_conflict_time_ms != float('inf'):
                order_time_str = TimeUtil.millis_to_formatted_date_str(smallest_conflict_time_ms)
                last_acked_time_str = TimeUtil.millis_to_formatted_date_str(last_acked_order_time_ms)
                ledger_last_update_time_str = TimeUtil.millis_to_formatted_date_str(ledger_last_update_time)
                logger.info(f"Recovering checkpoints for {hk}. Order came in at {order_time_str} after last acked time {last_acked_time_str} but before perf ledger update time {ledger_last_update_time_str}")
                pl.trim_checkpoints(smallest_conflict_time_ms)
