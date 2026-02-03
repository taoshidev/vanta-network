# developer: jbonilla
# Copyright (c) 2024 Taoshi Inc
"""
MDDChecker - Core logic for MDD (Maximum Drawdown) checking and price corrections.

This class contains the business logic for:
- Real-time price corrections for recent orders
- Position return updates using live prices
- MDD checking for all miners

The MDDCheckerServer wraps this class and exposes it via RPC.
"""
import time
import traceback
from typing import List, Dict

import bittensor as bt

from collections import defaultdict

from shared_objects.cache_controller import CacheController
from shared_objects.rpc.common_data_client import CommonDataClient
from time_util.time_util import TimeUtil
from vali_objects.vali_dataclasses.position import Position
from vali_objects.price_fetcher.live_price_client import LivePriceFetcherClient
from shared_objects.locks.position_lock_client import PositionLockClient
from vali_objects.position_management.position_manager_client import PositionManagerClient
from vali_objects.utils.price_slippage_model import PriceSlippageModel
from vali_objects.utils.vali_bkp_utils import ValiBkpUtils
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import ValiConfig, TradePair, RPCConnectionMode
from vali_objects.vali_dataclasses.price_source import PriceSource


class MDDChecker(CacheController):
    """
    Core MDD checking and price correction logic.

    This class contains all the business logic for MDD checking.
    The MDDCheckerServer wraps this and exposes it via RPC.
    """

    def __init__(
        self,
        running_unit_tests: bool = False,
        connection_mode: RPCConnectionMode = RPCConnectionMode.RPC
    ):
        """
        Initialize MDDChecker.

        Args:
            running_unit_tests: Whether running in unit test mode
            connection_mode: RPCConnectionMode for client connections
        """
        super().__init__(running_unit_tests=running_unit_tests, connection_mode=connection_mode)

        self.last_price_fetch_time_ms = None
        self.last_quote_fetch_time_ms = None
        self.price_correction_enabled = True

        # Create RPC clients for external dependencies
        self._common_data_client = CommonDataClient(connection_mode=connection_mode)
        self._live_price_client = LivePriceFetcherClient(
            connection_mode=connection_mode,
            running_unit_tests=running_unit_tests
        )
        self._position_client = PositionManagerClient()
        self._position_lock_client = PositionLockClient(running_unit_tests=running_unit_tests)

        self.all_trade_pairs = [trade_pair for trade_pair in TradePair]
        self._prev_iteration_prices = {}
        self.reset_debug_counters()
        self.n_poly_api_requests = 0

        # Load persisted slippage features from disk for equities slippage calculation
        try:
            features_file = ValiBkpUtils.get_slippage_model_features_file()
            persisted_features = ValiUtils.get_vali_json_file_dict(features_file)
            if persisted_features:
                PriceSlippageModel.features = defaultdict(dict, persisted_features)
                bt.logging.info(f"MDDChecker loaded {len(persisted_features)} days of slippage features")
        except Exception as e:
            bt.logging.warning(f"MDDChecker could not load slippage features: {e}")

        bt.logging.info("MDDChecker initialized")

    # ==================== Properties ====================

    @property
    def sync_in_progress(self):
        """Get sync_in_progress flag via CommonDataClient."""
        return self._common_data_client.get_sync_in_progress()

    @property
    def sync_epoch(self):
        """Get sync_epoch value via CommonDataClient."""
        return self._common_data_client.get_sync_epoch()

    # ==================== Core Logic Methods ====================

    def reset_debug_counters(self):
        """Reset debug counters."""
        self.n_orders_corrected = 0
        self.miners_corrected = set()

    def _position_is_candidate_for_price_correction(self, position: Position, now_ms: int) -> bool:
        """Check if position is candidate for price correction."""
        return (position.is_open_position or
                position.newest_order_age_ms(now_ms) <= ValiConfig.RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS)

    def get_sorted_price_sources(self, hotkey_positions: Dict[str, List[Position]]) -> Dict[TradePair, List[PriceSource]]:
        """Get sorted price sources for all required trade pairs."""
        try:
            required_trade_pairs_for_candles = set()
            trade_pair_to_market_open = {}
            now_ms = TimeUtil.now_in_millis()

            for sorted_positions in hotkey_positions.values():
                for position in sorted_positions:
                    if self._position_is_candidate_for_price_correction(position, now_ms):
                        tp = position.trade_pair
                        if tp not in trade_pair_to_market_open:
                            trade_pair_to_market_open[tp] = self._live_price_client.is_market_open(tp, now_ms)
                        if trade_pair_to_market_open[tp]:
                            required_trade_pairs_for_candles.add(tp)

            now = TimeUtil.now_in_millis()
            trade_pair_to_price_sources = self._live_price_client.get_tp_to_sorted_price_sources(
                list(required_trade_pairs_for_candles),
                now
            )

            for tp, sources in trade_pair_to_price_sources.items():
                if sources and any(x and not x.websocket for x in sources):
                    self.n_poly_api_requests += 1

            self.last_price_fetch_time_ms = now
            return trade_pair_to_price_sources

        except Exception as e:
            bt.logging.error(f"Error in get_sorted_price_sources: {e}")
            bt.logging.error(traceback.format_exc())
            return {}

    def mdd_check(self, iteration_epoch: int = None):
        """
        Run MDD check with price corrections.

        Args:
            iteration_epoch: Sync epoch captured at start of iteration. Used to detect stale data.
        """
        self.n_poly_api_requests = 0
        if not self.refresh_allowed(ValiConfig.MDD_CHECK_REFRESH_TIME_MS):
            time.sleep(1)
            return

        self.reset_debug_counters()
        self.position_refresh_sum_ms = 0.0
        self.lock_acquisition_sum_ms = 0.0
        self.position_refresh_count = 0

        # Time the RPC read of positions
        rpc_start = time.perf_counter()
        hotkey_to_positions = self._position_client.get_positions_for_hotkeys(
            self._metagraph_client.get_hotkeys(),
            filter_eliminations=True,
            sort_positions=True
        )
        rpc_ms = (time.perf_counter() - rpc_start) * 1000

        total_positions = sum(len(positions) for positions in hotkey_to_positions.values())
        bt.logging.info(
            f"[MDD_RPC_TIMING] get_positions_for_hotkeys RPC read={rpc_ms:.2f}ms, "
            f"total_positions={total_positions}"
        )

        # Time price source fetching
        price_fetch_start = time.perf_counter()
        tp_to_price_sources = self.get_sorted_price_sources(hotkey_to_positions)
        price_fetch_ms = (time.perf_counter() - price_fetch_start) * 1000

        now_ms = TimeUtil.now_in_millis()
        today_date_est = TimeUtil.timestamp_ms_to_eastern_time_str(now_ms, short=True)
        last_update_date_est = TimeUtil.timestamp_ms_to_eastern_time_str(self._last_update_time_ms, short=True)
        is_new_day = today_date_est != last_update_date_est

        # Fetch all stock splits once for efficiency
        stock_splits = {}
        should_check_splits = is_new_day or any(
            tp.is_equities and self._prev_iteration_prices.get(tp) and sources[0].close
            and abs((sources[0].close - self._prev_iteration_prices[tp]) / self._prev_iteration_prices[tp]) >= 0.1
            for tp, sources in tp_to_price_sources.items()
        )
        if should_check_splits:
            bt.logging.info(f"[STOCK SPLITS] Fetching stock splits for {today_date_est}")
            stock_splits = self._live_price_client.get_stock_splits(now_ms)
            if stock_splits:
                bt.logging.info(f"[STOCK SPLITS] Found splits: {stock_splits}")
            else:
                bt.logging.info(f"[STOCK SPLITS] No stock splits found for {today_date_est}")

        # Check and apply if a stock split occurred on new trading day (EST) or price fluctuation > 10%
        for tp, sources in tp_to_price_sources.items():
            if not tp.is_equities:
                continue
            new_price = sources[0].close
            prev_price = self._prev_iteration_prices.get(tp)

            stock_split_ratio = stock_splits.get(tp.trade_pair_id)
            if stock_split_ratio is not None:
                self._position_client.apply_stock_split(tp.trade_pair_id, stock_split_ratio, today_date_est)

            self._prev_iteration_prices[tp] = sources[0].close

        for hotkey, sorted_positions in hotkey_to_positions.items():
            self.perform_price_corrections(hotkey, sorted_positions, tp_to_price_sources, iteration_epoch)

        # Log aggregate timing statistics
        if self.position_refresh_count > 0:
            avg_lock_ms = self.lock_acquisition_sum_ms / self.position_refresh_count
            avg_refresh_ms = self.position_refresh_sum_ms / self.position_refresh_count
            bt.logging.info(
                f"[MDD_RPC_TIMING] price_sources_fetch={price_fetch_ms:.2f}ms, "
                f"positions_refreshed={self.position_refresh_count}, "
                f"avg_lock_wait={avg_lock_ms:.2f}ms, avg_refresh={avg_refresh_ms:.2f}ms"
            )
        else:
            bt.logging.info(f"[MDD_RPC_TIMING] price_sources_fetch={price_fetch_ms:.2f}ms, positions_refreshed=0")

        bt.logging.info(
            f"mdd checker completed. n orders corrected: {self.n_orders_corrected}. "
            f"n miners corrected: {len(self.miners_corrected)}. n_poly_api_requests: {self.n_poly_api_requests}."
        )
        self.set_last_update_time(skip_message=False)

    def update_order_with_newest_price_sources(
        self,
        order,
        candidate_price_sources: List[PriceSource],
        hotkey: str,
        position: Position
    ) -> bool:
        """Update order with newest price sources. Returns True if any changes were made."""
        if not candidate_price_sources:
            return False

        trade_pair = position.trade_pair
        trade_pair_str = trade_pair.trade_pair
        order_time_ms = order.processed_ms
        existing_dict = {ps.source: ps for ps in order.price_sources}
        candidates_dict = {ps.source: ps for ps in candidate_price_sources}
        new_price_sources = []
        any_changes = False

        for k, candidate_ps in candidates_dict.items():
            if k in existing_dict:
                existing_ps = existing_dict[k]
                if candidate_ps.time_delta_from_now_ms(order_time_ms) < existing_ps.time_delta_from_now_ms(order_time_ms):
                    bt.logging.info(
                        f"Found a better price source for {hotkey} {trade_pair_str}! "
                        f"Replacing {existing_ps.debug_str(order_time_ms)} with {candidate_ps.debug_str(order_time_ms)}"
                    )
                    new_price_sources.append(candidate_ps)
                    any_changes = True
                else:
                    new_price_sources.append(existing_ps)
            else:
                bt.logging.info(
                    f"Found a new price source for {hotkey} {trade_pair_str}! Adding {candidate_ps.debug_str(order_time_ms)}"
                )
                new_price_sources.append(candidate_ps)
                any_changes = True

        for k, existing_ps in existing_dict.items():
            if k not in candidates_dict:
                new_price_sources.append(existing_ps)

        new_price_sources = PriceSource.non_null_events_sorted(new_price_sources, order_time_ms)
        winning_event: PriceSource = new_price_sources[0] if new_price_sources else None

        if not winning_event:
            bt.logging.error(f"Could not find a winning event for {hotkey} {trade_pair_str}!")
            return False

        # Try to find a bid/ask for it if it is missing (Polygon and Tiingo equities)
        if winning_event and (not winning_event.bid or not winning_event.ask):
            bid, ask, _ = self._live_price_client.get_quote(trade_pair, order.processed_ms)
            if bid and ask:
                winning_event.bid = bid
                winning_event.ask = ask
                bt.logging.info(f"Found a bid/ask for {hotkey} {trade_pair_str} ps {winning_event}")
                any_changes = True

        if any_changes:
            order.price = winning_event.parse_appropriate_price(order_time_ms, trade_pair.is_forex, order.order_type, position)
            order.bid = winning_event.bid
            order.ask = winning_event.ask
            order.slippage = PriceSlippageModel.calculate_slippage(winning_event.bid, winning_event.ask, order)
            order.price_sources = new_price_sources
            return True

        return False

    def _update_position_returns_and_persist_to_disk(
        self,
        hotkey: str,
        position: Position,
        tp_to_price_sources_for_realtime_price: Dict[TradePair, List[PriceSource]],
        iteration_epoch: int = None
    ):
        """
        Set latest returns and persist to disk for accurate MDD calculation.

        Args:
            hotkey: Miner hotkey
            position: Position to update
            tp_to_price_sources_for_realtime_price: Price sources for realtime price
            iteration_epoch: Epoch captured at start of iteration. If changed, data is stale.
        """
        def _get_sources_for_order(order, trade_pair: TradePair):
            self.n_poly_api_requests += 1

            fetch_start = time.perf_counter()
            price_sources = self._live_price_client.get_sorted_price_sources_for_trade_pair(trade_pair, order.processed_ms)
            fetch_ms = (time.perf_counter() - fetch_start) * 1000

            now_ms = TimeUtil.now_in_millis()
            order_age_ms = now_ms - order.processed_ms

            bt.logging.info(
                f"[MDD_PRICE_TIMING] get_price_sources for order={fetch_ms:.2f}ms, "
                f"order_age={order_age_ms/1000:.1f}s, trade_pair={trade_pair.trade_pair_id}, "
                f"sources_found={len(price_sources) if price_sources else 0}"
            )
            return price_sources

        trade_pair = position.trade_pair
        trade_pair_id = trade_pair.trade_pair_id
        orig_return = position.return_at_close
        orig_avg_price = position.average_entry_price
        orig_iep = position.initial_entry_price
        now_ms = TimeUtil.now_in_millis()

        # Acquire lock and refresh position for TOCTOU protection
        lock_request_time = time.perf_counter()
        with self._position_lock_client.get_lock(hotkey, trade_pair_id):
            lock_acquired_ms = (time.perf_counter() - lock_request_time) * 1000
            bt.logging.trace(f"[MDD_LOCK_TIMING] Lock acquired for {hotkey[:8]}.../{trade_pair_id} in {lock_acquired_ms:.2f}ms")

            # Refresh position inside lock for TOCTOU protection
            refresh_start = time.perf_counter()
            position_refreshed = self._position_client.get_miner_position_by_uuid(hotkey, position.position_uuid)
            refresh_ms = (time.perf_counter() - refresh_start) * 1000

            if position_refreshed is None:
                bt.logging.warning(
                    f"mdd_checker: Position not found (uuid {position.position_uuid[:8]}... "
                    f"for {hotkey[:8]}.../{trade_pair_id}). Skipping."
                )
                return

            # Track timing for aggregate logging
            self.lock_acquisition_sum_ms += lock_acquired_ms
            self.position_refresh_sum_ms += refresh_ms
            self.position_refresh_count += 1
            position = position_refreshed
            n_orders_updated = 0

            for i, order in enumerate(reversed(position.orders)):
                if not self.price_correction_enabled:
                    break

                order_age = now_ms - order.processed_ms
                if order_age > ValiConfig.RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS:
                    break  # No need to check older records

                price_sources_for_retro_fix = _get_sources_for_order(order, position.trade_pair)
                if not price_sources_for_retro_fix:
                    bt.logging.warning(
                        f"Unexpectedly could not find any new price sources for order "
                        f"{order.order_uuid} in {hotkey} {position.trade_pair.trade_pair}. "
                        f"If this issue persists, alert the team."
                    )
                    continue
                else:
                    any_order_updates = self.update_order_with_newest_price_sources(
                        order, price_sources_for_retro_fix, hotkey, position
                    )
                    n_orders_updated += int(any_order_updates)

            # Rebuild the position with the newest price
            if n_orders_updated:
                position.rebuild_position_with_updated_orders(self._live_price_client)
                bt.logging.info(
                    f"Retroactively updated {n_orders_updated} order prices for {position.miner_hotkey} "
                    f"{position.trade_pair.trade_pair} return_at_close changed from {orig_return:.8f} to "
                    f"{position.return_at_close:.8f} avg_price changed from {orig_avg_price:.8f} to "
                    f"{position.average_entry_price:.8f} initial_entry_price changed from {orig_iep:.8f} to "
                    f"{position.initial_entry_price:.8f}"
                )

            temp = tp_to_price_sources_for_realtime_price.get(trade_pair, [])
            realtime_price = temp[0].close if temp else None
            ret_changed = False

            if position.is_open_position and realtime_price is not None:
                orig_return = position.return_at_close
                position.set_returns(realtime_price, self._live_price_client)
                ret_changed = orig_return != position.return_at_close

            if n_orders_updated or ret_changed:
                # Epoch-based validation: check if sync occurred during our iteration
                if iteration_epoch is not None:
                    current_epoch = self.sync_epoch
                    if current_epoch != iteration_epoch:
                        bt.logging.warning(
                            f"Sync occurred during MDDChecker iteration for {hotkey} {trade_pair_id} "
                            f"(epoch {iteration_epoch} -> {current_epoch}). "
                            f"Skipping save to avoid data corruption"
                        )
                        return

                is_liquidated = position.current_return == 0
                self._position_client.save_miner_position(position, delete_open_position_if_exists=is_liquidated)
                self.n_orders_corrected += n_orders_updated
                self.miners_corrected.add(hotkey)

    def perform_price_corrections(
        self,
        hotkey: str,
        sorted_positions: List[Position],
        tp_to_price_sources: Dict[TradePair, List[PriceSource]],
        iteration_epoch: int = None
    ) -> bool:
        """Perform price corrections for a miner's positions."""
        if len(sorted_positions) == 0:
            return False

        now_ms = TimeUtil.now_in_millis()
        for position in sorted_positions:
            is_candidate = self._position_is_candidate_for_price_correction(position, now_ms)
            if is_candidate:
                self._update_position_returns_and_persist_to_disk(
                    hotkey, position, tp_to_price_sources, iteration_epoch
                )

        return False
