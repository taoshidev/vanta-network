# developer: jbonilla
# Copyright © 2024 Taoshi Inc
import threading
import time
import traceback
from typing import List, Dict

from time_util.time_util import TimeUtil
from vali_objects.vali_config import ValiConfig, TradePair
from shared_objects.cache_controller import CacheController
from vali_objects.position import Position
from vali_objects.utils.live_price_fetcher import LivePriceFetcher

from vali_objects.utils.vali_utils import ValiUtils

import bittensor as bt

from vali_objects.vali_dataclasses.price_source import PriceSource

class MDDChecker(CacheController):

    def __init__(self, metagraph, position_manager, running_unit_tests=False,
                 live_price_fetcher=None, shutdown_dict=None):
        super().__init__(metagraph, running_unit_tests=running_unit_tests)
        self.last_price_fetch_time_ms = None
        self.last_quote_fetch_time_ms = None
        self.price_correction_enabled = True
        secrets = ValiUtils.get_secrets(running_unit_tests=running_unit_tests)
        self.position_manager = position_manager
        assert self.running_unit_tests == self.position_manager.running_unit_tests
        self.all_trade_pairs = [trade_pair for trade_pair in TradePair]
        if live_price_fetcher is None:
            self.live_price_fetcher = LivePriceFetcher(secrets=secrets)
        else:
            self.live_price_fetcher = live_price_fetcher
        self.elimination_manager = position_manager.elimination_manager
        self.reset_debug_counters()
        self.shutdown_dict = shutdown_dict
        self.n_poly_api_requests = 0

    def reset_debug_counters(self):
        self.n_orders_corrected = 0
        self.miners_corrected = set()

    def _position_is_candidate_for_price_correction(self, position: Position, now_ms):
        return (position.is_open_position or
                position.newest_order_age_ms(now_ms) <= ValiConfig.RECENT_EVENT_TRACKER_OLDEST_ALLOWED_RECORD_MS)

    def get_sorted_price_sources(self, hotkey_positions) -> Dict[TradePair, List[PriceSource]]:
        try:
            required_trade_pairs_for_candles = set()
            trade_pair_to_market_open = {}
            now_ms = TimeUtil.now_in_millis()
            for sorted_positions in hotkey_positions.values():
                for position in sorted_positions:
                    # Only need live price for open positions in open markets.
                    if self._position_is_candidate_for_price_correction(position, now_ms):
                        tp = position.trade_pair
                        if tp not in trade_pair_to_market_open:
                            trade_pair_to_market_open[tp] = self.live_price_fetcher.polygon_data_service.is_market_open(tp)
                        if trade_pair_to_market_open[tp]:
                            required_trade_pairs_for_candles.add(tp)

            now = TimeUtil.now_in_millis()
            trade_pair_to_price_sources = self.live_price_fetcher.get_tp_to_sorted_price_sources(list(required_trade_pairs_for_candles))
            #bt.logging.info(f"Got candle data for {len(candle_data)} {candle_data}")
            for tp, sources in trade_pair_to_price_sources.items():
                if sources and any(x and not x.websocket for x in sources):
                    self.n_poly_api_requests += 1

            self.last_price_fetch_time_ms = now
            return trade_pair_to_price_sources
        except Exception as e:
            bt.logging.error(f"Error in get_sorted_price_sources: {e}")
            bt.logging.error(traceback.format_exc())
            return {}

    
    def mdd_check(self, position_locks):
        self.n_poly_api_requests = 0
        if not self.refresh_allowed(ValiConfig.MDD_CHECK_REFRESH_TIME_MS):
            time.sleep(1)
            return

        if self.shutdown_dict:
            return

        bt.logging.info("running mdd checker")
        self.reset_debug_counters()

        hotkey_to_positions = self.position_manager.get_positions_for_hotkeys(
            self.metagraph.hotkeys, sort_positions=True,
            eliminations=self.elimination_manager.get_eliminations_from_memory(),
        )
        tp_to_price_sources = self.get_sorted_price_sources(hotkey_to_positions)
        for hotkey, sorted_positions in hotkey_to_positions.items():
            if self.shutdown_dict:
                return
            self.perform_price_corrections(hotkey, sorted_positions, tp_to_price_sources, position_locks)

        bt.logging.info(f"mdd checker completed."
                        f" n orders corrected: {self.n_orders_corrected}. n miners corrected: {len(self.miners_corrected)}."
                        f" n_poly_api_requests: {self.n_poly_api_requests}")
        self.set_last_update_time(skip_message=False)

    def update_order_with_newest_price_sources(self, order, candidate_price_sources, hotkey, position) -> bool:
        from vali_objects.utils.price_slippage_model import PriceSlippageModel

        if not candidate_price_sources:
            return False
        trade_pair = position.trade_pair
        trade_pair_str = trade_pair.trade_pair
        order_time_ms = order.processed_ms
        existing_dict = {ps.source: ps for ps in order.price_sources}
        candidates_dict = {ps.source: ps for ps in candidate_price_sources}
        new_price_sources = []
        # We need to create new price sources. If there is overlap, take the one with the smallest time lag to order_time_ms
        any_changes = False
        for k, candidate_ps in candidates_dict.items():
            if k in existing_dict:
                existing_ps = existing_dict[k]
                if candidate_ps.time_delta_from_now_ms(order_time_ms) < existing_ps.time_delta_from_now_ms(
                        order_time_ms):  # Prefer the ws price in the past rather than the future
                    bt.logging.info(
                        f"Found a better price source for {hotkey} {trade_pair_str}! Replacing {existing_ps.debug_str(order_time_ms)} with {candidate_ps.debug_str(order_time_ms)}")
                    new_price_sources.append(candidate_ps)
                    any_changes = True
                else:
                    new_price_sources.append(existing_ps)
            else:
                bt.logging.info(
                    f"Found a new price source for {hotkey} {trade_pair_str}! Adding {candidate_ps.debug_str(order_time_ms)}")
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
            bid, ask, _ = self.live_price_fetcher.get_quote(trade_pair, order.processed_ms)
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


    def _update_position_returns_and_persist_to_disk(self, hotkey, position, tp_to_price_sources_for_realtime_price: Dict[TradePair, List[PriceSource]], position_locks):
        """
        Setting the latest returns and persisting to disk for accurate MDD calculation and logging in get_positions

        Won't account for a position that was added in the time between mdd_check being called and this function
        being called. But that's ok as we will process such new positions the next round.
        """

        def _get_sources_for_order(order, trade_pair: TradePair):
            # Only fall back to REST if the order is the latest. Don't want to get slowed down
            # By a flurry of recent orders.
            #ws_only = not is_last_order
            self.n_poly_api_requests += 1#0 if ws_only else 1
            price_sources = self.live_price_fetcher.get_sorted_price_sources_for_trade_pair(trade_pair, order.processed_ms)
            return price_sources

        trade_pair = position.trade_pair
        trade_pair_id = trade_pair.trade_pair_id
        orig_return = position.return_at_close
        orig_avg_price = position.average_entry_price
        orig_iep = position.initial_entry_price
        now_ms = TimeUtil.now_in_millis()
        with (position_locks.get_lock(hotkey, trade_pair_id)):
            # Position could have updated in the time between mdd_check being called and this function being called
            position_refreshed = self.position_manager.get_miner_position_by_uuid(hotkey, position.position_uuid)
            if position_refreshed is None:
                bt.logging.warning(f"mdd_checker: Unexpectedly could not find position with uuid "
                                   f"{position.position_uuid} for hotkey {hotkey} and trade pair {trade_pair_id}.")
                return
            if not self._position_is_candidate_for_price_correction(position_refreshed, now_ms):
                bt.logging.warning(f'mdd_checker: Position with uuid {position.position_uuid} for hotkey {hotkey} '
                                   f'and trade pair {trade_pair_id} is no longer a candidate for price correction.')
                return
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
                    bt.logging.warning(f"Unexpectedly could not find any new price sources for order"
                                     f" {order.order_uuid} in {hotkey} {position.trade_pair.trade_pair}. If this"
                                     f" issue persists, alert the team.")
                    continue
                else:
                    any_order_updates = self.update_order_with_newest_price_sources(order, price_sources_for_retro_fix, hotkey, position)
                    n_orders_updated += int(any_order_updates)

            # Rebuild the position with the newest price
            if n_orders_updated:
                position.rebuild_position_with_updated_orders(self.live_price_fetcher)
                bt.logging.info(f"Retroactively updated {n_orders_updated} order prices for {position.miner_hotkey} {position.trade_pair.trade_pair}  "
                                    f"return_at_close changed from {orig_return:.8f} to {position.return_at_close:.8f} "
                                    f"avg_price changed from {orig_avg_price:.8f} to {position.average_entry_price:.8f} "
                                   f"initial_entry_price changed from {orig_iep:.8f} to {position.initial_entry_price:.8f}")

            # Log return before calling set_returns
            #bt.logging.info(f"current return with fees for open position with trade pair[{open_position.trade_pair.trade_pair_id}] is [{open_position.return_at_close}]. Position: {position}")
            temp = tp_to_price_sources_for_realtime_price.get(trade_pair, [])
            realtime_price = temp[0].close if temp else None
            ret_changed = False
            if position.is_open_position and realtime_price is not None:
                orig_return = position.return_at_close

                position.set_returns(realtime_price, self.position_manager.live_price_fetcher)
                ret_changed = orig_return != position.return_at_close

            if n_orders_updated or ret_changed:
                is_liquidated = position.current_return == 0
                self.position_manager.save_miner_position(position, delete_open_position_if_exists=is_liquidated)
                self.n_orders_corrected += n_orders_updated
                self.miners_corrected.add(hotkey)


    def perform_price_corrections(self, hotkey, sorted_positions, tp_to_price_sources: Dict[TradePair, List[PriceSource]], position_locks) -> bool:
        if len(sorted_positions) == 0:
            return False

        now_ms = TimeUtil.now_in_millis()
        for position in sorted_positions:
            if self.shutdown_dict:
                return False
            # Perform needed updates
            if self._position_is_candidate_for_price_correction(position, now_ms):
                self._update_position_returns_and_persist_to_disk(hotkey, position, tp_to_price_sources, position_locks)








                

