import time
from typing import List, Tuple, Dict

import numpy as np
from data_generator.tiingo_data_service import TiingoDataService
from data_generator.polygon_data_service import PolygonDataService
from time_util.time_util import TimeUtil
from vali_objects.utils.vali_utils import ValiUtils
from vali_objects.vali_config import TradePair, ValiConfig
import bittensor as bt
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from vali_objects.vali_dataclasses.order import OrderSource
from vali_objects.vali_dataclasses.price_source import PriceSource


class LivePriceFetcher:
    def __init__(self, secrets, disable_ws=False, is_backtesting=False, running_unit_tests=False):
        self.is_backtesting = is_backtesting
        self.running_unit_tests = running_unit_tests
        self.last_health_check_ms = 0
        if "tiingo_apikey" in secrets:
            self.tiingo_data_service = TiingoDataService(api_key=secrets["tiingo_apikey"], disable_ws=disable_ws,
                                                         running_unit_tests=running_unit_tests)
        else:
            raise Exception("Tiingo API key not found in secrets.json")
        if "polygon_apikey" in secrets:
            self.polygon_data_service = PolygonDataService(api_key=secrets["polygon_apikey"], disable_ws=disable_ws,
                                                           is_backtesting=is_backtesting, running_unit_tests=running_unit_tests)
        else:
            raise Exception("Polygon API key not found in secrets.json")

    def stop_all_threads(self):
        self.tiingo_data_service.stop_threads()
        self.polygon_data_service.stop_threads()

    def set_test_price_source(self, trade_pair: TradePair, price_source: PriceSource) -> None:
        """
        Test-only method to inject price sources for specific trade pairs.
        Delegates to PolygonDataService.
        """
        self.polygon_data_service.set_test_price_source(trade_pair, price_source)

    def clear_test_price_sources(self) -> None:
        """Clear all test price sources. Delegates to PolygonDataService."""
        self.polygon_data_service.clear_test_price_sources()

    def set_test_market_open(self, is_open: bool) -> None:
        """
        Test-only method to override market open status.
        When set, all markets will return this status regardless of actual time.
        """
        self.polygon_data_service.set_test_market_open(is_open)

    def clear_test_market_open(self) -> None:
        """Clear market open override and use real calendar."""
        self.polygon_data_service.clear_test_market_open()

    def set_test_candle_data(self, trade_pair: TradePair, start_ms: int, end_ms: int, candles: List[PriceSource]) -> None:
        """
        Test-only method to inject candle data for specific trade pair and time window.
        Delegates to PolygonDataService.
        """
        self.polygon_data_service.set_test_candle_data(trade_pair, start_ms, end_ms, candles)

    def clear_test_candle_data(self) -> None:
        """Clear all test candle data. Delegates to PolygonDataService."""
        self.polygon_data_service.clear_test_candle_data()

    def health_check(self) -> dict:
        """
        Health check method for RPC connection between client and server.
        Returns a simple status indicating the server is alive and responsive.
        """
        current_time_ms = TimeUtil.now_in_millis()
        return {
            "status": "ok",
            "timestamp_ms": current_time_ms,
            "is_backtesting": self.is_backtesting
        }

    def is_market_open(self, trade_pair: TradePair, time_ms=None) -> bool:
        """
        Check if market is open for a trade pair.

        Args:
            trade_pair: The trade pair to check
            time_ms: Optional timestamp in milliseconds (defaults to now)

        Returns:
            bool: True if market is open, False otherwise
        """
        if time_ms is None:
            time_ms = TimeUtil.now_in_millis()
        return self.polygon_data_service.is_market_open(trade_pair, time_ms)

    def get_unsupported_trade_pairs(self):
        """
        Return static tuple of unsupported trade pairs without RPC overhead.

        These trade pairs are permanently unsupported (not temporarily halted),
        so no need to fetch from polygon_data_service on every call.

        Returns:
            Tuple of TradePair constants that are unsupported
        """
        # Return ValiConfig constant
        return ValiConfig.UNSUPPORTED_TRADE_PAIRS

    def get_currency_conversion(self, base: str, quote: str):
        return self.polygon_data_service.get_currency_conversion(base=base, quote=quote)

    def unified_candle_fetcher(self, trade_pair, start_date, order_date, timespan="day"):
        return self.polygon_data_service.unified_candle_fetcher(trade_pair, start_date, order_date, timespan=timespan)

    def sorted_valid_price_sources(self, price_events: List[PriceSource | None], current_time_ms: int, filter_recent_only=True) -> List[PriceSource] | None:
        """
        Sorts a list of price events by their recency and validity.
        """
        valid_events = [event for event in price_events if event]
        if not valid_events:
            return None

        if not current_time_ms:
            current_time_ms = TimeUtil.now_in_millis()

        best_event = PriceSource.get_winning_event(valid_events, current_time_ms)
        if not best_event:
            return None

        # DEBUG: Log filtering decision
        time_delta = best_event.time_delta_from_now_ms(current_time_ms)
        with open('/tmp/mdd_debug.log', 'a') as f:
            f.write(f"[DEBUG] sorted_valid_price_sources: filter_recent_only={filter_recent_only}, time_delta={time_delta}ms\n")
            f.flush()
        if filter_recent_only and time_delta > 8000:
            with open('/tmp/mdd_debug.log', 'a') as f:
                f.write(f"[DEBUG] FILTERED OUT: delta={time_delta}ms > 8000ms, source_time={best_event.start_ms}, query_time={current_time_ms}\n")
                f.flush()
            return None

        return PriceSource.non_null_events_sorted(valid_events, current_time_ms)

    def dual_rest_get(self, trade_pairs: List[TradePair], time_ms, live) -> Tuple[Dict[TradePair, PriceSource], Dict[TradePair, PriceSource]]:
        """
        Fetch REST closes from both Polygon and Tiingo in parallel,
        using ThreadPoolExecutor to run both calls concurrently.
        """
        polygon_results = {}
        tiingo_results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both REST calls to the executor
            poly_fut = executor.submit(self.polygon_data_service.get_closes_rest, trade_pairs, time_ms, live)
            tiingo_fut = executor.submit(self.tiingo_data_service.get_closes_rest, trade_pairs, time_ms, live)

            try:
                # Wait for both futures to complete with a 10s timeout
                polygon_results = poly_fut.result(timeout=10)
                tiingo_results = tiingo_fut.result(timeout=10)
            except FuturesTimeoutError:
                poly_fut.cancel()
                tiingo_fut.cancel()
                bt.logging.warning(f"dual_rest_get REST API requests timed out. trade_pairs: {trade_pairs}.")

        return polygon_results, tiingo_results

    def get_ws_price_sources_in_window(self, trade_pair: TradePair, start_ms: int, end_ms: int) -> List[PriceSource]:
        # Utilize get_events_in_range
        poly_sources = self.polygon_data_service.trade_pair_to_recent_events[trade_pair.trade_pair].get_events_in_range(start_ms, end_ms)
        t_sources = self.tiingo_data_service.trade_pair_to_recent_events[trade_pair.trade_pair].get_events_in_range(start_ms, end_ms)
        return poly_sources + t_sources

    def get_latest_price(self, trade_pair: TradePair, time_ms=None) -> Tuple[float, List[PriceSource]] | Tuple[None, None]:
        """
        Gets the latest price for a single trade pair by utilizing WebSocket and possibly REST data sources.
        Tries to get the price as close to time_ms as possible.
        """
        price_sources = self.get_sorted_price_sources_for_trade_pair(trade_pair, time_ms)
        winning_event = PriceSource.get_winning_event(price_sources, time_ms)
        return winning_event.parse_best_best_price_legacy(time_ms), price_sources

    def get_sorted_price_sources_for_trade_pair(self, trade_pair: TradePair, time_ms: int, live=True) -> List[PriceSource] | None:
        temp = self.get_tp_to_sorted_price_sources([trade_pair], time_ms, live)
        return temp.get(trade_pair)

    def get_tp_to_sorted_price_sources(self, trade_pairs: List[TradePair], time_ms: int, live=True) -> Dict[TradePair, List[PriceSource]]:
        """
        Retrieves the latest prices for multiple trade pairs, leveraging both WebSocket and REST APIs as needed.
        """
        if not time_ms:
            time_ms = TimeUtil.now_in_millis()

        websocket_prices_polygon = self.polygon_data_service.get_closes_websocket(trade_pairs, time_ms)
        websocket_prices_tiingo_data = self.tiingo_data_service.get_closes_websocket(trade_pairs, time_ms)
        trade_pairs_needing_rest_data = []

        results = {}

        # Initial check using WebSocket data
        for trade_pair in trade_pairs:
            events = [websocket_prices_polygon.get(trade_pair), websocket_prices_tiingo_data.get(trade_pair)]
            sources = self.sorted_valid_price_sources(events, time_ms, filter_recent_only=True)
            if sources:
                results[trade_pair] = sources
            else:
                trade_pairs_needing_rest_data.append(trade_pair)

        # Fetch from REST APIs if needed
        if not trade_pairs_needing_rest_data:
            return results

        rest_prices_polygon, rest_prices_tiingo_data = self.dual_rest_get(trade_pairs_needing_rest_data, time_ms, live)

        for trade_pair in trade_pairs_needing_rest_data:
            sources = self.sorted_valid_price_sources([
                websocket_prices_polygon.get(trade_pair),
                websocket_prices_tiingo_data.get(trade_pair),
                rest_prices_polygon.get(trade_pair),
                rest_prices_tiingo_data.get(trade_pair)
            ], time_ms, filter_recent_only=False)
            results[trade_pair] = sources

        return results

    def time_since_last_ws_ping_s(self, trade_pair: TradePair) -> float | None:
        if trade_pair in self.polygon_data_service.UNSUPPORTED_TRADE_PAIRS:
            return None
        now_ms = TimeUtil.now_in_millis()
        t1 = self.polygon_data_service.get_websocket_lag_for_trade_pair_s(tp=trade_pair.trade_pair, now_ms=now_ms)
        t2 = self.tiingo_data_service.get_websocket_lag_for_trade_pair_s(tp=trade_pair.trade_pair, now_ms=now_ms)
        return max([x for x in (t1, t2) if x])

    def filter_outliers(self, unique_data: List[PriceSource]) -> List[PriceSource]:
        """
        Filters out outliers and duplicates from a list of price sources.
        """
        if not unique_data:
            return []

        # Function to calculate bounds
        def calculate_bounds(prices):
            median_val = np.median(prices)
            # Calculate bounds as 5% less than and more than the median
            lower_bound = median_val * 0.95
            upper_bound = median_val * 1.05
            return lower_bound, upper_bound

        # Calculate bounds for each price type
        close_prices = np.array([x.close for x in unique_data])
        # high_prices = np.array([x.high for x in unique_data])
        # low_prices = np.array([x.low for x in unique_data])

        close_lower_bound, close_upper_bound = calculate_bounds(close_prices)
        # high_lower_bound, high_upper_bound = calculate_bounds(high_prices)
        # low_lower_bound, low_upper_bound = calculate_bounds(low_prices)

        # Filter data by checking all price points against their respective bounds
        filtered_data = [x for x in unique_data if close_lower_bound <= x.close <= close_upper_bound]
        # filtered_data = [x for x in unique_data if close_lower_bound <= x.close <= close_upper_bound and
        #                 high_lower_bound <= x.high <= high_upper_bound and
        #                 low_lower_bound <= x.low <= low_upper_bound]

        # Sort the data by timestamp in ascending order
        filtered_data.sort(key=lambda x: x.start_ms, reverse=True)
        return filtered_data


    def get_quote(self, trade_pair: TradePair, processed_ms: int) -> (float, float, int):
        """
        returns the bid and ask quote for a trade_pair at processed_ms. Only Polygon supports point-in-time bid/ask.
        """
        return self.polygon_data_service.get_quote(trade_pair, processed_ms)

    def get_candles(self, trade_pairs, start_time_ms, end_time_ms) -> dict:
        ans = {}
        debug = {}
        one_second_rest_candles = self.polygon_data_service.get_candles(
            trade_pairs=trade_pairs, start_time_ms=start_time_ms, end_time_ms=end_time_ms)

        for tp in trade_pairs:
            rest_candles = one_second_rest_candles.get(tp, [])
            ws_candles = self.get_ws_price_sources_in_window(tp, start_time_ms, end_time_ms)
            non_null_sources = list(set(rest_candles + ws_candles))
            filtered_sources = self.filter_outliers(non_null_sources)
            # Get the sources removed to debug
            removed_sources = [x for x in non_null_sources if x not in filtered_sources]
            ans[tp] = filtered_sources
            min_time = min((x.start_ms for x in non_null_sources)) if non_null_sources else 0
            max_time = max((x.end_ms for x in non_null_sources)) if non_null_sources else 0
            debug[
                tp.trade_pair] = f"R{len(rest_candles)}W{len(ws_candles)}U{len(non_null_sources)}T[{(max_time - min_time) / 1000.0:.2f}]"
            if removed_sources:
                mi = min((x.close for x in non_null_sources))
                ma = max((x.close for x in non_null_sources))
                debug[tp.trade_pair] += f" Removed {[x.close for x in removed_sources]} Original min/max {mi}/{ma}"

        bt.logging.info(f"Fetched candles {debug} in window"
                        f" {TimeUtil.millis_to_formatted_date_str(start_time_ms)} to "
                        f"{TimeUtil.millis_to_formatted_date_str(end_time_ms)}")

        # If Polygon has any missing keys, it is intentional and corresponds to a closed market. We don't want to use twelvedata for this TODO: fall back to live price from TD/POLY.
        # if self.twelvedata_available and len(ans) == 0:
        #    bt.logging.info(f"Fetching candles from TD for {[x.trade_pair for x in trade_pairs]} from {start_time_ms} to {end_time_ms}")
        #    closes = self.twelve_data.get_closes(trade_pairs=trade_pairs)
        #    ans.update(closes)
        return ans

    def get_close_at_date(self, trade_pair, timestamp_ms, order=None, verbose=True):
        if self.is_backtesting:
            assert order, 'Must provide order for validation during backtesting'

        price_source = None
        if not self.polygon_data_service.is_market_open(trade_pair, time_ms=timestamp_ms):
            if self.is_backtesting and order and order.src == 0:
                raise Exception(f"Backtesting validation failure: Attempting to price fill during closed market. TP {trade_pair.trade_pair_id} at {TimeUtil.millis_to_formatted_date_str(timestamp_ms)}")
            else:
                price_source = self.polygon_data_service.get_event_before_market_close(trade_pair, timestamp_ms)
                print(f'Used previous close to fill price for {trade_pair.trade_pair_id} at {TimeUtil.millis_to_formatted_date_str(timestamp_ms)}')

        if price_source is None:
            price_source = self.polygon_data_service.get_close_at_date_second(trade_pair=trade_pair, target_timestamp_ms=timestamp_ms)
        if price_source is None:
            price_source = self.polygon_data_service.get_close_at_date_minute_fallback(trade_pair=trade_pair, target_timestamp_ms=timestamp_ms)
            if price_source:
                bt.logging.warning(
                    f"Fell back to Polygon get_date_minute_fallback for price of {trade_pair.trade_pair} at {TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)}, price_source: {price_source}")

        if price_source is None:
            price_source = self.tiingo_data_service.get_close_rest(trade_pair=trade_pair, timestamp_ms=timestamp_ms, live=False)
            if verbose and price_source is not None:
                bt.logging.warning(
                    f"Fell back to Tiingo get_date for price of {trade_pair.trade_pair} at {TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)}, ms: {timestamp_ms}")

        """
        if price is None:
            price, time_delta = self.polygon_data_service.get_close_in_past_hour_fallback(trade_pair=trade_pair,
                                                                             timestamp_ms=timestamp_ms)
            if price:
                formatted_date = TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)
                bt.logging.warning(
                    f"Fell back to Polygon get_close_in_past_hour_fallback for price of {trade_pair.trade_pair} at {formatted_date}, ms: {timestamp_ms}")
        if price is None:
            formatted_date = TimeUtil.timestamp_ms_to_eastern_time_str(timestamp_ms)
            bt.logging.error(
                f"Failed to get data at ET date {formatted_date} for {trade_pair.trade_pair}. Timestamp ms: {timestamp_ms}."
                f" Ask a team member to investigate this issue.")
        """
        return price_source

    def get_quote_usd_conversion(self, order, position):
        """
        Return the conversion rate between an order's quote currency and USD
        """
        if order.price == 0:
            return 0.0

        if not (order.trade_pair.is_forex and order.trade_pair.quote != "USD"):
            return 1.0

        if order.trade_pair.base == "USD":
            return 1.0 / order.price

        # A/B cross pair: need to convert quote currency B to USD
        # Try B/USD first (more common)
        b_usd = True
        conversion_trade_pair = TradePair.from_trade_pair_id(f"{order.trade_pair.quote}USD")
        if conversion_trade_pair is None:
            # fall back to USD/B format
            b_usd = False
            conversion_trade_pair = TradePair.from_trade_pair_id(f"USD{order.trade_pair.quote}")

        price_sources = self.get_sorted_price_sources_for_trade_pair(
            trade_pair=conversion_trade_pair,
            time_ms=order.processed_ms
        )
        if price_sources and len(price_sources) > 0:
            best_price_source = price_sources[0]
            usd_conversion = best_price_source.parse_appropriate_price(
                now_ms=order.processed_ms,
                is_forex=True,          # from_currency is USD for crypto and equities
                order_type=order.order_type,
                position=position
            )
            return usd_conversion if b_usd else 1.0 / usd_conversion

        bt.logging.error(f"Unable to fetch quote currency {order.trade_pair.quote} to USD conversion at time {order.processed_ms}. No price sources available (websocket or REST).")
        return 1.0
        # TODO: raise Exception(f"Unable to fetch currency conversion from {from_currency} to USD at time {time_ms}.")

    def get_usd_base_conversion(self, trade_pair, time_ms, price, order_type, position):
        """
        Return the conversion rate between USD and an order's base currency
        """
        if price == 0:
            return 0.0

        if trade_pair.base == "USD":
            return 1.0

        if not trade_pair.is_forex or trade_pair.quote == "USD":
            return 1.0 / price

        # A/B cross pair: need to convert usd to base currency A
        # Try USD/A first (more common)
        usd_a = True
        conversion_trade_pair = TradePair.from_trade_pair_id(f"USD{trade_pair.base}")
        if conversion_trade_pair is None:
            # fall back to A/USD format
            usd_a = False
            conversion_trade_pair = TradePair.from_trade_pair_id(f"{trade_pair.base}USD")

        price_sources = self.get_sorted_price_sources_for_trade_pair(
            trade_pair=conversion_trade_pair,
            time_ms=time_ms
        )
        if price_sources and len(price_sources) > 0:
            best_price_source = price_sources[0]
            usd_conversion = best_price_source.parse_appropriate_price(
                now_ms=time_ms,
                is_forex=True,          # from_currency is USD for crypto and equities
                order_type=order_type,
                position=position
            )
            return usd_conversion if usd_a else 1.0 / usd_conversion

        bt.logging.error(f"Unable to fetch USD to base currency {trade_pair.base} conversion at time {time_ms}. No price sources available (websocket or REST).")
        return 1.0


if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()
    live_price_fetcher = LivePriceFetcher(secrets, disable_ws=True)
    ans = live_price_fetcher.get_close_at_date(TradePair.TAOUSD, 1733304060475)
    print('@@@@', ans, '@@@@@')
    time.sleep(100000)

    trade_pairs = [TradePair.BTCUSD, TradePair.ETHUSD, ]
    now_ms = TimeUtil.now_in_millis()
    while True:
        for tp in TradePair:
            print(f"{tp.trade_pair}: {live_price_fetcher.get_close_at_date(tp, now_ms)}")
        time.sleep(10)
    # ans = live_price_fetcher.get_closes(trade_pairs)
    # for k, v in ans.items():
    #    print(f"{k.trade_pair_id}: {v}")
    # print("Done")