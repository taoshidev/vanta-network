import threading
import traceback
from multiprocessing import Process

import requests

from typing import List

from vali_objects.vali_dataclasses.order import Order
from polygon.websocket import Market, EquityAgg, EquityTrade, CryptoTrade, ForexQuote, WebSocketClient, Feed
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_generator.base_data_service import BaseDataService, POLYGON_PROVIDER_NAME
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory
import time

from vali_objects.utils.vali_utils import ValiUtils
import bittensor as bt
from polygon import RESTClient

from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker

DEBUG = 0

class Agg:
    """
    An efficient representation of an aggregate price data point. Use this over the Polygon Agg for speed.
    """
    def __init__(self, open, close, high, low, vwap, timestamp, bid, ask, volume):
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.bid = bid
        self.ask = ask
        self.vwap = vwap
        self.timestamp = timestamp
        self.volume = volume

    def __str__(self):
        return (
            f"Agg("
            f"open={self.open}, close={self.close}, high={self.high}, low={self.low}, "
            f"bid={self.bid}, ask={self.ask}, vwap={self.vwap}, timestamp={self.timestamp}, vol={self.volume})"
        )

    # Optional: Add a __repr__ method for better representation in debugging and interactive sessions
    def __repr__(self):
        return (
            f"Agg("
            f"open={self.open}, close={self.close}, high={self.high}, low={self.low}, "
            f"bid={self.bid}, ask={self.ask}, vwap={self.vwap}, timestamp={self.timestamp}, vol={self.volume})"
        )


class ExchangeMappingHelper:
    def __init__(self, api_key, fetch_live_mapping=True):
        self.fetch_live_mapping = fetch_live_mapping
        self.api_key = api_key
        self.crypto_fallback_mapping = {
            'coinbase': 1,
            'bitfinex': 2,
            'bitstamp': 6,
            'binance': 10,
            'kraken': 23
        }
        self.stock_fallback_mapping = {
            "nyse american, llc": 1,
            "nasdaq omx bx, inc.": 2,
            "nyse national, inc.": 3,
            "finra alternative display facility": 4,
            "unlisted trading privileges": 5,
            "international securities exchange, llc - stocks": 6,
            "cboe edga": 7,
            "cboe edgx": 8,
            "nyse chicago, inc.": 9,
            "new york stock exchange": 10,
            "nyse arca, inc.": 11,
            "nasdaq": 12,
            "consolidated tape association": 13,
            "long-term stock exchange": 14,
            "investors exchange": 15
        }
        self.crypto_mapping = {}
        self.stock_mapping = {}

        self.create_crypto_mapping()
        self.create_stock_mapping()

    def create_crypto_mapping(self):
        if not self.fetch_live_mapping:
            self.crypto_mapping = self.crypto_fallback_mapping
            return
        endpoint = "https://api.polygon.io/v3/reference/exchanges"
        params = {
            "asset_class": "crypto",
            "apiKey": self.api_key
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the response
            data = response.json()
            if 'results' in data and isinstance(data['results'], list):
                self.crypto_mapping = {
                    entry['name'].lower(): entry['id']
                    for entry in data['results']
                }
                print("Successfully created crypto mapping from API.")
            else:
                print("Unexpected response structure. Using fallback mapping.")
                self.crypto_mapping = self.crypto_fallback_mapping

        except Exception as e:
            print(f"API request failed: {e}. Using fallback mapping.")
            self.crypto_mapping = self.crypto_fallback_mapping

    def create_stock_mapping(self):
        if not self.fetch_live_mapping:
            self.stock_mapping = self.stock_fallback_mapping
            return
        endpoint = "https://api.polygon.io/v3/reference/exchanges"
        params = {
            "asset_class": "stocks",
            "apiKey": self.api_key
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the response
            data = response.json()
            if 'results' in data and isinstance(data['results'], list):
                self.stock_mapping = {
                    entry['name'].lower(): entry['id']
                    for entry in data['results']
                }
                print("Successfully created stock mapping from API.")
            else:
                print("Unexpected response structure. Using fallback mapping.")
                self.stock_mapping = self.stock_fallback_mapping

        except Exception as e:
            print(f"API request failed: {e}. Using fallback mapping.")
            self.stock_mapping = self.stock_fallback_mapping



class PolygonDataService(BaseDataService):

    def __init__(self, api_key, disable_ws=False, ipc_manager=None, is_backtesting=False):
        self.init_time = time.time()
        self._api_key = api_key
        ehm = ExchangeMappingHelper(api_key, fetch_live_mapping = not disable_ws)
        self.crypto_mapping = ehm.crypto_mapping
        self.equities_mapping = ehm.stock_mapping
        self.disable_ws = disable_ws
        self.N_CANDLES_LIMIT = 50000
        self.tp_to_mfs = {}
        self.is_backtesting = is_backtesting
        self.stocks_feed_round_robin_map = {0: Feed.RealTime, 1: Feed.Business}
        self.stocks_feed_round_robin_counter = 0

        super().__init__(provider_name=POLYGON_PROVIDER_NAME, ipc_manager=ipc_manager)

        self.MARKET_STATUS = None

        self.POLYGON_CLIENT = None  # Instantiate later to allow process to start (non picklable)

        # Start thread to refresh market status
        if disable_ws:
            self.websocket_manager_thread = None
        else:
            if ipc_manager:
                self.websocket_manager_thread = Process(target=self.websocket_manager, daemon=True)
            else:
                self.websocket_manager_thread = threading.Thread(target=self.websocket_manager, daemon=True)
            self.websocket_manager_thread.start()

    def parse_price_for_forex(self, m, stats=None, is_ws=False) -> (float, float, float):
        t_ms = m.timestamp if is_ws else m.participant_timestamp // 1000000
        delta = abs(m.bid_price - m.ask_price) / m.bid_price * 100.0
        if stats:
            stats['n'] += 1
            stats['sum_deltas'] += delta
            stats['avg_delta'] = stats['sum_deltas'] / (stats['n'])
            stats['max_delta'] = max(stats['max_delta'], delta)
        if delta > .20:  # Wonky
            if stats:
                stats['n_skipped'] += 1
                if stats['n'] % 10 == 0:
                    bt.logging.warning(f"Ignoring unusual Forex price data bid: {m.bid_price:.4f}, ask: {m.ask_price:.4f}, "
                                   f"{delta:.4f} time {TimeUtil.millis_to_formatted_date_str(t_ms // 1000000)}")
            return None, None, None
        #elif stats:
        #    stats['lvp'] = midpoint_price
        #    stats['t_vlp'] = t_ms
        return m.bid_price, m.ask_price, delta

    async def handle_msg(self, msgs: List[ForexQuote | CryptoTrade | EquityAgg | EquityTrade]):
        """
        received message: CurrencyAgg(event_type='CAS', pair='USD/CHF', open=0.91313, close=0.91317, high=0.91318,
        low=0.91313, volume=3, vwap=None, start_timestamp=1713273701000, end_timestamp=1713273702000,
         avg_trade_size=None) <class 'polygon.websocket.models.models.CurrencyAgg'>

         CurrencyAgg(event_type='XAS', pair='ETH-USD', open=3084.37, close=3084.24, high=3084.37, low=3084.08,
         volume=0.99917426, vwap=3084.1452, start_timestamp=1713273981000, end_timestamp=1713273982000, avg_trade_size=0)

         CryptoTrade(event_type='XT', pair='SOL-USD', exchange=23, id='98a5b760-1884-475f-81e8-c215b74cc641',
          price=236.51, size=0.02152625, conditions=[2], timestamp=1732107788615, received_timestamp=1732107788983),

        """
        def msg_to_price_sources(m, tp):
            symbol = tp.trade_pair
            bid = 0
            ask = 0
            if tp.is_forex:
                bid, ask, _ = self.parse_price_for_forex(m, is_ws=True)
                if bid is None:
                    return None, None
                start_timestamp = m.timestamp
                #print(f'Received forex message {symbol} price {new_price} time {TimeUtil.millis_to_formatted_date_str(start_timestamp)}')
                end_timestamp = start_timestamp + 999
                if symbol in self.trade_pair_to_recent_events and self.trade_pair_to_recent_events[symbol].timestamp_exists(start_timestamp):
                    buffer = self.trade_pair_to_recent_events_realtime if self.using_ipc else self.trade_pair_to_recent_events
                    buffer[symbol].update_prices_for_median(start_timestamp, bid, ask)
                    buffer[symbol].update_prices_for_median(start_timestamp + 999, bid, ask)
                    return None, None
                else:
                    open = close = vwap = high = low = bid

            elif tp.is_equities:
                if m.exchange != self.equities_mapping['nasdaq']:
                    #print(f"Skipping equity trade from exchange {m.exchange} for {tp.trade_pair}")
                    return None, None
                if isinstance(m, EquityTrade) and isinstance(m.conditions, list) and 12 in m.conditions:
                    #print(f"Skipping Polygon websocket trade with afterhours condition for {m}")
                    self.n_equity_events_skipped_afterhours += 1
                    return None, None
                start_timestamp = round(m.timestamp, -3)  # round to nearest second which allows aggresssive filtering via dup logic
                end_timestamp = None
                open = close = vwap = high = low = m.price
            elif tp.is_crypto:
                if m.exchange != self.crypto_mapping['coinbase']:
                    #print(f"Skipping crypto trade from exchange {m.exchange} for {tp.trade_pair}")
                    return None, None
                start_timestamp = round(m.received_timestamp, -3) # round to nearest second which allows aggresssive filtering via dup logic
                end_timestamp = None
                open = close = vwap = high = low = m.price
            else:
                start_timestamp = m.start_timestamp
                end_timestamp = m.end_timestamp - 1   # prioritize a new candle's open over a previous candle's close
                open = m.open
                close = m.close
                vwap = m.vwap
                high = m.high
                low = m.low
                #print(f'Received message {symbol} price {close} time {TimeUtil.millis_to_formatted_date_str(start_timestamp)}')

            now_ms = TimeUtil.now_in_millis()
            price_source1 = PriceSource(
                source=f'{POLYGON_PROVIDER_NAME}_ws',
                timespan_ms=0,
                open=open,
                close=open,
                vwap=vwap,
                high=high,
                low=low,
                start_ms=start_timestamp,
                websocket=True,
                lag_ms=now_ms - start_timestamp,
                bid=bid,
                ask=ask
            )

            if tp.is_equities or tp.is_crypto:
                # This is a point in time trade. We can't make a candle out of it
                price_source2 = None
            else:
                price_source2 = PriceSource(
                    source=f'{POLYGON_PROVIDER_NAME}_ws',
                    timespan_ms=0,
                    open=close,
                    close=close,
                    vwap=vwap,
                    high=high,
                    low=low,
                    start_ms=end_timestamp,
                    websocket=True,
                    lag_ms=now_ms - end_timestamp,
                    bid=bid,
                    ask=ask
                )
            return price_source1, price_source2
        try:
            m = None
            for m in msgs:
                #bt.logging.info(f"Received price event: {m}")
                # print('received message:', m, type(m))
                if isinstance(m, EquityAgg):
                    tp = self.symbol_to_trade_pair(m.symbol[2:])  # I:SPX -> SPX
                elif isinstance(m, CryptoTrade):
                    tp = self.symbol_to_trade_pair(m.pair)
                elif isinstance(m, ForexQuote):
                    tp = self.symbol_to_trade_pair(m.pair)
                elif isinstance(m, EquityTrade):
                    tp = self.symbol_to_trade_pair(m.symbol)
                else:
                    raise ValueError(f"Unknown message in POLY websocket: {m}")

                tpc = tp.trade_pair_category
                self.tpc_to_n_events[tpc] += 1
                self.tpc_to_last_event_time[tpc] = time.time()
                # This could be a candle so we can make 2 prices, one for the open and one for the close
                symbol = tp.trade_pair
                ps1, ps2 = msg_to_price_sources(m, tp)
                if ps1 is None and ps2 is None:
                    continue

                # Reset the closed market price, indicating that a new close should be fetched after the current day's close
                self.closed_market_prices[tp] = None
                for ps in [ps1, ps2]:
                    if ps is None:
                        continue
                    self.latest_websocket_events[symbol] = ps
                    if symbol not in self.trade_pair_to_recent_events:
                        if self.using_ipc:
                            self.trade_pair_to_recent_events[symbol] = RecentEventTracker()
                        else:
                            self.trade_pair_to_recent_events_realtime[symbol] = RecentEventTracker()
                    if self.using_ipc:
                        self.trade_pair_to_recent_events_realtime[symbol].add_event(ps, tp.is_forex, f"{self.provider_name}:{tp.trade_pair}")
                    else:
                        self.trade_pair_to_recent_events[symbol].add_event(ps, tp.is_forex, f"{self.provider_name}:{tp.trade_pair}")

                if DEBUG:
                    formatted_time = TimeUtil.millis_to_formatted_date_str(TimeUtil.now_in_millis())
                    self.trade_pair_to_price_history[tp].append((formatted_time, ps1.price))
            if DEBUG:
                print('last message:', m, 'n msgs total:', len(msgs))
                history_size = sum(len(v) for v in self.trade_pair_to_price_history.values())
                bt.logging.info("History Size: " + str(history_size))
                bt.logging.info(f"n_events_global: {sum(self.tpc_to_n_events.values())} breakdown {self.tpc_to_n_events}")
        except Exception as e:
            full_traceback = traceback.format_exc()
            # Slice the last 1000 characters of the traceback
            limited_traceback = full_traceback[-1000:]
            bt.logging.error(f"Failed to handle POLY websocket message with error: {e}, last message {m} "
                             f"type: {type(e).__name__}, traceback: {limited_traceback}")

    def instantiate_not_pickleable_objects(self):
        self.POLYGON_CLIENT = RESTClient(api_key=self._api_key, num_pools=20)

    def _subscribe_websockets(self, tpc: TradePairCategory = None):
        subbed = []
        for tp in TradePair:
            if tp in self.UNSUPPORTED_TRADE_PAIRS:
                continue
            if tpc and tp.trade_pair_category != tpc:
                continue
            if tp.is_crypto:
                symbol = "XT." + tp.trade_pair.replace('/', '-')
                self.WEBSOCKET_OBJECTS[TradePairCategory.CRYPTO].subscribe(symbol)
                subbed.append(symbol)
            elif tp.is_forex:
                symbol = "C." + tp.trade_pair
                self.WEBSOCKET_OBJECTS[TradePairCategory.FOREX].subscribe(symbol)
                subbed.append(symbol)
            elif tp.is_equities:
                symbol = "T." + tp.trade_pair
                subbed.append(symbol)
                self.WEBSOCKET_OBJECTS[TradePairCategory.EQUITIES].subscribe(symbol)
            elif tp.is_indices:
                continue
            else:
                raise ValueError(f"Unknown trade pair category: {tp.trade_pair_category}")
        bt.logging.info(f"{self.provider_name} subscribed to {len(subbed)} symbols in {tpc} category: {subbed}")

    def symbol_to_trade_pair(self, symbol: str):
        # Should work for indices and forex
        tp = TradePair.get_latest_tade_pair_from_trade_pair_str(symbol)
        if tp:
            return tp
        # Should work for crypto. Anything else will return None
        tp = TradePair.get_latest_tade_pair_from_trade_pair_str(symbol.replace('-', '/'))
        if not tp:
            raise ValueError(f"Unknown symbol: {symbol}")
        return tp

    def get_closes_rest(self, pairs: List[TradePair]) -> dict:
        all_trade_pair_closes = {}
        # Multi-threaded fetching of REST data over all requested trade pairs. Max parallelism is 5.
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Dictionary to keep track of futures
            future_to_trade_pair = {executor.submit(self.get_close_rest, p): p for p in pairs}

            for future in as_completed(future_to_trade_pair):
                tp = future_to_trade_pair[future]
                try:
                    result = future.result()
                    if result is None:
                        result = {}
                    all_trade_pair_closes[tp] = result
                except Exception as exc:
                    bt.logging.error(f"{tp} generated an exception: {exc}. Continuing...")
                    bt.logging.error(traceback.format_exc())

        return all_trade_pair_closes

    def agg_to_price_source(self, a, now_ms:int, timespan:str, attempting_prev_close:bool=False):
        p_name = f'{POLYGON_PROVIDER_NAME}_rest'
        if attempting_prev_close:
            p_name += '_prev_close'

        return \
            PriceSource(
                source=p_name,
                timespan_ms=self.timespan_to_ms[timespan],
                open=a.open,
                close=a.close,
                vwap=a.vwap,
                high=a.high,
                low=a.low,
                start_ms=a.timestamp,
                websocket=False,
                lag_ms=now_ms - a.timestamp,
                bid=a.bid if hasattr(a, 'bid') else 0,
                ask=a.ask if hasattr(a, 'ask') else 0
            )

    def _create_websocket_client(self, tpc: TradePairCategory):
        feed = Feed.RealTime
        # instantiate new client
        if tpc == TradePairCategory.CRYPTO:
            market = Market.Crypto
        elif tpc == TradePairCategory.FOREX:
            market = Market.Forex
        elif tpc == TradePairCategory.EQUITIES:
            market = Market.Stocks
        else:
            raise Exception(f'Unexpected tpc {tpc}')

        # Depending on API key, the feed may be different for equities
        if tpc == TradePairCategory.EQUITIES:
            feed = self.stocks_feed_round_robin_map[self.stocks_feed_round_robin_counter]
            self.stocks_feed_round_robin_counter = (1 + self.stocks_feed_round_robin_counter) % len(self.stocks_feed_round_robin_map)
        client = WebSocketClient(market=market, api_key=self._api_key, feed=feed)
        bt.logging.info(f"Created {self.provider_name} websocket for {tpc}. feed {feed.name}")
        self.WEBSOCKET_OBJECTS[tpc] = client

    def get_close_rest(
        self,
        trade_pair: TradePair,
        timestamp_ms: int = None,
        order: Order = None
    ) -> PriceSource | None:
        if not timestamp_ms:
            timestamp_ms = TimeUtil.now_in_millis()
        if self.is_backtesting:
            # Check that we are within market hours for genuine ptn orders
            if order is not None and order.src == 0:
                assert self.is_market_open(trade_pair, time_ms=timestamp_ms)

        if not self.is_market_open(trade_pair, time_ms=timestamp_ms):
            return self.get_event_before_market_close(trade_pair, timestamp_ms)

        prev_timestamp = None
        final_agg = None
        timespan = "second"
        raw = self.unified_candle_fetcher(trade_pair, timestamp_ms - 10000, timestamp_ms + 2000, timespan)
        for a in raw:
            #print('agg:', a)
            """
                    agg Agg(open=111.91, high=111.91, low=111.902, close=111.909, volume=3, vwap=111.907,
                    timestamp=1713273876000, transactions=3, otc=None)
            """
            epoch_miliseconds = a.timestamp
            price_source = self.agg_to_price_source(a, timestamp_ms, timespan)
            assert prev_timestamp is None or prev_timestamp < epoch_miliseconds
            #formatted_date = TimeUtil.millis_to_formatted_date_str(epoch_miliseconds // 1000)
            final_agg = price_source
            prev_timestamp = epoch_miliseconds
        if not final_agg:
            bt.logging.warning(f"Polygon failed to fetch REST data for {trade_pair.trade_pair} at time "
                               f"{TimeUtil.millis_to_formatted_date_str(timestamp_ms)}. "
                               f"If you keep seeing this warning, report it to the team ASAP")
            final_agg = None

        return final_agg


    def trade_pair_to_polygon_ticker(self, trade_pair: TradePair):
        if trade_pair.is_crypto:
            return 'X:' + trade_pair.trade_pair_id
        elif trade_pair.is_forex:
            return 'C:' + trade_pair.trade_pair_id
        elif trade_pair.is_indices:
            return 'I:' + trade_pair.trade_pair_id
        elif trade_pair.is_equities:
            return trade_pair.trade_pair_id
        else:
            raise ValueError(f"Unknown trade pair category: {trade_pair.trade_pair_category}")

    def get_event_before_market_close(self, trade_pair: TradePair, target_time_ms:int) -> PriceSource | None:
        # The caller made sure the market is closed.
        if trade_pair in self.UNSUPPORTED_TRADE_PAIRS:
            return None

        if self.closed_market_prices[trade_pair] is not None:
            delta_time = target_time_ms - self.closed_market_prices_timestamp_ms[trade_pair]
            # If this closed market price happened within the last 24 hours, reuse it
            if delta_time >= 0 and delta_time < 1000 * 60 * 60 * 24:
                return self.closed_market_prices[trade_pair]

        # start 7 days ago
        start_time_ms = target_time_ms - 1000 * 60 * 60 * 24 * 7
        candles = self.get_candles_for_trade_pair(trade_pair, start_time_ms, target_time_ms, target_time_ms, attempting_prev_close=True, force_timespan='day')
        if len(candles) == 0:
            msg = f"get_event_before_market_close: Failed to fetch market close for {trade_pair.trade_pair}"
            raise ValueError(msg)

        ans = candles[-1]
        self.closed_market_prices[trade_pair] = ans
        self.closed_market_prices_timestamp_ms[trade_pair] = target_time_ms
        return ans


    def get_websocket_event(self, trade_pair: TradePair) -> PriceSource | None:
        symbol = trade_pair.trade_pair
        cur_event = self.latest_websocket_events.get(symbol)
        if not cur_event:
            return None

        timestamp_ms = cur_event.end_ms
        max_allowed_lag_s = self.trade_pair_category_to_longest_allowed_lag_s[trade_pair.trade_pair_category]
        lag_s = time.time() - timestamp_ms / 1000.0
        is_stale = lag_s > max_allowed_lag_s
        if is_stale:
            bt.logging.info(f"Found stale TD websocket data for {trade_pair.trade_pair}. Lag_s: {lag_s} "
                            f"seconds. Max allowed lag for category: {max_allowed_lag_s} seconds. Ignoring this data.")
        return cur_event


    def get_close_in_past_hour_fallback(self, trade_pair: TradePair, timestamp_ms: int):
        polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)  # noqa: F841

        prev_timestamp = None
        smallest_delta = None
        corresponding_price = None
        start_time = None
        n_responses = 0
        candle = None
        timespan = "hour"

        def try_updating_found_price(t, p):
            nonlocal smallest_delta, corresponding_price, start_time, candle
            time_delta_ms = abs(t - timestamp_ms)
            if smallest_delta is None or time_delta_ms < smallest_delta:
                smallest_delta = time_delta_ms
                corresponding_price = p
                start_time = epoch_miliseconds
                candle = a

        raw = self.unified_candle_fetcher(trade_pair, timestamp_ms - 1000 * 60 * 60 * 48, timestamp_ms + 1000 * 60 * 30, timespan)
        for a in raw:
            n_responses += 1
            epoch_miliseconds = a.timestamp

            try_updating_found_price(epoch_miliseconds, a.open)
            try_updating_found_price(epoch_miliseconds + self.timespan_to_ms[timespan], a.close)

            assert prev_timestamp is None or prev_timestamp < epoch_miliseconds
            prev_timestamp = epoch_miliseconds

        #print(f"hourly fallback smallest delta s: {smallest_delta / 1000 if smallest_delta else None}, input_timestamp: {timestamp_ms}, candle_start_time_ms: {start_time}, candle: {candle}, n_responses: {n_responses}")
        return corresponding_price



    def get_close_at_date_minute_fallback(self, trade_pair: TradePair, target_timestamp_ms: int) -> PriceSource | None:
        polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)  # noqa: F841

        prev_timestamp = None
        smallest_delta = None
        corresponding_price_source = None
        n_responses = 0
        timespan = "minute"

        def try_updating_found_price(t_ms, agg):
            nonlocal smallest_delta, corresponding_price_source
            time_delta_ms = abs(t_ms - target_timestamp_ms)
            if smallest_delta is None or time_delta_ms <= smallest_delta:
                smallest_delta = time_delta_ms
                corresponding_price_source = self.agg_to_price_source(agg, target_timestamp_ms, timespan)

        raw = self.unified_candle_fetcher(trade_pair,
                                          target_timestamp_ms - 1000 * 60 * 2,
                                          target_timestamp_ms + 1000 * 60 * 2, timespan)
        for a in raw:
            n_responses += 1
            epoch_miliseconds = a.timestamp

            try_updating_found_price(epoch_miliseconds, a)
            try_updating_found_price(epoch_miliseconds + self.timespan_to_ms[timespan], a)

            assert prev_timestamp is None or prev_timestamp < epoch_miliseconds
            prev_timestamp = epoch_miliseconds

        #print(f"minute fallback smallest delta ms: {smallest_delta}, input_timestamp: {timestamp_ms}, candle_start_time_ms: {start_time}, candle: {candle}, n_responses: {n_responses}")
        return corresponding_price_source

    def get_close_at_date_second(self, trade_pair: TradePair, target_timestamp_ms: int, order: Order = None) -> PriceSource | None:
        prev_timestamp = None
        smallest_delta = None
        corresponding_price_source = None
        n_responses = 0
        timespan = "second"
        def try_updating_found_price(t, agg):
            nonlocal smallest_delta, corresponding_price_source, target_timestamp_ms
            time_delta_ms = abs(t - target_timestamp_ms)
            if smallest_delta is None or time_delta_ms <= smallest_delta:
                #print('Updated best answer', time_delta_ms, smallest_delta, t, p)
                smallest_delta = time_delta_ms
                corresponding_price_source = self.agg_to_price_source(agg, target_timestamp_ms, timespan)

        raw = self.unified_candle_fetcher(trade_pair, target_timestamp_ms - 1000 * 59, target_timestamp_ms + 1000 * 59, timespan)
        for a in raw:
            #print('agg', a, 'dt', target_timestamp_ms - a.timestamp, 'ms')
            n_responses += 1
            try_updating_found_price(a.timestamp, a)

            assert prev_timestamp is None or prev_timestamp < a.timestamp, raw
            prev_timestamp = a.timestamp

        #print(f"smallest delta ms: {smallest_delta}, input_timestamp: {timestamp_ms}, candle_start_time_ms: {start_time}, candle: {candle}, n_responses: {n_responses}")
        return corresponding_price_source


    def get_candles(self, trade_pairs: List[TradePair], start_time_ms:int, end_time_ms:int):
        # Dictionary to store the minimum prices for each trade pair
        ret = {}

        # Create a ThreadPoolExecutor with a maximum of 5 threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Future objects dictionary to hold the ongoing computations
            futures = {executor.submit(self.get_candles_for_trade_pair, tp, start_time_ms, end_time_ms): tp for tp in trade_pairs}

            # Retrieve the results as they complete
            for future in as_completed(futures):
                trade_pair = futures[future]
                try:
                    # Collect the result from future
                    result = future.result()
                    ret[trade_pair] = result
                except Exception as exc:
                    print(f'{trade_pair} get_candles_for_trade_pair generated an exception: {exc}')

        # Return the collected results
        return ret

    def unified_candle_fetcher(self, trade_pair: TradePair, start_timestamp_ms: int, end_timestamp_ms: int, timespan: str=None):

        def _fetch_raw_polygon_aggs():
            return self.POLYGON_CLIENT.list_aggs(
                polygon_ticker,
                1,
                timespan,
                start_timestamp_ms,
                end_timestamp_ms,
                limit=self.N_CANDLES_LIMIT
            )
        def _intra_vwap_valid(agg):
            return abs(agg.vwap - agg.close) / agg.close < .004

        def _consecutive_candle_spiked(a, b):
            return abs(b.close - a.close) / a.close > .015

        def _agg_to_payload(agg, prev, nxt, last_valid_price, waiting_for_valid_payload, tp_id, verbose=False):
            if waiting_for_valid_payload > 20 and prev and nxt:  # waited long enough
                return _agg_to_payload(agg, prev, nxt, None, 0, tp_id)
            elif waiting_for_valid_payload > 0:
                delta = abs(agg.close - last_valid_price) / last_valid_price
                vv = _intra_vwap_valid(agg)
                price_valid = delta < .005 and vv  # close enough to the tether
                if verbose:
                    if price_valid:
                        print(f'Breaking out. tp_id {tp_id}, waiting_for_valid_payload {waiting_for_valid_payload}, delta {delta} last_valid_price {last_valid_price} agg {agg}')
                    else:
                        print(f'tp_id {tp_id}, waiting_for_valid_payload {waiting_for_valid_payload}, delta {delta} vv {vv} last_valid_price {last_valid_price} agg.close {agg.close} agg.vwap {agg.vwap}')
                return agg, price_valid

            # forex candles are subject to spikes. particularly at the end of day.
            if prev and nxt is None and _consecutive_candle_spiked(prev, agg):
                agg.close = prev.vwap
                return agg, True  # Don't enter the waiting_for_valid_payload state machine as this is likely a one-off
            elif nxt and prev is None and _consecutive_candle_spiked(agg, nxt):
                agg.close = nxt.vwap
                return agg, True  # Don't enter the waiting_for_valid_payload state machine as this is likely a one-off
            elif _intra_vwap_valid(agg):
                return agg, True
            else:  # spike detected
                vv = agg.low <= agg.vwap <= agg.high
                smoothed_price = agg.vwap if vv else (agg.high + agg.low) / 2
                agg.close = smoothed_price
                if verbose:
                    print('--------------------------------------')
                    print(f'Rejecting {tp_id}. delta {abs(agg.vwap - agg.close) / agg.close}. vv {vv} last_valid_price {last_valid_price} agg {agg}')
                return agg, False

        def _get_filtered_forex_minute_data():
            price_info_raw = list(_fetch_raw_polygon_aggs())
            n_points = len(price_info_raw)
            last_valid_price = None
            tp_id = trade_pair.trade_pair_id
            waiting_for_valid_payload = 0  # minutes we've been ignoring data
            ans = []
            self.tp_to_mfs = {}
            for i in range(n_points):
                a = price_info_raw[i]

                prev = price_info_raw[i - 1] if i > 0 else None
                nxt = price_info_raw[i + 1] if i < n_points - 1 else None

                agg, is_valid = _agg_to_payload(a, prev, nxt, last_valid_price, waiting_for_valid_payload, tp_id)
                if is_valid:
                    last_valid_price = agg.close
                    waiting_for_valid_payload = 0
                    ans.append(agg)
                if not is_valid:
                    if last_valid_price:
                        waiting_for_valid_payload += 1
                    else:
                        ans.append(agg)  # smoothed price. Don't enter state machine as we have no valid last price to tether to
                    # debug/metrics
                    if tp_id not in self.tp_to_mfs:
                        self.tp_to_mfs[tp_id] = waiting_for_valid_payload
                    else:
                        self.tp_to_mfs[tp_id] = max(self.tp_to_mfs[tp_id], waiting_for_valid_payload)
            return ans


        def _get_filtered_forex_second_data():
            ans = []
            prev_t_ms = None
            raw = self.POLYGON_CLIENT.list_quotes(ticker=polygon_ticker,
                                                                   timestamp_gte=start_timestamp_ms * 1000000,
                                                                   timestamp_lte=end_timestamp_ms * 1000000,
                                                                   sort='participant_timestamp',
                                                                   order='asc',
                                                                   limit=self.N_CANDLES_LIMIT)
            n_quotes = 0
            best_delta = float('inf')
            for r in raw:
                t_ms = r.participant_timestamp // 1000000
                if t_ms != prev_t_ms:
                    best_delta = float('inf')
                    if ans and hasattr(ans[-1], 'temp'):
                        del ans[-1].temp
                n_quotes += 1
                bid, ask, current_delta = self.parse_price_for_forex(r, stats=None)
                if bid is None:
                    continue
                midpoint_price = (bid + ask) / 2.0

                if best_delta == float('inf'):
                    best_delta = current_delta
                    ans.append(Agg(open=midpoint_price,
                                   close=midpoint_price,
                                   high=midpoint_price,
                                   low=midpoint_price,
                                   vwap=None,
                                   timestamp=t_ms,
                                   bid=bid,
                                   ask=ask,
                                   volume=0))
                    ans[-1].temp = ([bid], [ask])
                else:
                    best_delta = current_delta
                    dat = ans[-1].temp
                    dat[0].append(bid)
                    dat[0].sort()
                    dat[1].append(ask)
                    dat[1].sort()

                    # Get the median value if the length is odd. Otherwise, average the two middle values
                    median_bid = RecentEventTracker.forex_median_price(dat[0])
                    median_ask = RecentEventTracker.forex_median_price(dat[1])
                    midpoint_price = (median_bid + median_ask) / 2.0
                    ans[-1].open = ans[-1].close = ans[-1].low = ans[-1].high = midpoint_price
                    ans[-1].bid = median_bid
                    ans[-1].ask = median_ask

                prev_t_ms = t_ms

            return ans, n_quotes

        if self.POLYGON_CLIENT is None:
            self.instantiate_not_pickleable_objects()

        polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)
        if trade_pair.is_forex:
            if timespan == 'second':
                ans, n = _get_filtered_forex_second_data()
            elif timespan == 'minute':
                ans = _get_filtered_forex_minute_data()
            elif timespan == 'day':
                return list(_fetch_raw_polygon_aggs())
            else:
                raise Exception(f'Invalid timespan {timespan}')
            return ans
        else:
            return list(_fetch_raw_polygon_aggs())

    def get_candles_for_trade_pair(
        self,
        trade_pair: TradePair,
        start_timestamp_ms: int,
        end_timestamp_ms: int,
        target_timestamp_ms: int,
        attempting_prev_close: bool = False,
        force_timespan: str = None
    ) -> list[PriceSource] | None:
        """
        agg Agg(open=111.91, high=111.91, low=111.902, close=111.909, volume=3, vwap=111.907,
        timestamp=1713273876000, transactions=3, otc=None)

        agg Agg(open=63010.86, high=63010.86, low=63010.86, close=63010.86, volume=2.158e-05, vwap=63010.86,
         timestamp=1713273888000, transactions=1, otc=None)
        """

        delta_time_ms = end_timestamp_ms - start_timestamp_ms
        delta_time_seconds = delta_time_ms / 1000
        delta_time_minutes = delta_time_seconds / 60
        delta_time_hours = delta_time_minutes / 60

        if delta_time_seconds < 70:
            timespan = "second"
        elif delta_time_minutes < 70:
            timespan = "minute"
        elif delta_time_hours < 70:
            timespan = "hour"
        else:
            timespan = "day"

        if force_timespan:
            timespan = force_timespan

        aggs = []
        prev_timestamp = None
        raw = self.unified_candle_fetcher(trade_pair, start_timestamp_ms, end_timestamp_ms, timespan)
        for i, a in enumerate(raw):
            epoch_miliseconds = a.timestamp
            assert prev_timestamp is None or epoch_miliseconds >= prev_timestamp, ('candles not sorted', prev_timestamp, epoch_miliseconds)
            #formatted_date = TimeUtil.millis_to_formatted_date_str(epoch_miliseconds)
            #if i != -1:
                #print('        agg:', a.low, formatted_date, trade_pair.trade_pair_id, timespan)
            price_source = self.agg_to_price_source(a, target_timestamp_ms, timespan, attempting_prev_close=attempting_prev_close)
            #print(formatted_date, price_source)
            aggs.append(price_source)
            prev_timestamp = epoch_miliseconds

        if not aggs:
            bt.logging.trace(f"{POLYGON_PROVIDER_NAME} failed to fetch candle data for {trade_pair.trade_pair}. "
                             f" Perhaps this trade pair was closed during the specified window.")

        return aggs

    def get_quote(self, trade_pair: TradePair, processed_ms: int) -> (float, float, int):
        """
        returns the bid and ask quote for a trade_pair at processed_ms
        """
        # polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)

        if self.POLYGON_CLIENT is None:
            self.instantiate_not_pickleable_objects()

        if trade_pair.is_forex or trade_pair.is_equities:
            polygon_ticker = self.trade_pair_to_polygon_ticker(trade_pair)
            quotes = self.POLYGON_CLIENT.list_quotes(
                ticker=polygon_ticker,
                timestamp_lte=processed_ms * 1_000_000,
                sort="participant_timestamp",
                order="desc",
                limit=1
            )
            for q in quotes:
                return q.bid_price, q.ask_price, int(q.participant_timestamp/1_000_000)  # convert ns back to ms
        else:
            # crypto
            return 0, 0, 0

    def get_currency_conversion(self, trade_pair: TradePair=None, base: str=None, quote: str=None) -> float:
        """
        get the currency conversion rate from base currency to quote currency
        """
        if self.POLYGON_CLIENT is None:
            self.instantiate_not_pickleable_objects()

        if not (base and quote):
            if trade_pair and trade_pair.is_forex:
                base, quote = trade_pair.trade_pair.split("/")
            else:
                raise ValueError("Must provide either a valid forex pair or a base and quote for currency conversion")

        rate = self.POLYGON_CLIENT.get_real_time_currency_conversion(
            from_=base,
            to=quote,
            precision=4,
        )

        return rate.converted


if __name__ == "__main__":

    secrets = ValiUtils.get_secrets()

    polygon_data_provider = PolygonDataService(api_key=secrets['polygon_apikey'], disable_ws=True)
    ans = polygon_data_provider.get_close_rest(TradePair.TAOUSD, 1760753263000)
    print('@@', ans)
    time.sleep(10000)
    assert 0, ans
    for tp in TradePair:
        if tp.is_indices:
            continue
        #if tp != TradePair.GBPUSD:
        #    continue

        print('getting close for', tp.trade_pair_id, ':', polygon_data_provider.get_close_rest(tp))

    time.sleep(100000)

    polygon_data_provider = PolygonDataService(api_key=secrets['polygon_apikey'], disable_ws=True)
    target_timestamp_ms = 1735088280163#1715288494000

    """
    aggs = []
    for a in RESTClient(secrets['polygon_apikey']).list_aggs(
            "X:BNBUSD",
            1,
            "day",
            "2023-01-30",
            "2023-02-03",
            limit=50000,
    ):
        aggs.append(a)

    assert 0, aggs
    """