import asyncio
import threading
import traceback
import json
from datetime import timedelta
from multiprocessing import Process

import requests
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_generator.base_data_service import BaseDataService, TIINGO_PROVIDER_NAME, exception_handler_decorator
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory

import time
import websockets

from vali_objects.utils.vali_utils import ValiUtils
import bittensor as bt

from vali_objects.vali_dataclasses.price_source import PriceSource

from tiingo import TiingoClient#, TiingoWebsocketClient

from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker
from shared_objects.log import logger

DEBUG = 0
TIINGO_COINBASE_EXCHANGE_STR = 'gdax'
TIINGO_MAX_TICKERS_PER_REQUEST = 5

class _TiingoPseudoClient:
    def __init__(self, service, category):
        self._svc: TiingoDataService = service
        self._cat: TradePairCategory = category
        self._should_close = False

    async def connect(self, handle_msg):
        """
        Asynchronous connect method for the Tiingo websocket client
        """
        POLLING_INTERVAL_S = 8.6
        last_poll_time = 0

        while not self._should_close:
            current_time = time.time()
            elapsed_time = current_time - last_poll_time

            if elapsed_time < POLLING_INTERVAL_S:
                # Use await sleep to yield control back to the event loop
                await asyncio.sleep(1)
                continue

            try:
                # Get tradeable pairs to query (excluding blocked pairs)
                trade_pairs_to_query = self._svc.get_tradeable_pairs(
                    category=self._cat,
                    include_blocked=False,  # Don't subscribe to blocked pairs
                    market_open_only=True  # Only query pairs with open markets
                )

                if not trade_pairs_to_query:
                    await asyncio.sleep(1)
                    continue

                last_poll_time = current_time

                # Get price data synchronously but don't block the event loop
                loop = asyncio.get_event_loop()
                price_sources = await loop.run_in_executor(
                    None, self._svc.get_price_rest, trade_pairs_to_query, int(current_time * 1000), True
                )

                # Process each price source
                for trade_pair, price_source in price_sources.items():
                    if price_source:
                        price_source.websocket = True
                        # Process the price source directly
                        self._svc.process_ps_from_websocket(trade_pair, price_source)
                        # Update event counter
                        self._svc.tpc_to_n_events[trade_pair.trade_pair_category] += 1

            except Exception as e:
                logger.error(f"Error in {self._svc.provider_name} pseudo websocket for {self._cat}: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)


    def unsubscribe_all(self):
        """Signal that the client should stop polling."""
        self._should_close = True
        logger.info(f"Unsubscribed {self._svc.provider_name} websocket client for {self._cat}")

    async def close(self):
        """Close the client."""
        self._should_close = True
        logger.info(f"Closed {self._svc.provider_name} websocket client for {self._cat}")


class _TiingoWebsocketClient:
    def __init__(self, service, category, api_key):
        self._svc: TiingoDataService = service
        self._cat: TradePairCategory = category
        self._api_key = api_key
        self._ws = None
        self._should_close = False

    async def connect(self, handle_msg):
        """Connect to Tiingo websocket and process messages"""
        # Determine endpoint based on category
        if self._cat == TradePairCategory.EQUITIES:
            url = "wss://api.tiingo.com/iex"
            threshold_level = 6
        elif self._cat == TradePairCategory.FOREX:
            url = "wss://api.tiingo.com/fx"
            threshold_level = 5
        elif self._cat == TradePairCategory.CRYPTO:
            url = "wss://api.tiingo.com/crypto"
            threshold_level = 5
        else:
            raise ValueError(f"Unknown category: {self._cat}")

        # Connect using websockets library (async-native)
        # Don't use async with here - we want to keep connection open
        self._ws = await websockets.connect(url)

        try:
            # Build subscribe message
            trade_pairs_to_query = self._svc.get_tradeable_pairs(
                category=self._cat,
                include_blocked=False,
                market_open_only=False  # Subscribe to all pairs, filter on receipt
            )

            tickers = [tp.trade_pair_id.lower() for tp in trade_pairs_to_query]

            subscribe = {
                'eventName': 'subscribe',
                'authorization': self._api_key,
                'eventData': {
                    'thresholdLevel': threshold_level,
                    'tickers': tickers if tickers else ['*']
                }
            }

            logger.info(f"Subscribing to {self._cat} with tickers: {tickers}")

            # Send subscription (async send)
            await self._ws.send(json.dumps(subscribe))
            logger.info(f"Subscribed to {len(tickers)} {self._cat} tickers")

            # Receive loop - keep running until closed or error
            while not self._should_close:
                try:
                    msg = await self._ws.recv()
                    await handle_msg(msg)
                except websockets.exceptions.ConnectionClosed as e:
                    if e.rcvd is not None:
                        logger.warning(f"Tiingo {self._cat} websocket closed: code={e.rcvd.code}, reason={e.rcvd.reason}")
                    else:
                        logger.warning(f"Tiingo {self._cat} websocket closed abnormally (no close frame received)")
                    break
                except Exception as e:
                    if self._should_close:
                        break
                    logger.error(f"Error receiving/handling message for {self._cat}: {e}")
                    logger.error(f"Exception type: {type(e).__name__}")
                    logger.error(traceback.format_exc())
                    # Don't raise - continue receiving
                    continue
        finally:
            # Ensure cleanup happens
            if self._ws:
                await self._ws.close()

    async def close(self):
        """Close websocket connection"""
        self._should_close = True
        if self._ws:
            await self._ws.close()

    def unsubscribe_all(self):
        """Signal that client should close"""
        self._should_close = True


class TiingoDataService(BaseDataService):

    def __init__(self, api_key, disable_ws=False, running_unit_tests=False):
        self.init_time = time.time()
        self._api_key = api_key
        self.disable_ws = disable_ws
        self.running_unit_tests = running_unit_tests

        super().__init__(
            provider_name=TIINGO_PROVIDER_NAME,
            running_unit_tests=running_unit_tests,
            enabled_websocket_categories={TradePairCategory.CRYPTO, TradePairCategory.FOREX}
        )

        self.MARKET_STATUS = None

        self.config = {'api_key': self._api_key, 'session': True}
        self.TIINGO_CLIENT = None  # Instantiate the TiingoClient after process starts

        self.subscribe_message = {
            'eventName': 'subscribe',
            'authorization': self.config['api_key'],
            #see https://api.tiingo.com/documentation/websockets/iex > Request for more info
            'eventData': {
                'thresholdLevel':5
            }
        }

        # Start thread to refresh market status
        if disable_ws:
            self.websocket_manager_thread = None
        else:
            self.websocket_manager_thread = threading.Thread(target=self.websocket_manager, daemon=True)
            self.websocket_manager_thread.start()

    def _close_ws_for_category(self, tpc: TradePairCategory, loop):
        """
        “Close” the Tiingo pseudo‐websocket for the given category by
        forcing the runner to tear down and rebuild on the next loop.
        """
        client = self.WEBSOCKET_OBJECTS.get(tpc)
        if not client:
            return

        # 1) trigger the forced‐exception in run_pseudo_websocket()
        try:
            client.unsubscribe_all()
            logger.info(f"Unsubscribed {self.provider_name} websocket for {tpc.name.lower()}")
        except Exception:
            logger.warning(
                f"Failed to unsubscribe old {self.provider_name} websocket for {tpc.name.lower()}",
                exc_info=True
            )

        # 2) clear out the client so the runner loop will recreate it
        self.WEBSOCKET_OBJECTS[tpc] = None

    def _create_websocket_client(self, tpc):
        if self.TIINGO_CLIENT is None:
            self.instantiate_not_pickleable_objects()

        client = _TiingoWebsocketClient(self, tpc, self._api_key)
        # client = _TiingoPseudoClient(self, tpc)
        logger.info(f"Created {self.provider_name} pseudo-websocket (REST polling) for {tpc}")
        self.WEBSOCKET_OBJECTS[tpc] = client

    def _subscribe_websockets(self, tpc):
        """
        For Tiingo we never actually tear down the polling thread, so this
        can just log and move on.
        """
        #logger.info(f"Subscribing {self.provider_name} for {tpc} is a noop.")
        pass

    def instantiate_not_pickleable_objects(self):
        self.TIINGO_CLIENT = TiingoClient(self.config)

    @staticmethod
    def _batch_trade_pairs(trade_pairs: List[TradePair], batch_size: int = TIINGO_MAX_TICKERS_PER_REQUEST) -> List[List[TradePair]]:
        """
        Batches trade pairs into groups to respect Tiingo's API limit of 5 tickers per request.

        Args:
            trade_pairs: List of TradePair objects to batch
            batch_size: Maximum number of tickers per batch (default: 5)

        Returns:
            List of batches, where each batch is a list of TradePair objects
        """
        batches = []
        for i in range(0, len(trade_pairs), batch_size):
            batches.append(trade_pairs[i:i + batch_size])
        return batches

    async def handle_msg(self, msg):
        """
        IEX/Equities: {'messageType': 'A', 'service': 'iex', 'data': ['2019-01-30T13:33:45.383129126-05:00', 'vym', 81.585]}
        Forex: {'messageType': 'A', 'service': 'fx', 'data': ['Q', 'eurusd', '2019-01-30T18:33:45.000Z', 100000, 1.14555, 1.14560, 1.14565, 100000]}
        Crypto: {'messageType': 'A', 'service': 'crypto_data', 'data': ['T', 'btcusd', '2019-01-30T18:33:45.000Z', 'gdax', 0.5, 3400.50]}
        """
        def msg_to_price_sources(m:dict, tp:TradePair) -> PriceSource | None:
            symbol = tp.trade_pair
            data = m['data']
            bid_price = 0
            ask_price = 0
            if tp.is_forex:
                assert len(data) == 8, data
                mode, ticker, date_str, bid_size, bid_price, mid_price, ask_size, ask_price = data
                start_timestamp_orig = TimeUtil.parse_iso_to_ms(date_str)
                start_timestamp = round(start_timestamp_orig, -3)  # round to nearest second which allows aggresssive filtering via dup logic
                if symbol in self.trade_pair_to_recent_events and self.trade_pair_to_recent_events[symbol].timestamp_exists(start_timestamp):
                    self.trade_pair_to_recent_events[symbol].update_prices_for_median(start_timestamp, bid_price, ask_price)
                    return None

                open = vwap = high = low = bid_price
            elif tp.is_crypto:
                mode, ticker, date_str, exchange, volume, price = data
                start_timestamp = TimeUtil.parse_iso_to_ms(date_str)
                start_timestamp = round(start_timestamp, -3)  # round to nearest second which allows aggresssive filtering via dup logic
                if mode != 'T':
                    print(f'Skipping crypto due to non-T mode {m}')
                    return None
                open = vwap = high = low = price

            elif tp.is_equities:
                date_str, ticker, price = data
                start_timestamp = TimeUtil.parse_iso_to_ms(date_str)
                start_timestamp = round(start_timestamp, -3)  # round to nearest second
                open = vwap = high = low = price

            elif tp.is_indices:
                raise Exception(f'TODO! {msg}')
            else:
                raise Exception(f'Unknown trade pair category {msg}')

            now_ms = TimeUtil.now_in_millis()
            price_source1 = PriceSource(
                source=f'{TIINGO_PROVIDER_NAME}_ws',
                timespan_ms=0,
                open=open,
                close=open,
                vwap=vwap,
                high=high,
                low=low,
                start_ms=start_timestamp,
                websocket=True,
                lag_ms=now_ms - start_timestamp,
                bid=bid_price,
                ask=ask_price
            )

            return price_source1

        try:
            msg = json.loads(msg)
            if not isinstance(msg, dict):
                # print(f'Non-dict message: {msg}')
                raise ValueError(f'Non-dict message: {msg}')
            if msg['messageType'] != 'A':
                # print(f'Non-A message type: {msg}')
                return
            if msg['service'] == 'fx':
                raw_tp = msg['data'][1]
                tp0, tp1 = raw_tp[0:3].upper(), raw_tp[3:].upper()
                ptn_trade_pair_id = f'{tp0}{tp1}'
                tp = TradePair.from_trade_pair_id(ptn_trade_pair_id)
            elif msg['service'] == 'iex':
                ptn_trade_pair_id = msg['data'][1].upper()
                tp = TradePair.from_trade_pair_id(ptn_trade_pair_id)
                #if tp:
                #    print(msg)
            elif msg['service'] == 'crypto_data':
                ptn_trade_pair_id = msg['data'][1].upper()
                tiingo_exchange_str = msg['data'][3].lower()
                tp = TradePair.from_trade_pair_id(ptn_trade_pair_id)
                if tp and tiingo_exchange_str == TIINGO_COINBASE_EXCHANGE_STR and tp.is_crypto:  # gbpusd shows up in crypto feed
                    #print(msg)
                    pass
                else:
                    return
            else:
                raise ValueError(f"Unknown service: {msg}")
            if not tp:
                return

            self.tpc_to_n_events[tp.trade_pair_category] += 1
            self.tpc_to_last_event_time[tp.trade_pair_category] = time.time()
            ps1 = msg_to_price_sources(msg, tp)

            self.process_ps_from_websocket(tp, ps1)

        except Exception as e:
            full_traceback = traceback.format_exc()
            # Slice the last 1000 characters of the traceback
            limited_traceback = full_traceback[-1000:]
            logger.error(f"Failed to handle {TIINGO_PROVIDER_NAME} websocket message with error: {e}, "
                             f"type: {type(e).__name__}, traceback: {limited_traceback}")

    def process_ps_from_websocket(self, tp: TradePair, ps1: PriceSource):
        if ps1 is None:
            return

        symbol = tp.trade_pair
        # Reset the closed market price, indicating that a new close should be fetched after the current day's close
        self.closed_market_prices[tp] = None

        self.latest_websocket_events[symbol] = ps1
        if symbol not in self.trade_pair_to_recent_events:
            self.trade_pair_to_recent_events[symbol] = RecentEventTracker()


        self.trade_pair_to_recent_events[symbol].add_event(ps1, tp.is_forex,
                                                           f"{self.provider_name}:{tp.trade_pair}")

        if DEBUG:
            formatted_time = TimeUtil.millis_to_formatted_date_str(TimeUtil.now_in_millis())
            self.trade_pair_to_price_history[tp].append((formatted_time, ps1.close))


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

    def get_price_rest(
        self,
        trade_pairs: List[TradePair],
        timestamp_ms: int,
        live: bool
    ) -> dict[TradePair, PriceSource]:
        """
        Fetch prices via REST.

        Args:
            trade_pairs: Pairs to fetch
            timestamp_ms: Target timestamp (used when live=False, ignored when live=True)
            live: True = current prices (market fills), False = historical (perf ledger)

        Returns:
            Map of trade pair to price source. Missing pairs excluded.
        """
        if live:
            timestamp_ms = TimeUtil.now_in_millis()

        # In unit test mode, return default test price sources instead of making network calls
        if self.running_unit_tests:
            from data_generator.polygon_data_service import PolygonDataService
            return {tp: PolygonDataService.DEFAULT_TESTING_FALLBACK_PRICE_SOURCE for tp in trade_pairs}

        tp_equities = [tp for tp in trade_pairs if tp.trade_pair_category == TradePairCategory.EQUITIES]
        tp_crypto = [tp for tp in trade_pairs if tp.trade_pair_category == TradePairCategory.CRYPTO]
        tp_forex = [tp for tp in trade_pairs if tp.trade_pair_category == TradePairCategory.FOREX]

        jobs = []
        if tp_equities:
            jobs.append((self._get_closes_equities, tp_equities, timestamp_ms, live))
        if tp_crypto:
            jobs.append((self._get_closes_crypto, tp_crypto, timestamp_ms, live))
        if tp_forex:
            jobs.append((self._get_closes_forex, tp_forex, timestamp_ms, live))

        tp_to_price = {}

        if len(jobs) == 0:
            return tp_to_price
        elif len(jobs) == 1:
            func, tp_list, target_time_ms, live_flag = jobs[0]
            return func(tp_list, target_time_ms, live_flag)

        with ThreadPoolExecutor() as executor:
            future_to_category = {
                executor.submit(func, tp_list, timestamp_ms, live_flag): func.__name__
                for func, tp_list, timestamp_ms, live_flag in jobs
            }
            for future in as_completed(future_to_category):
                price_result = future.result()
                if price_result:
                    tp_to_price.update(price_result)

        return tp_to_price

    @exception_handler_decorator()
    def _get_closes_equities(self, trade_pairs: List[TradePair], target_time_ms: int, live: bool) -> dict[TradePair, PriceSource]:
        if not live:
            raise Exception('TODO: historical equities not implemented for Tiingo')
        tp_to_price = {}
        if not trade_pairs:
            return tp_to_price
        assert all(tp.trade_pair_category == TradePairCategory.EQUITIES for tp in trade_pairs), trade_pairs

        if all(not self.is_market_open(tp) for tp in trade_pairs) and all(self.closed_market_prices.get(tp) for tp in trade_pairs):
            return {tp: self.closed_market_prices[tp] for tp in trade_pairs}

        def tickers_to_tiingo_iex_url(tickers: List[str]) -> str:
            return f"https://api.tiingo.com/iex/?tickers={','.join(tickers)}&token={self.config['api_key']}"

        batches = self._batch_trade_pairs(trade_pairs)

        for batch in batches:
            url = tickers_to_tiingo_iex_url([self.trade_pair_to_tiingo_ticker(x) for x in batch])
            requestResponse = requests.get(url, headers={'Content-Type': 'application/json'}, timeout=5)
            if requestResponse.status_code == 429:
                continue
            if requestResponse.status_code == 200:
                time_now_ms = TimeUtil.now_in_millis()
                for x in requestResponse.json():
                    tp = TradePair.get_latest_trade_pair_from_trade_pair_id(x['ticker'].upper())
                    data_time_ms = TimeUtil.parse_iso_to_ms(x['timestamp'])

                    price = float(x['tngoLast'])
                    bid_price = x['bidPrice']
                    ask_price = x['askPrice']
                    p_name = f'{TIINGO_PROVIDER_NAME}_rest'
                    attempting_previous_close = not self.is_market_open(tp)
                    if attempting_previous_close:
                        p_name += '_prev_close'
                    tp_to_price[tp] = PriceSource(
                        source=p_name,
                        timespan_ms=0,
                        open=price,
                        close=price,
                        vwap=price,
                        high=price,
                        low=price,
                        start_ms=data_time_ms,
                        websocket=False,
                        lag_ms=time_now_ms - data_time_ms,
                        bid=float(bid_price) if bid_price else 0,
                        ask=float(ask_price) if ask_price else 0
                    )
                    if attempting_previous_close and tp_to_price[tp]:
                        self.closed_market_prices[tp] = tp_to_price[tp]
            else:
                logger.warning(f"Tiingo equities API request failed with status code {requestResponse.status_code}. "
                                 f"URL: {url[:100]}... Response: {requestResponse.text[:200]}")
                continue

        return tp_to_price

    def target_ms_to_start_end_formatted(self, target_time_ms):
        start_day_formatted = TimeUtil.millis_to_short_date_str(target_time_ms)
        end_day_datetime = TimeUtil.millis_to_datetime(target_time_ms)
        # One day ahead.
        end_day_datetime += timedelta(days=1)
        end_day_formatted = end_day_datetime.strftime("%Y-%m-%d")
        return start_day_formatted, end_day_formatted

    @exception_handler_decorator()
    def _get_closes_forex(self, trade_pairs: List[TradePair], target_time_ms: int, live: bool) -> dict[TradePair, PriceSource]:
        def tickers_to_tiingo_forex_url(tickers: List[str]) -> str:
            if not live:
                start_day_formatted, end_day_formatted = self.target_ms_to_start_end_formatted(target_time_ms)
                return f"https://api.tiingo.com/tiingo/fx/prices?tickers={','.join(tickers)}&startDate={start_day_formatted}&endDate={end_day_formatted}&resampleFreq=1min&token={self.config['api_key']}"
            return f"https://api.tiingo.com/tiingo/fx/top?tickers={','.join(tickers)}&token={self.config['api_key']}"

        tp_to_price = {}
        if not trade_pairs:
            return tp_to_price

        assert all(tp.trade_pair_category == TradePairCategory.FOREX for tp in trade_pairs), trade_pairs
        if all(not self.is_market_open(tp) for tp in trade_pairs) and all(self.closed_market_prices.get(tp) for tp in trade_pairs):
            return {tp: self.closed_market_prices[tp] for tp in trade_pairs}

        batches = self._batch_trade_pairs(trade_pairs)
        time_now_ms = TimeUtil.now_in_millis()

        for batch in batches:
            url = tickers_to_tiingo_forex_url([self.trade_pair_to_tiingo_ticker(x) for x in batch])
            requestResponse = requests.get(url, headers={'Content-Type': 'application/json'}, timeout=5)
            if requestResponse.status_code == 429:
                continue
            if requestResponse.status_code == 200:
                lowest_delta = float('inf')
                for x in requestResponse.json():
                    tp = TradePair.get_latest_trade_pair_from_trade_pair_id(x['ticker'].upper())
                    if not live:
                        attempting_previous_close = not self.is_market_open(tp, time_ms=target_time_ms)
                        data_time_ms = TimeUtil.parse_iso_to_ms(x['date'])
                        delta = abs(data_time_ms - target_time_ms)
                        if delta < lowest_delta:
                            bid = ask = 0
                            p_name = f'{TIINGO_PROVIDER_NAME}_historical'
                            open = float(x['open'])
                            close = float(x['close'])
                            vwap = close
                            high = float(x['high'])
                            low = float(x['low'])
                            lag_ms = target_time_ms - data_time_ms
                            timespan_ms = self.timespan_to_ms['minute']
                            lowest_delta = delta
                        else:
                            continue
                    else:
                        attempting_previous_close = not self.is_market_open(tp, time_ms=time_now_ms)
                        bid_raw = x['bidPrice']
                        ask_raw = x['askPrice']
                        if not bid_raw or not ask_raw:
                            continue
                        bid = float(bid_raw)
                        ask = float(ask_raw)
                        high = ask
                        low = bid
                        mid_price = (bid + ask) / 2.0
                        open = close = vwap = mid_price
                        data_time_ms = TimeUtil.parse_iso_to_ms(x['quoteTimestamp'])
                        timespan_ms = 0
                        lag_ms = time_now_ms - data_time_ms
                        p_name = f'{TIINGO_PROVIDER_NAME}_rest'

                    if attempting_previous_close:
                        p_name += '_prev_close'
                    tp_to_price[tp] = PriceSource(
                        source=p_name,
                        timespan_ms=timespan_ms,
                        open=open,
                        close=close,
                        vwap=vwap,
                        high=high,
                        low=low,
                        start_ms=data_time_ms,
                        websocket=False,
                        lag_ms=lag_ms,
                        bid=bid,
                        ask=ask
                    )

                    if attempting_previous_close and tp_to_price[tp]:
                        self.closed_market_prices[tp] = tp_to_price[tp]
            else:
                logger.warning(f"Tiingo forex API request failed with status code {requestResponse.status_code}. "
                                 f"URL: {url[:100]}... Response: {requestResponse.text[:200]}")
                continue

        return tp_to_price

    @exception_handler_decorator()
    def _get_closes_crypto(self, trade_pairs: List[TradePair], target_time_ms: int, live: bool) -> dict[TradePair, PriceSource]:
        def tickers_to_crypto_url(tickers: List[str]) -> str:
            if not live:
                start_day_formatted, end_day_formatted = self.target_ms_to_start_end_formatted(target_time_ms)
                return f"https://api.tiingo.com/tiingo/crypto/prices?tickers={','.join(tickers)}&startDate={start_day_formatted}&endDate={end_day_formatted}&resampleFreq=1min&token={self.config['api_key']}&exchanges={TIINGO_COINBASE_EXCHANGE_STR.upper()}"
            return f"https://api.tiingo.com/tiingo/crypto/prices?tickers={','.join(tickers)}&resampleFreq=1min&token={self.config['api_key']}&exchanges={TIINGO_COINBASE_EXCHANGE_STR.upper()}"

        tp_to_price = {}
        if not trade_pairs:
            return tp_to_price

        assert all(tp.trade_pair_category == TradePairCategory.CRYPTO for tp in trade_pairs), trade_pairs

        poll_time_ms = TimeUtil.now_in_millis() if live else None
        batches = self._batch_trade_pairs(trade_pairs)

        for batch in batches:
            url = tickers_to_crypto_url([self.trade_pair_to_tiingo_ticker(x) for x in batch])
            requestResponse = requests.get(url, headers={'Content-Type': 'application/json'}, timeout=5)
            if requestResponse.status_code == 429:
                continue
            if requestResponse.status_code != 200:
                logger.warning(f"Tiingo crypto API request failed with status code {requestResponse.status_code}. "
                                 f"URL: {url[:100]}... Response: {requestResponse.text[:200]}")
                continue

            response_data = requestResponse.json()
            timespan_ms = self.timespan_to_ms['minute']

            for crypto_data in response_data:
                ticker = crypto_data['ticker']
                price_data = crypto_data.get('priceData', None)
                if not price_data:
                    continue

                closest_data = min(price_data,
                    key=lambda x: abs(TimeUtil.parse_iso_to_ms(x['date']) + timespan_ms - target_time_ms))

                candle_start_ms = TimeUtil.parse_iso_to_ms(closest_data["date"])

                if live:
                    data_time_ms = poll_time_ms
                else:
                    candle_close_ms = candle_start_ms + timespan_ms
                    data_time_ms = candle_close_ms

                price_close = float(closest_data['close'])
                bid_price = ask_price = 0

                tp = TradePair.get_latest_trade_pair_from_trade_pair_id(ticker.upper())
                source_name = f'{TIINGO_PROVIDER_NAME}_{TIINGO_COINBASE_EXCHANGE_STR}'

                tp_to_price[tp] = PriceSource(
                    source=source_name,
                    timespan_ms=timespan_ms,
                    open=float(closest_data['open']),
                    close=float(closest_data['close']),
                    vwap=price_close,
                    high=price_close,
                    low=float(closest_data['low']),
                    start_ms=data_time_ms,
                    websocket=False,
                    lag_ms=(poll_time_ms if live else target_time_ms) - data_time_ms,
                    bid=bid_price,
                    ask=ask_price,
                )

        return tp_to_price

    def trade_pair_to_tiingo_ticker(self, trade_pair: TradePair):
        return trade_pair.trade_pair_id.lower()


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
            logger.info(f"Found stale Tiingo websocket data for {trade_pair.trade_pair}. Lag_s: {lag_s} "
                            f"seconds. Max allowed lag for category: {max_allowed_lag_s} seconds. Ignoring this data.")
        return cur_event



if __name__ == "__main__":
    secrets = ValiUtils.get_secrets()
    tds = TiingoDataService(api_key=secrets['tiingo_apikey'], disable_ws=True)
    now_ms = TimeUtil.now_in_millis()
    tp_to_prices = tds.get_price_rest([TradePair.TAOUSD], now_ms, live=True)
    print('@@@@@', tp_to_prices)
    target_timestamp_ms = 1715288502999
    tp_to_prices = tds.get_price_rest([TradePair.BTCUSD, TradePair.USDJPY, TradePair.NVDA], target_timestamp_ms, live=False)
    print({x.trade_pair_id: y for x, y in tp_to_prices.items()})




