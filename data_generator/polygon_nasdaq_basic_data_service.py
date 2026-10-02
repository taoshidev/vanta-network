import json
import queue
import threading
import time
from collections import Counter
from typing import List

from polygon.websocket import WebSocketClient, Feed, Market

from data_generator.base_data_service import BaseDataService, POLYGON_PROVIDER_NAME
from shared_objects.error_utils import ErrorUtils
from time_util.time_util import TimeUtil
from vali_objects.trade_pair import TradePair, TradePairCategory, TradePairSource
from vali_objects.vali_config import ValiConfig
from vali_objects.vali_dataclasses.price_source import PriceSource, NASDAQ_BASIC_SOURCE
from shared_objects.log import logger

NASDAQ_BASIC_PROVIDER_NAME = f"{POLYGON_PROVIDER_NAME}_nasdaq"


class PolygonNasdaqBasicDataService(BaseDataService):
    """
    Nasdaq Basic quotes (Q.<ticker>) from the Polygon/Massive nasdaq-basic-business feed for
    Vanta-sourced equities.

    Entitlement is detected from the subscribe replies: "success" enables quotes, while "not authorized"
    disables them and closes the connection until the next retry. Validators without the Nasdaq Basic
    expansion keep pricing equities from the Business FMV feed exactly as before.

    Storage: the newest quote per ticker is always kept, and the last quote of every
    NASDAQ_QUOTE_TRACKER_SAMPLE_MS window goes into the recent event tracker. Messages are parsed off the
    websocket's event loop by a worker thread so a slow parse cannot back up the socket.
    """

    def __init__(self, api_key, disable_ws=False, running_unit_tests=False):
        self._api_key = api_key
        enabled_websocket_categories = {TradePairCategory.EQUITIES} if self.get_tradeable_pairs(
            category=TradePairCategory.EQUITIES, include_blocked=False, src=TradePairSource.VANTA) else set()
        super().__init__(
            provider_name=NASDAQ_BASIC_PROVIDER_NAME,
            running_unit_tests=running_unit_tests,
            enabled_websocket_categories=enabled_websocket_categories
        )

        # None until the first subscribe reply; True on "success", False on "not authorized"
        self.entitled = None
        self._entitlement_retry_at_s = 0.0
        self._next_connect_allowed_at_s = 0.0
        self._closing_denied_client = False
        self._window_start_ms = {}
        self._raw_queue = queue.Queue()
        self._stop_event = threading.Event()

        self._stats_lock = threading.Lock()
        self.n_subscribed = 0
        self.n_quotes = 0
        self.n_quotes_stored = 0
        self.n_quotes_out_of_order = 0
        self.dropped_counts = Counter()
        self._stats_since_s = time.time()

        self._worker_thread = None
        if disable_ws:
            self.websocket_manager_thread = None
        else:
            self._worker_thread = threading.Thread(target=self._process_queue_loop, daemon=True,
                                                   name="nasdaq_basic_quotes")
            self._worker_thread.start()
            self.websocket_manager_thread = threading.Thread(target=self.websocket_manager, daemon=True)
            self.websocket_manager_thread.start()

    def is_enabled(self) -> bool:
        return self.entitled is True

    @ErrorUtils.require_test_mode
    def set_test_entitlement(self, entitled: bool | None) -> None:
        self.entitled = entitled

    @ErrorUtils.require_test_mode
    def set_test_quote(self, trade_pair: TradePair, price_source: PriceSource) -> None:
        """Inject a quote as the newest for trade_pair and enable quotes. Applies the same filter as received quotes."""
        self.entitled = True
        if self._quote_rejection(price_source.bid, price_source.ask, 1, 1):
            return
        symbol = trade_pair.trade_pair
        self.trade_pair_to_recent_events[symbol].add_event(price_source)
        latest = self.latest_websocket_events.get(symbol)
        if latest is None or price_source.start_ms >= latest.start_ms:
            self.latest_websocket_events[symbol] = price_source

    @ErrorUtils.require_test_mode
    def clear_test_quotes(self) -> None:
        self.entitled = None
        self.latest_websocket_events.clear()
        self.trade_pair_to_recent_events.clear()
        self._window_start_ms.clear()

    def stop_threads(self):
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=1)
        super().stop_threads()

    def get_price_rest(self, trade_pairs: List[TradePair], timestamp_ms: int, live: bool) -> dict[TradePair, PriceSource]:
        # Quotes come from the websocket only
        return {}

    def instantiate_not_pickleable_objects(self):
        pass

    # ==================== Websocket lifecycle ====================

    def _create_websocket_client(self, tpc: TradePairCategory):
        now_s = time.time()
        if self.entitled is False and now_s < self._entitlement_retry_at_s:
            self.WEBSOCKET_OBJECTS[tpc] = None
            return
        if now_s < self._next_connect_allowed_at_s:
            self.WEBSOCKET_OBJECTS[tpc] = None
            return

        self._next_connect_allowed_at_s = now_s + ValiConfig.NASDAQ_MIN_RECONNECT_INTERVAL_S
        self._closing_denied_client = False
        # raw=True so subscribe status replies reach handle_msg; the client otherwise swallows them.
        # max_reconnects=0 hands reconnects to the base manager, which spaces them out.
        self.WEBSOCKET_OBJECTS[tpc] = WebSocketClient(market=Market.Stocks, api_key=self._api_key,
                                                      feed=Feed.NasdaqBasicBusiness, raw=True, max_reconnects=0)
        logger.info(f"Created {self.provider_name} websocket for {tpc}. feed {Feed.NasdaqBasicBusiness.name}")

    def _client_unavailable_delay_s(self, tpc) -> float:
        retry_at_s = self._entitlement_retry_at_s if self.entitled is False else 0.0
        return max(1.0, max(retry_at_s, self._next_connect_allowed_at_s) - time.time())

    def _subscribe_websockets(self, tpc: TradePairCategory = None):
        client = self.WEBSOCKET_OBJECTS.get(TradePairCategory.EQUITIES)
        if client is None:
            return
        symbols = ["Q." + tp.trade_pair for tp in self.get_tradeable_pairs(
            category=TradePairCategory.EQUITIES, include_blocked=False, src=TradePairSource.VANTA)]
        client.subscribe(*symbols)
        self.n_subscribed = len(symbols)
        logger.info(f"{self.provider_name} subscribing to {len(symbols)} quote symbols")

    async def handle_msg(self, raw):
        recv_ms = TimeUtil.now_in_millis()
        self.tpc_to_last_event_time[TradePairCategory.EQUITIES] = time.time()
        if isinstance(raw, bytes):
            raw = raw.decode()

        # Status replies are rare (one per subscription at connect) and decide entitlement, so handle
        # them here where the client can be closed. Quote batches go to the worker thread.
        if '"status"' not in raw:
            self._raw_queue.put((raw, recv_ms))
            return

        quotes = []
        for m in json.loads(raw):
            if m.get('ev') != 'status':
                quotes.append(m)
            elif self._apply_status(m) == 'denied' and not self._closing_denied_client:
                self._closing_denied_client = True
                client = self.WEBSOCKET_OBJECTS.get(TradePairCategory.EQUITIES)
                if client is not None:
                    await client.close()
        if quotes:
            self._raw_queue.put((quotes, recv_ms))

    def _apply_status(self, m: dict) -> str | None:
        status = m.get('status')
        message = m.get('message') or ''
        if 'not authorized' in message.lower():
            if self.entitled is not False:
                logger.warning(f"Nasdaq Basic quotes: DISABLED ({message}). Equities stay on Business FMV. "
                               f"Re-checking in {ValiConfig.NASDAQ_ENTITLEMENT_RETRY_S}s")
            self.entitled = False
            self._entitlement_retry_at_s = time.time() + ValiConfig.NASDAQ_ENTITLEMENT_RETRY_S
            # No quotes will arrive, so don't let the health check treat the silence as a stale socket
            self.tpc_to_last_event_time[TradePairCategory.EQUITIES] = 0
            return 'denied'
        if status == 'max_connections':
            # Another connection on this key is using the slot. Reconnecting quickly would just
            # bump it (Massive drops the older connection), so back off for the retry interval.
            logger.error(f"{self.provider_name} max_connections: {message}. Retrying in "
                         f"{ValiConfig.NASDAQ_ENTITLEMENT_RETRY_S}s")
            self._next_connect_allowed_at_s = time.time() + ValiConfig.NASDAQ_ENTITLEMENT_RETRY_S
            return 'max_connections'
        if status == 'success' and message.lower().startswith('subscribed to'):
            if self.entitled is not True:
                logger.info(f"Nasdaq Basic quotes: ENABLED ({message})")
            self.entitled = True
            return 'subscribed'
        logger.info(f"{self.provider_name} status: {m}")
        return None

    # ==================== Quote processing ====================

    def _process_queue_loop(self):
        while not self._stop_event.is_set():
            try:
                raw, recv_ms = self._raw_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._process_raw(raw, recv_ms)
            except Exception as e:
                logger.error(f"{self.provider_name} failed to process message: {type(e).__name__}: {e}")

    def _process_raw(self, raw, recv_ms: int):
        msgs = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        for m in msgs:
            if m.get('ev') == 'Q':
                self._add_quote(m, recv_ms)

    def _add_quote(self, m: dict, now_ms: int):
        symbol = m.get('sym')
        tp = self.trade_pair_lookup.get(symbol)
        t = m.get('t')
        if tp is None or not tp.is_equities or t is None:
            return

        with self._stats_lock:
            self.n_quotes += 1

        bid = m.get('bp') or 0.0
        ask = m.get('ap') or 0.0
        reason = self._quote_rejection(bid, ask, m.get('bs') or 0, m.get('as') or 0)
        if reason:
            with self._stats_lock:
                self.dropped_counts[reason] += 1
            return

        # Quotes carry no sequence number and can arrive a few ms behind the previous quote. Like the other
        # sources, order by timestamp: a quote older than the latest is ignored. Quotes keep their real
        # timestamps, so stale ones (e.g. replays after a reconnect) fail the WEBSOCKET_PRICE_MAX_AGE_MS check.
        latest = self.latest_websocket_events.get(symbol)
        if latest is not None and t < latest.start_ms:
            with self._stats_lock:
                self.n_quotes_out_of_order += 1
            return

        mid = (bid + ask) / 2.0
        ps = PriceSource(
            source=NASDAQ_BASIC_SOURCE,
            timespan_ms=0,
            open=mid,
            close=mid,
            vwap=mid,
            high=mid,
            low=mid,
            start_ms=t,
            websocket=True,
            lag_ms=now_ms - t,
            bid=bid,
            ask=ask
        )

        # When a new sample window starts, the newest quote of the previous window goes into the tracker
        window_start_ms = self._window_start_ms.get(symbol)
        if window_start_ms is None or t - window_start_ms >= ValiConfig.NASDAQ_QUOTE_TRACKER_SAMPLE_MS:
            if latest is not None:
                self.trade_pair_to_recent_events[symbol].add_event(latest, tp_debug_str=f"{self.provider_name}:{symbol}")
                with self._stats_lock:
                    self.n_quotes_stored += 1
            self._window_start_ms[symbol] = t
        self.latest_websocket_events[symbol] = ps

    @staticmethod
    def _quote_rejection(bid: float, ask: float, bid_size: float, ask_size: float) -> str | None:
        """
        Why a quote cannot be traded against, or None. Such quotes are dropped on receipt (like the Polygon forex
        spread filter in parse_price_for_forex), so the previous quote stays current until it ages out.
        """
        if bid <= 0 or ask <= 0 or bid_size <= 0 or ask_size <= 0:
            return 'missing_side'
        if bid >= ask:
            return 'crossed_or_locked'
        if (ask - bid) / ((bid + ask) / 2.0) * 10000 > ValiConfig.NASDAQ_QUOTE_MAX_SPREAD_BPS:
            return 'wide_spread'
        return None

    # ==================== Lookup ====================

    def get_closest_quote(self, trade_pair: TradePair, time_ms: int) -> PriceSource | None:
        """Closest quote to time_ms, like the other websocket sources. The newest quote is not in the
        tracker until its sample window closes, so it is compared separately."""
        symbol = trade_pair.trade_pair
        latest = self.latest_websocket_events.get(symbol)
        tracker = self.trade_pair_to_recent_events.get(symbol)
        sampled = tracker.get_closest_event(time_ms) if tracker else None
        candidates = [ps for ps in (latest, sampled) if ps is not None]
        return min(candidates, key=lambda ps: abs(time_ms - ps.start_ms)) if candidates else None

    def get_events_in_range(self, trade_pair: TradePair, start_ms: int, end_ms: int) -> List[PriceSource]:
        """Sampled quotes in [start_ms, end_ms] plus the newest quote, which is not sampled until its window closes."""
        symbol = trade_pair.trade_pair
        tracker = self.trade_pair_to_recent_events.get(symbol)
        events = tracker.get_events_in_range(start_ms, end_ms) if tracker else []
        latest = self.latest_websocket_events.get(symbol)
        if latest is not None and start_ms <= latest.start_ms <= end_ms and not any(e is latest for e in events):
            events.append(latest)
        return events

    def get_closes_websocket(self, trade_pairs: List[TradePair], time_ms) -> dict[TradePair, PriceSource]:
        events = {}
        for trade_pair in trade_pairs:
            quote = self.get_closest_quote(trade_pair, time_ms)
            if quote is not None:
                events[trade_pair] = quote
        return events

    def debug_log(self):
        now_s = time.time()
        with self._stats_lock:
            elapsed_s = max(now_s - self._stats_since_s, 1e-9)
            n_quotes, n_stored, n_out_of_order = self.n_quotes, self.n_quotes_stored, self.n_quotes_out_of_order
            dropped = dict(self.dropped_counts)
            self.n_quotes = self.n_quotes_stored = self.n_quotes_out_of_order = 0
            self.dropped_counts = Counter()
            self._stats_since_s = now_s
        n_tracker_events = sum(t.count_events() for t in list(self.trade_pair_to_recent_events.values()))
        logger.info(
            f"[NASDAQ_BASIC] entitled={self.entitled} subscribed={self.n_subscribed} "
            f"tickers_quoted={len(self.latest_websocket_events)} quotes/s={n_quotes / elapsed_s:.1f} "
            f"stored/s={n_stored / elapsed_s:.1f} out_of_order={n_out_of_order} queue={self._raw_queue.qsize()} "
            f"tracker_events={n_tracker_events} dropped={dropped}"
        )
