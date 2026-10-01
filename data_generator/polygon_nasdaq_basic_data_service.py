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
from vali_objects.vali_dataclasses.price_source import PriceSource
from shared_objects.log import logger

NASDAQ_BASIC_PROVIDER_NAME = f"{POLYGON_PROVIDER_NAME}_nasdaq"
NASDAQ_BASIC_SOURCE = f"{NASDAQ_BASIC_PROVIDER_NAME}_ws"


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
        self.last_message_ms = 0

        self._window_start_ms = {}
        self._raw_queue = queue.Queue()
        self._stop_event = threading.Event()

        self._stats_lock = threading.Lock()
        self.n_subscribed = 0
        self.n_quotes = 0
        self.n_quotes_stored = 0
        self.n_quotes_out_of_order = 0
        self.n_valid = 0
        self.rejection_counts = Counter()
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
        self.last_message_ms = recv_ms
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
        self.last_message_ms = max(self.last_message_ms, recv_ms)
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

        # Quotes carry no sequence number and can arrive a few ms behind the previous quote's timestamp.
        # Arrival order is the book's order, so a slightly late quote is still the newest state; clamp its
        # timestamp to keep the ticker's quotes monotonic. Only drop quotes far behind (stale replays).
        latest = self.latest_websocket_events.get(symbol)
        if latest is not None and t < latest.start_ms:
            with self._stats_lock:
                self.n_quotes_out_of_order += 1
            if latest.start_ms - t > ValiConfig.NASDAQ_QUOTE_OUT_OF_ORDER_TOLERANCE_MS:
                return
            t = latest.start_ms

        bp = m.get('bp') or 0.0
        ap = m.get('ap') or 0.0
        # A side with no size has no executable price. Zero it so validation rejects this quote as the
        # current market instead of falling back to an older one.
        bid = bp if (m.get('bs') or 0) > 0 else 0.0
        ask = ap if (m.get('as') or 0) > 0 else 0.0
        mid = (bp + ap) / 2.0 if bp > 0 and ap > 0 else (bp or ap)
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

    # ==================== Lookup and validation ====================

    def get_quote_at_or_before(self, trade_pair: TradePair, time_ms: int) -> PriceSource | None:
        symbol = trade_pair.trade_pair
        latest = self.latest_websocket_events.get(symbol)
        if latest is not None and latest.start_ms <= time_ms:
            return latest
        tracker = self.trade_pair_to_recent_events.get(symbol)
        return tracker.get_latest_event_at_or_before(time_ms) if tracker else None

    def get_valid_quote(self, trade_pair: TradePair, time_ms: int, fmv_ps: PriceSource | None = None,
                        max_age_ms: int | None = None) -> tuple[PriceSource | None, str | None]:
        """
        Return (quote, None) for the newest quote at or before time_ms that passes every validity rule,
        or (None, reason) for the first rule it fails.
        """
        if not self.is_enabled():
            return None, 'disabled'
        if not trade_pair.is_equities or trade_pair.src != TradePairSource.VANTA:
            return None, 'not_applicable'

        if max_age_ms is None:
            max_age_ms = ValiConfig.NASDAQ_QUOTE_MAX_AGE_MS
        quote = None
        reason = None
        if not self.is_market_open(trade_pair, time_ms):
            reason = 'market_closed'
        else:
            quote = self.get_quote_at_or_before(trade_pair, time_ms)
            if quote is None:
                reason = 'no_quote'
            elif TimeUtil.now_in_millis() - self.last_message_ms > ValiConfig.NASDAQ_QUOTE_FEED_HEALTH_MS:
                reason = 'feed_unhealthy'
            elif time_ms - quote.start_ms > max_age_ms:
                reason = 'stale'
            elif not (quote.bid > 0 and quote.ask > 0):
                reason = 'missing_side'
            elif quote.bid >= quote.ask:
                reason = 'crossed_or_locked'
            else:
                mid = (quote.bid + quote.ask) / 2.0
                spread_bps = (quote.ask - quote.bid) / mid * 10000
                if spread_bps > ValiConfig.NASDAQ_QUOTE_MAX_SPREAD_BPS:
                    reason = 'wide_spread'
                elif (fmv_ps is not None and fmv_ps.open and fmv_ps.open > 0
                      and abs(time_ms - fmv_ps.start_ms) <= ValiConfig.NASDAQ_QUOTE_FMV_MAX_AGE_MS):
                    band_bps = max(ValiConfig.NASDAQ_QUOTE_FMV_BAND_BPS, spread_bps)
                    if abs(mid - fmv_ps.open) / fmv_ps.open * 10000 > band_bps:
                        reason = 'fmv_deviation'

        with self._stats_lock:
            if reason:
                self.rejection_counts[reason] += 1
            else:
                self.n_valid += 1
        return (None, reason) if reason else (quote, None)

    def debug_log(self):
        now_s = time.time()
        with self._stats_lock:
            elapsed_s = max(now_s - self._stats_since_s, 1e-9)
            n_quotes, n_stored, n_out_of_order = self.n_quotes, self.n_quotes_stored, self.n_quotes_out_of_order
            n_valid, rejections = self.n_valid, dict(self.rejection_counts)
            self.n_quotes = self.n_quotes_stored = self.n_quotes_out_of_order = self.n_valid = 0
            self.rejection_counts = Counter()
            self._stats_since_s = now_s
        n_tracker_events = sum(t.count_events() for t in list(self.trade_pair_to_recent_events.values()))
        logger.info(
            f"[NASDAQ_BASIC] entitled={self.entitled} subscribed={self.n_subscribed} "
            f"tickers_quoted={len(self.latest_websocket_events)} quotes/s={n_quotes / elapsed_s:.1f} "
            f"stored/s={n_stored / elapsed_s:.1f} out_of_order={n_out_of_order} queue={self._raw_queue.qsize()} "
            f"tracker_events={n_tracker_events} valid={n_valid} rejections={rejections}"
        )
