import asyncio
import json
import threading
import time
import traceback
from typing import List

import bittensor as bt
import requests
import websockets

from data_generator.base_data_service import BaseDataService, HYPERLIQUID_PROVIDER_NAME
from entity_management.hl_orderbook_utils import simulate_fill
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory, TradePairSource, ValiConfig
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker

REST_TIMEOUT_S = 10
RECV_TIMEOUT_S = 30
# Reject a cached L2 book older than this for slippage/fill simulation, rather than
# silently using arbitrarily old data from a feed that has stopped updating (e.g. a
# resolution stuck between disconnect and reconnect). Set comfortably above the
# per-resolution reconnect backoff cap (30s) plus resubscribe time.
L2_BOOK_STALENESS_MS = 60_000

# (interval, candle_span_ms) - sorted by span ascending, threshold is span * 5000 candles
HL_CANDLE_INTERVALS = [
    ("1m", 60 * 1000),
    ("5m", 5 * 60 * 1000),
    ("15m", 15 * 60 * 1000),
    ("1h", 60 * 60 * 1000),
    ("12h", 12 * 60 * 60 * 1000),
    ("1d", 24 * 60 * 60 * 1000),
]


class _HyperliquidWebsocketClient:
    """Websocket client for Hyperliquid L2 orderbook data at a given resolution."""

    def __init__(self, service, category, n_sig_figs: int | None = None):
        self._svc = service
        self._cat = category
        self._n_sig_figs = n_sig_figs
        self._ws = None
        self._should_close = False

    async def connect(self, handle_msg):
        """Connect to Hyperliquid L2 orderbook websocket and process messages."""
        self._ws = await websockets.connect(ValiConfig.hl_ws_url())

        try:
            # Get the filtered, env-aware coin list from the service (filtered by
            # allMids availability, across default and non-default dexes, to prevent
            # testnet socket closes).
            coins = self._svc._get_subscription_coins()

            for coin in coins:
                subscription = {"type": "l2Book", "coin": coin}
                if self._n_sig_figs is not None:
                    subscription["nSigFigs"] = self._n_sig_figs
                subscribe_msg = {"method": "subscribe", "subscription": subscription}
                await self._ws.send(json.dumps(subscribe_msg))

            precision = f"nSigFigs={self._n_sig_figs}" if self._n_sig_figs is not None else "full precision"
            bt.logging.info(f"Subscribed to Hyperliquid l2Book ({precision}) for "
                            f"{len(coins)} coins: {sorted(coins)}")

            # Receive loop
            while not self._should_close:
                try:
                    msg = await asyncio.wait_for(self._ws.recv(), timeout=RECV_TIMEOUT_S)
                    await handle_msg(msg)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    try:
                        await self._ws.ping()
                    except Exception:
                        bt.logging.warning("Hyperliquid websocket ping failed, reconnecting")
                        break
                except websockets.exceptions.ConnectionClosed as e:
                    bt.logging.warning(f"Hyperliquid websocket closed: code={e.code}, reason={e.reason}")
                    break
                except Exception as e:
                    if self._should_close:
                        break
                    bt.logging.error(f"Error receiving Hyperliquid message: {type(e).__name__}: {e}")
                    continue

        finally:
            if self._ws:
                await self._ws.close()

    async def close(self):
        self._should_close = True
        if self._ws:
            await self._ws.close()

    def unsubscribe_all(self):
        self._should_close = True


class _MultiResolutionL2BookClient:
    """Manages one concurrent L2 book WebSocket connection per resolution in the cascade.

    Full precision (no nSigFigs) provides native tick-size pricing for accurate mid and
    near-spread slippage, and is the only resolution that feeds the validator price feed.
    Each successively coarser resolution (ValiConfig.HL_L2_SIG_FIGS_CASCADE[1:]) provides
    wider depth for large orders that exhaust the finer levels. Messages are routed to the
    service's resolution-specific handlers.
    """

    def __init__(self, service, category):
        self._svc = service
        cascade = ValiConfig.HL_L2_SIG_FIGS_CASCADE
        self._full = _HyperliquidWebsocketClient(service, category, n_sig_figs=cascade[0])
        self._coarse_sig_figs = cascade[1:]
        self._coarse_clients = [
            _HyperliquidWebsocketClient(service, category, n_sig_figs=sig_figs)
            for sig_figs in self._coarse_sig_figs
        ]

    async def _run_with_reconnect(self, client, handle_msg, label):
        """Run one resolution's websocket client, reconnecting it independently (with
        backoff) if its connection drops, without waiting on the other resolutions in
        the cascade to also disconnect.
        """
        backoff = 1.0
        while not client._should_close:
            try:
                await client.connect(handle_msg)
            except Exception as e:
                bt.logging.error(f"Hyperliquid websocket client ({label}) error: {type(e).__name__}: {e}")
            if client._should_close:
                break
            bt.logging.warning(f"Hyperliquid websocket client ({label}) disconnected, "
                               f"reconnecting in {backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 1.5, 30.0)

    async def connect(self, handle_msg):
        """Run one WebSocket connection per cascade resolution concurrently, each
        reconnecting independently on disconnect."""
        coros = [self._run_with_reconnect(self._full, self._svc.handle_msg_full, "full precision")]
        for client, sig_figs in zip(self._coarse_clients, self._coarse_sig_figs):
            coros.append(self._run_with_reconnect(
                client,
                lambda msg, sf=sig_figs: self._svc.handle_msg_coarse(sf, msg),
                f"nSigFigs={sig_figs}",
            ))
        await asyncio.gather(*coros)

    async def close(self):
        await asyncio.gather(self._full.close(), *[c.close() for c in self._coarse_clients])

    def unsubscribe_all(self):
        self._full.unsubscribe_all()
        for c in self._coarse_clients:
            c.unsubscribe_all()


class HyperliquidDataService(BaseDataService):
    """Crypto-only live WebSocket feed from Hyperliquid using L2 orderbook data."""

    def __init__(self, disable_ws=False, running_unit_tests=False):
        super().__init__(
            provider_name=HYPERLIQUID_PROVIDER_NAME,
            running_unit_tests=running_unit_tests,
            enabled_websocket_categories={TradePairCategory.CRYPTO}
        )

        # Build coin name -> TradePair mapping for static pairs.
        self._coin_to_trade_pair: dict[str, TradePair] = {}
        for tp in TradePair:
            if tp.src == TradePairSource.HYPERLIQUID and not tp.is_blocked:
                self._coin_to_trade_pair[tp.hl_coin] = tp

        # Multi-resolution L2 orderbook cache per coin, per ValiConfig.HL_L2_SIG_FIGS_CASCADE.
        # Full precision (no nSigFigs): native tick-size pricing for accurate mid and near-spread slippage.
        # Each coarser resolution (5, 4, 3, 2 sig figs): wider depth for large orders that exhaust
        # the finer resolutions.
        # Value shape: {"bids": [...], "asks": [...], "time": timestamp_ms}
        self._orderbooks_full: dict[str, dict] = {}
        # Keyed by sig_figs, then coin name (e.g. "BTC").
        self._orderbooks_coarse_by_sigfigs: dict[int, dict[str, dict]] = {
            sig_figs: {} for sig_figs in ValiConfig.HL_L2_SIG_FIGS_CASCADE[1:]
        }

        if disable_ws:
            self.websocket_manager_thread = None
        else:
            self.websocket_manager_thread = threading.Thread(
                target=self.websocket_manager, daemon=True
            )
            self.websocket_manager_thread.start()

    def _create_websocket_client(self, tpc):
        if tpc != TradePairCategory.CRYPTO:
            return
        client = _MultiResolutionL2BookClient(self, tpc)
        self.WEBSOCKET_OBJECTS[tpc] = client
        bt.logging.info(f"Created {self.provider_name} dual-resolution websocket client for {tpc}")

    def _subscribe_websockets(self, tpc):
        # Subscription happens inside connect()
        pass

    def _parse_l2_book_msg(self, msg):
        """Parse an l2Book WebSocket message.

        Returns (coin, tp, bids, asks, timestamp_ms) or None.
        tp is a TradePair for known static coins, or None (whole result is None)
        for unknown coins.
        """
        data = json.loads(msg)
        if data.get("channel") != "l2Book":
            return None
        book_data = data.get("data", {})
        coin = book_data.get("coin")
        if not coin:
            return None
        tp = self._coin_to_trade_pair.get(coin)
        if tp is None:
            return None
        levels = book_data.get("levels", [])
        if len(levels) < 2 or not levels[0] or not levels[1]:
            return None
        timestamp_ms = round(book_data.get("time", TimeUtil.now_in_millis()), -3)
        return coin, tp, levels[0], levels[1], timestamp_ms

    async def handle_msg_full(self, msg):
        """Handle nSigFigs=None l2Book messages: update price feed and full orderbook cache."""
        try:
            parsed = self._parse_l2_book_msg(msg)
            if parsed is None:
                return
            coin, tp, bids, asks, timestamp_ms = parsed

            self._orderbooks_full[coin] = {"bids": bids, "asks": asks, "time": timestamp_ms}
            self.tpc_to_n_events[TradePairCategory.CRYPTO] += 1
            self.tpc_to_last_event_time[TradePairCategory.CRYPTO] = time.time()

            # Only push to the validator price feed for static TradePair enum members.
            # Dynamic altcoins provide orderbook data for slippage but are not validator price sources.
            if isinstance(tp, TradePair):
                best_bid = float(bids[0]["px"])
                best_ask = float(asks[0]["px"])
                mid_price = (best_bid + best_ask) / 2.0
                now_ms = TimeUtil.now_in_millis()
                ps = PriceSource(
                    source=f"{HYPERLIQUID_PROVIDER_NAME}_ws",
                    timespan_ms=0,
                    open=mid_price,
                    close=mid_price,
                    vwap=mid_price,
                    high=mid_price,
                    low=mid_price,
                    start_ms=timestamp_ms,
                    websocket=True,
                    lag_ms=now_ms - timestamp_ms,
                    bid=best_bid,
                    ask=best_ask,
                )
                symbol = tp.trade_pair
                self.latest_websocket_events[symbol] = ps
                if symbol not in self.trade_pair_to_recent_events:
                    self.trade_pair_to_recent_events[symbol] = RecentEventTracker()
                self.trade_pair_to_recent_events[symbol].add_event(
                    ps, False, f"{self.provider_name}:{tp.trade_pair}"
                )
                self.closed_market_prices[tp] = None

        except Exception as e:
            limited_traceback = traceback.format_exc()[-1000:]
            bt.logging.error(
                f"Failed to handle {HYPERLIQUID_PROVIDER_NAME} full websocket message "
                f"with error: {e}, type: {type(e).__name__}, traceback: {limited_traceback}"
            )

    async def handle_msg_coarse(self, sig_figs: int, msg):
        """Handle a coarse-resolution l2Book message: update that resolution's orderbook cache."""
        try:
            parsed = self._parse_l2_book_msg(msg)
            if parsed is None:
                return
            coin, _tp, bids, asks, timestamp_ms = parsed
            self._orderbooks_coarse_by_sigfigs.setdefault(sig_figs, {})[coin] = {
                "bids": bids, "asks": asks, "time": timestamp_ms
            }
        except Exception as e:
            limited_traceback = traceback.format_exc()[-1000:]
            bt.logging.error(
                f"Failed to handle {HYPERLIQUID_PROVIDER_NAME} coarse (nSigFigs={sig_figs}) websocket message "
                f"with error: {e}, type: {type(e).__name__}, traceback: {limited_traceback}"
            )

    def _fetch_all_mids(self) -> dict[str, float]:
        """Fetch mid prices for all coins across all dexes via the REST API.

        Fetches the default crypto dex first, then merges in each non-default dex
        (identified by the colon-prefixed hl_coin names in the configured coin set).
        Returns {coin: mid_price} with prefixed keys for non-default dex coins (e.g. "xyz:AAPL").
        """
        result: dict[str, float] = {}

        # Default dex
        try:
            resp = requests.post(
                ValiConfig.hl_info_url(),
                json={"type": "allMids"},
                timeout=REST_TIMEOUT_S,
            )
            resp.raise_for_status()
            result.update({coin: float(price) for coin, price in resp.json().items()})
        except Exception as e:
            bt.logging.error(f"Hyperliquid REST allMids (default dex) failed: {type(e).__name__}: {e}")

        # Non-default dexes — derive names from prefixed hl_coin entries in the configured coin set
        non_default_dexes = {
            coin.split(":")[0]
            for coin in self._coin_to_trade_pair.keys()
            if ":" in coin
        }
        for dex in non_default_dexes:
            try:
                resp = requests.post(
                    ValiConfig.hl_info_url(),
                    json={"type": "allMids", "dex": dex},
                    timeout=REST_TIMEOUT_S,
                )
                resp.raise_for_status()
                result.update({coin: float(price) for coin, price in resp.json().items()})
            except Exception as e:
                bt.logging.error(f"Hyperliquid REST allMids (dex={dex}) failed: {type(e).__name__}: {e}")

        return result

    def _fetch_l2_book(self, coin: str) -> tuple[float, float] | None:
        """Fetch best bid/ask for a single coin via the REST API."""
        try:
            resp = requests.post(
                ValiConfig.hl_info_url(),
                json={"type": "l2Book", "coin": coin},
                timeout=REST_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
            levels = data.get("levels", [])
            if len(levels) < 2 or not levels[0] or not levels[1]:
                return None
            best_bid = float(levels[0][0]["px"])
            best_ask = float(levels[1][0]["px"])
            return best_bid, best_ask
        except Exception as e:
            bt.logging.error(f"Hyperliquid REST l2Book({coin}) failed: {type(e).__name__}: {e}")
            return None

    def _fetch_candle_snapshot(self, hl_coin: str, target_ms: int) -> PriceSource | None:
        """
        Fetch the candle closest to target_ms using the candleSnapshot endpoint.
        The candle interval is dynamically selected based on how far back target_ms is from now:
        """
        now_ms = TimeUtil.now_in_millis()
        age_ms = now_ms - target_ms

        # Select interval based on age (5000-candle limit per request)
        interval, candle_span_ms = HL_CANDLE_INTERVALS[-1]
        for _interval, span_ms in HL_CANDLE_INTERVALS:
            if age_ms < span_ms * 5000:
                interval, candle_span_ms = _interval, span_ms
                break
        start_ms = target_ms - 3 * candle_span_ms
        end_ms = target_ms + candle_span_ms

        req: dict = {"coin": hl_coin, "interval": interval, "startTime": start_ms, "endTime": end_ms}

        try:
            resp = requests.post(
                ValiConfig.hl_info_url(),
                json={"type": "candleSnapshot", "req": req},
                timeout=REST_TIMEOUT_S,
            )
            resp.raise_for_status()
            candles = resp.json()
        except Exception as e:
            bt.logging.error(f"Hyperliquid candleSnapshot({hl_coin}) failed: {type(e).__name__}: {e}")
            return None

        if not candles:
            return None

        # Pick the candle whose open time is closest to (and not after) target_ms.
        best = min(candles, key=lambda c: abs(target_ms - int(c["t"])))
        candle_start_ms = int(best["t"])

        return PriceSource(
            source=f"{HYPERLIQUID_PROVIDER_NAME}_candle",
            timespan_ms=candle_span_ms,
            open=float(best["o"]),
            close=float(best["c"]),
            high=float(best["h"]),
            low=float(best["l"]),
            vwap=float(best["c"]),  # HL candles have no vwap; use close as best proxy
            start_ms=candle_start_ms,
            websocket=False,
            lag_ms=target_ms - candle_start_ms,
        )

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
        if self.running_unit_tests:
            from data_generator.polygon_data_service import PolygonDataService
            return {tp: PolygonDataService.DEFAULT_TESTING_FALLBACK_PRICE_SOURCE for tp in trade_pairs}

        hl_pairs = [tp for tp in trade_pairs if tp.src == TradePairSource.HYPERLIQUID and not tp.is_blocked]
        if not hl_pairs:
            return {}

        now_ms = TimeUtil.now_in_millis()

        if not live:
            # Historical lookup — use candleSnapshot for each pair.
            results: dict[TradePair, PriceSource] = {}
            for tp in hl_pairs:
                price_source = self._fetch_candle_snapshot(tp.hl_coin, timestamp_ms)
                if price_source is not None:
                    results[tp] = price_source
            return results

        # Live lookup — use the bulk allMids endpoint first.
        all_mids = self._fetch_all_mids()

        results: dict[TradePair, PriceSource] = {}
        pairs_needing_book = []

        for tp in hl_pairs:
            mid = all_mids.get(tp.hl_coin)
            if mid is not None and mid > 0:
                results[tp] = PriceSource(
                    source=f"{HYPERLIQUID_PROVIDER_NAME}_rest",
                    timespan_ms=0,
                    open=mid,
                    close=mid,
                    vwap=mid,
                    high=mid,
                    low=mid,
                    start_ms=now_ms,
                    websocket=False,
                    lag_ms=0,
                )
            else:
                pairs_needing_book.append(tp)

        # Fall back to individual l2Book calls for any coins missing from allMids.
        for tp in pairs_needing_book:
            book = self._fetch_l2_book(tp.hl_coin)
            if book is None:
                continue
            best_bid, best_ask = book
            mid = (best_bid + best_ask) / 2.0
            results[tp] = PriceSource(
                source=f"{HYPERLIQUID_PROVIDER_NAME}_rest",
                timespan_ms=0,
                open=mid,
                close=mid,
                vwap=mid,
                high=mid,
                low=mid,
                start_ms=now_ms,
                websocket=False,
                lag_ms=0,
                bid=best_bid,
                ask=best_ask,
            )

        return results

    def simulate_slippage(self, trade_pair: TradePair, size_usd: float, is_buy: bool,
                           order_uuid: str = None) -> float | None:
        """Simulate slippage using an N-phase orderbook walk across the full sig-figs cascade.

        Phase 1 walks the full-precision book (nSigFigs=None) for accurate near-spread
        pricing. Each subsequent phase continues with the next-coarser resolution in
        ValiConfig.HL_L2_SIG_FIGS_CASCADE, using only levels priced beyond the last level
        consumed so far, for orders that exhaust the finer resolutions. Falls back to the
        coarsest available resolution if finer ones are not yet populated.

        Args:
            trade_pair: The trade pair to calculate slippage for.
            size_usd: The order size in USD.
            is_buy: True for LONG orders (fill against asks),
                    False for SHORT orders (fill against bids).
            order_uuid: Optional order identifier, included in the audit log line when
                        slippage exceeds ValiConfig.HL_SLIPPAGE_AUDIT_LOG_THRESHOLD.

        Returns:
            Slippage as a fraction (e.g. 0.001 for 0.1%), or None if no
            orderbook data is available.
        """
        coin = trade_pair.hl_coin
        cascade = ValiConfig.HL_L2_SIG_FIGS_CASCADE
        now_ms = TimeUtil.now_in_millis()

        full_book = self._orderbooks_full.get(coin, {})
        if full_book and now_ms - full_book.get("time", 0) > L2_BOOK_STALENESS_MS:
            full_book = {}
        books = []
        for sig_figs in cascade[1:]:
            book = self._orderbooks_coarse_by_sigfigs.get(sig_figs, {}).get(coin, {})
            if book and now_ms - book.get("time", 0) > L2_BOOK_STALENESS_MS:
                book = {}
            books.append(book)
        books = [full_book] + books

        primary = next((b for b in books if b), {})
        if not primary:
            return None

        bids = primary.get("bids", [])
        asks = primary.get("asks", [])
        if not bids or not asks:
            return None

        mid = (float(bids[0]["px"]) + float(asks[0]["px"])) / 2.0
        if mid <= 0:
            return None

        side = "asks" if is_buy else "bids"
        fills = []
        remaining = size_usd
        last_px = None
        for sig_figs, book in zip(cascade, books):
            if remaining <= 0:
                break
            levels = book.get(side, [])
            if not levels:
                continue
            if last_px is not None:
                levels = [l for l in levels if (float(l["px"]) > last_px if is_buy else float(l["px"]) < last_px)]
                if not levels:
                    continue
            phase_fills, remaining = simulate_fill(levels, remaining, "usd")
            fills.extend((sig_figs,) + f for f in phase_fills)
            last_px = float(levels[-1]["px"])

        if not fills:
            return None

        total_coins = sum(f[2] for f in fills)
        total_usd = sum(f[3] for f in fills)
        if total_coins <= 0:
            return None

        avg_price = total_usd / total_coins
        slippage_pct = max(0.0, (avg_price - mid) / mid if is_buy else (mid - avg_price) / mid)

        if slippage_pct > ValiConfig.HL_SLIPPAGE_AUDIT_LOG_THRESHOLD:
            fills_desc = [
                {"sig_figs": sig_figs, "price": px, "filled_coins": coins, "filled_usd": usd}
                for sig_figs, px, coins, usd in fills
            ]
            bt.logging.warning(
                f"[SLIPPAGE_AUDIT] order_uuid={order_uuid} {trade_pair.trade_pair_id} "
                f"size_usd={size_usd:.2f} is_buy={is_buy} mid={mid} avg_price={avg_price} "
                f"slippage_pct={slippage_pct:.6f} fills={fills_desc}"
            )

        return slippage_pct

    def simulate_avg_fill_price(self, trade_pair: TradePair, size_usd: float, is_buy: bool) -> float | None:
        """Simulate the average fill price for a market order using the L2 orderbook.

        Uses the same N-phase sig-figs cascade walk as simulate_slippage, but returns the
        raw avg fill price instead of a slippage fraction. This is used for HL taker fills
        where we want to record the actual execution price directly.

        Args:
            trade_pair: The trade pair to simulate.
            size_usd: The order size in USD.
            is_buy: True for LONG orders (fill against asks),
                    False for SHORT orders (fill against bids).

        Returns:
            Average fill price in quote currency, or None if no orderbook data is available.
        """
        coin = trade_pair.hl_coin
        cascade = ValiConfig.HL_L2_SIG_FIGS_CASCADE
        now_ms = TimeUtil.now_in_millis()

        full_book = self._orderbooks_full.get(coin, {})
        if full_book and now_ms - full_book.get("time", 0) > L2_BOOK_STALENESS_MS:
            full_book = {}
        books = []
        for sig_figs in cascade[1:]:
            book = self._orderbooks_coarse_by_sigfigs.get(sig_figs, {}).get(coin, {})
            if book and now_ms - book.get("time", 0) > L2_BOOK_STALENESS_MS:
                book = {}
            books.append(book)
        books = [full_book] + books

        primary = next((b for b in books if b), {})
        if not primary:
            return None

        bids = primary.get("bids", [])
        asks = primary.get("asks", [])
        if not bids or not asks:
            return None

        mid = (float(bids[0]["px"]) + float(asks[0]["px"])) / 2.0
        if mid <= 0:
            return None

        side = "asks" if is_buy else "bids"
        fills = []
        remaining = size_usd
        last_px = None
        for book in books:
            if remaining <= 0:
                break
            levels = book.get(side, [])
            if not levels:
                continue
            if last_px is not None:
                levels = [l for l in levels if (float(l["px"]) > last_px if is_buy else float(l["px"]) < last_px)]
                if not levels:
                    continue
            phase_fills, remaining = simulate_fill(levels, remaining, "usd")
            fills.extend(phase_fills)
            last_px = float(levels[-1]["px"])

        if not fills:
            return None

        total_coins = sum(f[1] for f in fills)
        total_usd = sum(f[2] for f in fills)
        if total_coins <= 0:
            return None

        return total_usd / total_coins

    def _get_subscription_coins(self) -> set[str]:
        """Return the filtered set of HL coins to subscribe to for l2Book streams.

        Intersects the statically configured coin set with allMids availability
        (default dex plus any non-default dexes referenced by colon-prefixed
        hl_coin names, e.g. "xyz:AAPL") to avoid subscribing to coins unsupported
        on the current HL env, which causes the HL server to close the WebSocket
        connection.
        """
        configured_coins = set(self._coin_to_trade_pair.keys())
        all_supported_keys: set[str] = set()

        # Default dex
        try:
            resp = requests.post(
                ValiConfig.hl_info_url(),
                json={"type": "allMids"},
                timeout=REST_TIMEOUT_S,
            )
            resp.raise_for_status()
            mids = resp.json()
            if isinstance(mids, dict):
                all_supported_keys.update(mids.keys())
        except Exception as e:
            bt.logging.warning(f"[HL_DATA_SVC] Failed to fetch allMids (default dex) for coin filtering: {e}")

        # Non-default dexes (e.g. HIP-3 equities/commodities/indices) aren't included
        # in the default-dex allMids response and must be queried separately.
        non_default_dexes = {
            coin.split(":")[0] for coin in configured_coins if ":" in coin
        }
        for dex in non_default_dexes:
            try:
                resp = requests.post(
                    ValiConfig.hl_info_url(),
                    json={"type": "allMids", "dex": dex},
                    timeout=REST_TIMEOUT_S,
                )
                resp.raise_for_status()
                all_supported_keys.update(resp.json().keys())
            except Exception as e:
                bt.logging.warning(f"[HL_DATA_SVC] Failed to fetch allMids for dex={dex}: {e}")

        if not all_supported_keys:
            # Every dex fetch failed outright — fail open rather than unsubscribing from everything.
            bt.logging.warning(
                "[HL_DATA_SVC] Failed to fetch allMids for coin filtering. Falling back to configured coins."
            )
            return configured_coins

        supported = configured_coins.intersection(all_supported_keys)
        if not supported:
            supported = configured_coins
        elif supported != configured_coins:
            skipped = sorted(configured_coins - supported)
            bt.logging.info(
                f"[HL_DATA_SVC] Skipping unsupported l2Book coins on current HL env: {skipped}"
            )
        return supported

    def instantiate_not_pickleable_objects(self):
        pass


if __name__ == "__main__":
    import asyncio as _asyncio

    print("Creating HyperliquidDataService...")
    service = HyperliquidDataService(disable_ws=True, running_unit_tests=True)

    coins = list(service._coin_to_trade_pair.keys())
    print(f"Crypto coins ({len(coins)}): {coins}")

    print(f"\nConnecting to {ValiConfig.hl_ws_url()}...")

    async def run():
        ws = await websockets.connect(ValiConfig.hl_ws_url())
        try:
            for coin in coins:
                sub = {"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}
                await ws.send(json.dumps(sub))
            print(f"Subscribed to {len(coins)} coins")

            msg_count = 0
            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                if data.get("channel") != "l2Book":
                    print(f"  [{data.get('channel', 'unknown')}] {str(data)[:120]}")
                    continue

                msg_count += 1
                book = data["data"]
                coin = book["coin"]
                levels = book.get("levels", [])
                if len(levels) >= 2 and levels[0] and levels[1]:
                    bid = float(levels[0][0]["px"])
                    ask = float(levels[1][0]["px"])
                    mid = (bid + ask) / 2
                    print(f"[{msg_count}] {coin}: mid={mid:.2f} bid={bid:.2f} ask={ask:.2f}")
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            await ws.close()

    try:
        _asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped by user")
