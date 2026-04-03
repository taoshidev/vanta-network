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
from vali_objects.vali_config import TradePair, TradePairCategory, TradePairLike, ValiConfig
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker

REST_TIMEOUT_S = 10
RECV_TIMEOUT_S = 30
_L2_COIN_CACHE_TTL_S = 300.0


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
            # Get the filtered, env-aware coin list from the service (includes dynamic coins,
            # filtered by allMids availability to prevent testnet socket closes).
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


class _DualL2BookClient:
    """Manages two concurrent L2 book WebSocket connections at different resolutions.

    Full precision (no nSigFigs) provides native tick-size pricing for accurate mid and slippage.
    Coarse (nSigFigs=2) provides wider depth for large orders that exhaust the full-precision book.
    Messages are routed to the service's resolution-specific handlers.
    """

    def __init__(self, service, category):
        self._full = _HyperliquidWebsocketClient(service, category, n_sig_figs=None)
        self._coarse = _HyperliquidWebsocketClient(service, category, n_sig_figs=ValiConfig.HL_L2_COARSE_SIG_FIGS)
        self._svc = service

    async def connect(self, handle_msg):
        """Run both WebSocket connections concurrently."""
        await asyncio.gather(
            self._full.connect(self._svc.handle_msg_full),
            self._coarse.connect(self._svc.handle_msg_coarse),
        )

    async def close(self):
        await asyncio.gather(self._full.close(), self._coarse.close())

    def unsubscribe_all(self):
        self._full.unsubscribe_all()
        self._coarse.unsubscribe_all()


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
            if tp.is_crypto and tp not in self.UNSUPPORTED_TRADE_PAIRS:
                self._coin_to_trade_pair[tp.base] = tp

        # Dual-resolution L2 orderbook cache per coin.
        # Full precision (no nSigFigs): native tick-size pricing for accurate mid and near-spread slippage.
        # Coarse (nSigFigs=2): wider depth for large orders that exhaust the full-precision book.
        # Key: coin name (e.g. "BTC"), Value: {"bids": [...], "asks": [...], "time": timestamp_ms}
        self._orderbooks_full: dict[str, dict] = {}
        self._orderbooks_coarse: dict[str, dict] = {}

        # Cache for subscription coin list (filtered by allMids availability).
        # Persists across reconnects to avoid repeated REST calls on reconnect storms.
        self._l2_coin_cache: set[str] | None = None
        self._l2_coin_cache_ts: float = 0.0

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
        client = _DualL2BookClient(self, tpc)
        self.WEBSOCKET_OBJECTS[tpc] = client
        bt.logging.info(f"Created {self.provider_name} dual-resolution websocket client for {tpc}")

    def _subscribe_websockets(self, tpc):
        # Subscription happens inside connect()
        pass

    def _parse_l2_book_msg(self, msg):
        """Parse an l2Book WebSocket message.

        Returns (coin, tp, bids, asks, timestamp_ms) or None.
        tp is a TradePair for static coins, a DynamicTradePair for dynamic coins,
        or None (and the whole result is None) for completely unknown coins.
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
            from vali_objects.vali_config import HL_COIN_TO_DYNAMIC_TRADE_PAIR
            tp = HL_COIN_TO_DYNAMIC_TRADE_PAIR.get(coin)
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

    async def handle_msg_coarse(self, msg):
        """Handle nSigFigs=2 l2Book messages: update coarse orderbook cache for deep slippage walks."""
        try:
            parsed = self._parse_l2_book_msg(msg)
            if parsed is None:
                return
            coin, _tp, bids, asks, timestamp_ms = parsed
            self._orderbooks_coarse[coin] = {"bids": bids, "asks": asks, "time": timestamp_ms}
        except Exception as e:
            limited_traceback = traceback.format_exc()[-1000:]
            bt.logging.error(
                f"Failed to handle {HYPERLIQUID_PROVIDER_NAME} coarse websocket message "
                f"with error: {e}, type: {type(e).__name__}, traceback: {limited_traceback}"
            )

    def _fetch_all_mids(self) -> dict[str, float]:
        """Fetch mid prices for all coins across all dexes via the REST API.

        Fetches the default crypto dex first, then merges in each non-default dex
        (identified by the colon-prefixed hl_coin names in HL_DYNAMIC_REGISTRY).
        Returns {coin: mid_price} with prefixed keys for non-default dex coins (e.g. "xyz:TSLA").
        """
        from vali_objects.vali_config import HL_DYNAMIC_REGISTRY

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

        # Non-default dexes — derive names from prefixed hl_coin entries in the registry
        non_default_dexes = {
            dtp.hl_coin.split(":")[0]
            for dtp in HL_DYNAMIC_REGISTRY.values()
            if ":" in dtp.hl_coin
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

    def get_closes_rest(self, trade_pairs: List[TradePairLike], time_ms, live=True) -> dict[TradePairLike, PriceSource]:
        """REST fallback: fetch mid prices from Hyperliquid for the requested crypto pairs."""
        if self.running_unit_tests:
            from data_generator.polygon_data_service import PolygonDataService
            return {tp: PolygonDataService.DEFAULT_TESTING_FALLBACK_PRICE_SOURCE for tp in trade_pairs}

        crypto_pairs = [tp for tp in trade_pairs if tp.is_crypto and tp not in self.UNSUPPORTED_TRADE_PAIRS]
        if not crypto_pairs:
            return {}

        now_ms = TimeUtil.now_in_millis()

        # Use the bulk allMids endpoint first
        all_mids = self._fetch_all_mids()

        results: dict[TradePair, PriceSource] = {}
        pairs_needing_book = []

        for tp in crypto_pairs:
            mid = all_mids.get(tp.base)
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

        # Fall back to individual l2Book calls for any coins missing from allMids
        for tp in pairs_needing_book:
            book = self._fetch_l2_book(tp.base)
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

    def simulate_slippage(self, trade_pair: TradePairLike, size_usd: float, is_buy: bool) -> float | None:
        """Simulate slippage using a dual-resolution two-phase orderbook walk.

        Phase 1 walks the full-precision book (nSigFigs=None) for accurate near-spread
        pricing. Phase 2 continues with coarse levels (nSigFigs=2) priced beyond the
        last full level if the order exhausts the full-precision book. Falls back to coarse-only
        if the full-precision book is not yet populated.

        Args:
            trade_pair: The trade pair to calculate slippage for.
            size_usd: The order size in USD.
            is_buy: True for LONG orders (fill against asks),
                    False for SHORT orders (fill against bids).

        Returns:
            Slippage as a fraction (e.g. 0.001 for 0.1%), or None if no
            orderbook data is available.
        """
        coin = trade_pair.base
        full_book = self._orderbooks_full.get(coin, {})
        coarse_book = self._orderbooks_coarse.get(coin, {})

        primary = full_book or coarse_book
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
        full_levels = full_book.get(side, [])
        coarse_levels = coarse_book.get(side, [])

        # Phase 1: walk full-grained levels
        if full_levels:
            fills, remaining = simulate_fill(full_levels, size_usd, "usd")
        else:
            fills, remaining = [], size_usd

        # Phase 2: continue with coarse levels beyond full book's price coverage
        if remaining > 0 and coarse_levels:
            if full_levels:
                last_full_px = float(full_levels[-1]["px"])
                if is_buy:
                    deeper = [l for l in coarse_levels if float(l["px"]) > last_full_px]
                else:
                    deeper = [l for l in coarse_levels if float(l["px"]) < last_full_px]
            else:
                deeper = coarse_levels
            coarse_fills, _ = simulate_fill(deeper, remaining, "usd")
            fills.extend(coarse_fills)

        if not fills:
            return None

        total_coins = sum(f[1] for f in fills)
        total_usd = sum(f[2] for f in fills)
        if total_coins <= 0:
            return None

        avg_price = total_usd / total_coins
        slippage_pct = (avg_price - mid) / mid if is_buy else (mid - avg_price) / mid
        return max(0.0, slippage_pct)

    def simulate_avg_fill_price(self, trade_pair: TradePairLike, size_usd: float, is_buy: bool) -> float | None:
        """Simulate the average fill price for a market order using the L2 orderbook.

        Uses the same dual-resolution two-phase orderbook walk as simulate_slippage,
        but returns the raw avg fill price instead of a slippage fraction. This is used
        for HL taker fills where we want to record the actual execution price directly.

        Args:
            trade_pair: The trade pair to simulate.
            size_usd: The order size in USD.
            is_buy: True for LONG orders (fill against asks),
                    False for SHORT orders (fill against bids).

        Returns:
            Average fill price in quote currency, or None if no orderbook data is available.
        """
        coin = trade_pair.base
        full_book = self._orderbooks_full.get(coin, {})
        coarse_book = self._orderbooks_coarse.get(coin, {})

        primary = full_book or coarse_book
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
        full_levels = full_book.get(side, [])
        coarse_levels = coarse_book.get(side, [])

        # Phase 1: walk full-grained levels
        if full_levels:
            fills, remaining = simulate_fill(full_levels, size_usd, "usd")
        else:
            fills, remaining = [], size_usd

        # Phase 2: continue with coarse levels beyond full book's price coverage
        if remaining > 0 and coarse_levels:
            if full_levels:
                last_full_px = float(full_levels[-1]["px"])
                if is_buy:
                    deeper = [l for l in coarse_levels if float(l["px"]) > last_full_px]
                else:
                    deeper = [l for l in coarse_levels if float(l["px"]) < last_full_px]
            else:
                deeper = coarse_levels
            coarse_fills, _ = simulate_fill(deeper, remaining, "usd")
            fills.extend(coarse_fills)

        if not fills:
            return None

        total_coins = sum(f[1] for f in fills)
        total_usd = sum(f[2] for f in fills)
        if total_coins <= 0:
            return None

        return total_usd / total_coins

    def get_close_rest(self, trade_pair: TradePairLike, timestamp_ms: int) -> PriceSource | None:
        """Single-pair REST fallback."""
        results = self.get_closes_rest([trade_pair], timestamp_ms)
        return results.get(trade_pair)

    def _get_subscription_coins(self) -> set[str]:
        """Return the filtered set of HL coins to subscribe to for l2Book streams.

        Builds the configured coin set from (in priority order):
          1. Static TradePair crypto members
          2. Dynamic coins from HL_DYNAMIC_REGISTRY

        Then intersects with allMids availability to avoid subscribing to unsupported
        coins on testnet, which causes the HL server to close the WebSocket connection.
        Result is cached for _L2_COIN_CACHE_TTL_S seconds across reconnects.
        """
        from vali_objects.vali_config import HL_DYNAMIC_REGISTRY

        configured_coins = set(self._coin_to_trade_pair.keys())
        configured_coins.update(dtp.hl_coin for dtp in HL_DYNAMIC_REGISTRY.values())

        now = time.time()
        if self._l2_coin_cache is not None and (now - self._l2_coin_cache_ts) < _L2_COIN_CACHE_TTL_S:
            # Re-expand cached coins with any new dynamic entries not yet in the cache.
            return self._l2_coin_cache | (configured_coins - self._l2_coin_cache)

        try:
            resp = requests.post(
                ValiConfig.hl_info_url(),
                json={"type": "allMids"},
                timeout=5,
            )
            mids = resp.json()
            if isinstance(mids, dict):
                all_supported_keys = set(mids.keys())
                # Also fetch non-default dex allMids so prefixed coins are not filtered out
                non_default_dexes = {
                    dtp.hl_coin.split(":")[0]
                    for dtp in HL_DYNAMIC_REGISTRY.values()
                    if ":" in dtp.hl_coin
                }
                for dex in non_default_dexes:
                    try:
                        r = requests.post(
                            ValiConfig.hl_info_url(),
                            json={"type": "allMids", "dex": dex},
                            timeout=5,
                        )
                        all_supported_keys.update(r.json().keys())
                    except Exception:
                        pass
                supported = configured_coins.intersection(all_supported_keys)
            else:
                supported = configured_coins

            if not supported:
                supported = configured_coins
            elif supported != configured_coins:
                skipped = sorted(configured_coins - supported)
                bt.logging.info(
                    f"[HL_DATA_SVC] Skipping unsupported l2Book coins on current HL env: {skipped}"
                )

            self._l2_coin_cache = supported
            self._l2_coin_cache_ts = now
            return supported
        except Exception as e:
            bt.logging.warning(
                f"[HL_DATA_SVC] Failed to fetch allMids for coin filtering: {e}. "
                "Falling back to configured coins."
            )
            if self._l2_coin_cache:
                return self._l2_coin_cache | configured_coins
            return configured_coins

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
