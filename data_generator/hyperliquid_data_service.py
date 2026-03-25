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
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory
from vali_objects.vali_dataclasses.price_source import PriceSource
from vali_objects.vali_dataclasses.recent_event_tracker import RecentEventTracker

HYPERLIQUID_WS_URL = "wss://api.hyperliquid.xyz/ws"
HYPERLIQUID_REST_URL = "https://api.hyperliquid.xyz/info"
REST_TIMEOUT_S = 10
RECV_TIMEOUT_S = 30


class _HyperliquidWebsocketClient:
    """Websocket client for Hyperliquid L2 orderbook data."""

    def __init__(self, service, category):
        self._svc = service
        self._cat = category
        self._ws = None
        self._should_close = False

    async def connect(self, handle_msg):
        """Connect to Hyperliquid L2 orderbook websocket and process messages."""
        self._ws = await websockets.connect(HYPERLIQUID_WS_URL)

        try:
            # Subscribe to l2Book for each crypto coin
            trade_pairs = self._svc.get_tradeable_pairs(
                category=self._cat,
                include_blocked=False,
                market_open_only=False
            )

            for tp in trade_pairs:
                subscribe_msg = {
                    "method": "subscribe",
                    "subscription": {"type": "l2Book", "coin": tp.base}
                }
                await self._ws.send(json.dumps(subscribe_msg))

            bt.logging.info(f"Subscribed to Hyperliquid l2Book for {len(trade_pairs)} coins: "
                            f"{[tp.base for tp in trade_pairs]}")

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


class HyperliquidDataService(BaseDataService):
    """Crypto-only live WebSocket feed from Hyperliquid using L2 orderbook data."""

    def __init__(self, disable_ws=False, running_unit_tests=False):
        super().__init__(
            provider_name=HYPERLIQUID_PROVIDER_NAME,
            running_unit_tests=running_unit_tests,
            enabled_websocket_categories={TradePairCategory.CRYPTO}
        )

        # Build coin name -> TradePair mapping
        self._coin_to_trade_pair = {}
        for tp in TradePair:
            if tp.is_crypto and tp not in self.UNSUPPORTED_TRADE_PAIRS:
                self._coin_to_trade_pair[tp.base] = tp

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
        client = _HyperliquidWebsocketClient(self, tpc)
        self.WEBSOCKET_OBJECTS[tpc] = client
        bt.logging.info(f"Created {self.provider_name} websocket client for {tpc}")

    def _subscribe_websockets(self, tpc):
        # Subscription happens inside connect()
        pass

    async def handle_msg(self, msg):
        try:
            data = json.loads(msg)

            if data.get("channel") != "l2Book":
                return

            book_data = data.get("data", {})
            coin = book_data.get("coin")
            if not coin:
                return

            tp = self._coin_to_trade_pair.get(coin)
            if not tp:
                return

            levels = book_data.get("levels", [])
            if len(levels) < 2 or not levels[0] or not levels[1]:
                return

            best_bid = float(levels[0][0]["px"])
            best_ask = float(levels[1][0]["px"])
            mid_price = (best_bid + best_ask) / 2.0

            timestamp_ms = book_data.get("time", TimeUtil.now_in_millis())
            timestamp_ms = round(timestamp_ms, -3)  # Round to nearest second for dedup

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
                ask=best_ask
            )

            symbol = tp.trade_pair
            self.latest_websocket_events[symbol] = ps
            if symbol not in self.trade_pair_to_recent_events:
                self.trade_pair_to_recent_events[symbol] = RecentEventTracker()
            self.trade_pair_to_recent_events[symbol].add_event(
                ps, False, f"{self.provider_name}:{tp.trade_pair}"
            )

            self.tpc_to_n_events[TradePairCategory.CRYPTO] += 1
            self.tpc_to_last_event_time[TradePairCategory.CRYPTO] = time.time()
            self.closed_market_prices[tp] = None

        except Exception as e:
            full_traceback = traceback.format_exc()
            limited_traceback = full_traceback[-1000:]
            bt.logging.error(
                f"Failed to handle {HYPERLIQUID_PROVIDER_NAME} websocket message "
                f"with error: {e}, type: {type(e).__name__}, "
                f"traceback: {limited_traceback}"
            )

    def _fetch_all_mids(self) -> dict[str, float]:
        """Fetch mid prices for all coins via the REST API. Returns {coin: mid_price}."""
        try:
            resp = requests.post(
                HYPERLIQUID_REST_URL,
                json={"type": "allMids"},
                timeout=REST_TIMEOUT_S,
            )
            resp.raise_for_status()
            return {coin: float(price) for coin, price in resp.json().items()}
        except Exception as e:
            bt.logging.error(f"Hyperliquid REST allMids failed: {type(e).__name__}: {e}")
            return {}

    def _fetch_l2_book(self, coin: str) -> tuple[float, float] | None:
        """Fetch best bid/ask for a single coin via the REST API."""
        try:
            resp = requests.post(
                HYPERLIQUID_REST_URL,
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

    def get_closes_rest(self, trade_pairs: List[TradePair], time_ms, live=True) -> dict[TradePair, PriceSource]:
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

    def get_close_rest(self, trade_pair: TradePair, timestamp_ms: int) -> PriceSource | None:
        """Single-pair REST fallback."""
        results = self.get_closes_rest([trade_pair], timestamp_ms)
        return results.get(trade_pair)

    def instantiate_not_pickleable_objects(self):
        pass


if __name__ == "__main__":
    import asyncio as _asyncio

    print("Creating HyperliquidDataService...")
    service = HyperliquidDataService(disable_ws=True, running_unit_tests=True)

    coins = list(service._coin_to_trade_pair.keys())
    print(f"Crypto coins ({len(coins)}): {coins}")

    print(f"\nConnecting to {HYPERLIQUID_WS_URL}...")

    async def run():
        ws = await websockets.connect(HYPERLIQUID_WS_URL)
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
