import threading
import time
from datetime import datetime, timedelta
from typing import List
from zoneinfo import ZoneInfo
import bittensor as bt
import databento as db

from data_generator.base_data_service import BaseDataService
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory
from vali_objects.vali_dataclasses.corporate_actions import CorporateActions, DividendEvent
from vali_objects.vali_dataclasses.price_source import PriceSource

DATABENTO_PROVIDER_NAME = "Databento"


class DatabentoWebSocketClient:
    """
    Wrapper around db.Live to match Polygon WebSocketClient interface.

    db.Live uses a class-level singleton thread that can only be started once,
    so we reuse the same client instance across reconnections. After stop(),
    calling subscribe() and iterating will reconnect with a fresh session.
    """

    DATASET = "EQUS.MINI"
    SCHEMA = "bbo-1s"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = None
        self._symbols = []
        self._instrument_map = {}

    def subscribe(self, symbol: str):
        """Queue symbol for subscription (called before connect)."""
        self._symbols.append(symbol)

    async def connect(self, handler):
        """Connect and process messages via callback."""
        # Reset dead singleton thread before creating new db.Live instance.
        # db.Live uses a class-level _thread that runs _loop.run_forever().
        # When the thread dies, we must replace it with a new thread that
        # has the same target so the library can start it again.
        if hasattr(db.Live, '_thread') and db.Live._thread is not None:
            if not db.Live._thread.is_alive():
                bt.logging.info("Replacing dead Databento singleton thread")
                # Recreate with same target as original: _loop.run_forever
                db.Live._thread = threading.Thread(
                    target=db.Live._loop.run_forever,
                    name="databento_live",
                    daemon=True,
                )

        # Create fresh client each time - the old one can't be reused after stop()
        self._client = db.Live(key=self._api_key)
        bt.logging.info("Created new Databento Live client")
        self._client.subscribe(
            dataset=self.DATASET,
            schema=self.SCHEMA,
            symbols=self._symbols
        )

        bt.logging.info(f"Databento websocket connected, subscribed to {len(self._symbols)} symbols")

        # Translate async iteration to callback pattern
        async for msg in self._client:
            # Handle symbol mapping messages internally
            if isinstance(msg, db.SymbolMappingMsg):
                self._instrument_map[msg.instrument_id] = msg.stype_in_symbol
                continue

            # Attach symbol resolution to message for handler
            if hasattr(msg, 'instrument_id'):
                msg._resolved_symbol = self._instrument_map.get(msg.instrument_id)

            await handler(msg)

    def unsubscribe_all(self):
        """Clear pending subscriptions."""
        self._symbols.clear()

    def stop(self):
        """Stop the client connection."""
        if self._client:
            try:
                self._client.stop()
            except Exception as e:
                bt.logging.warning(f"Error stopping Databento client: {e}")
            # Set to None so a fresh client is created on reconnect.
            # db.Live's singleton thread can't be restarted, so we need a new instance.
            self._client = None

    def get_symbol(self, instrument_id: int) -> str | None:
        """Get resolved symbol for an instrument ID."""
        return self._instrument_map.get(instrument_id)


class DatabentoDataService(BaseDataService):
    """Equities-only live WebSocket feed from Databento using bbo-1s schema."""

    def __init__(self, api_key: str, disable_ws=False, running_unit_tests=False):
        super().__init__(
            DATABENTO_PROVIDER_NAME,
            running_unit_tests,
            enabled_websocket_categories={TradePairCategory.EQUITIES}
        )
        self._api_key = api_key
        self._ref_client = db.Reference(key=api_key)
        self._hist_client = db.Historical(key=api_key)
        self._corporate_actions_cache: dict[str, CorporateActions] = {}

        # Start websocket manager thread (uses base class implementation)
        if disable_ws:
            self.websocket_manager_thread = None
        else:
            self.websocket_manager_thread = threading.Thread(target=self.websocket_manager, daemon=True)
            self.websocket_manager_thread.start()

    # Symbols not present in Databento's corporate actions dataset
    CORPORATE_ACTIONS_EXCLUDED_SYMBOLS = {"BRK.B"}

    def _get_equity_symbols(self) -> list[str]:
        """Get all equity symbols from TradePair config."""
        symbols = []
        for tp in TradePair:
            if tp.is_equities and tp not in self.UNSUPPORTED_TRADE_PAIRS:
                symbols.append(tp.trade_pair)
        return symbols

    def _get_corporate_action_symbols(self) -> list[str]:
        return [s for s in self._get_equity_symbols() if s not in self.CORPORATE_ACTIONS_EXCLUDED_SYMBOLS]

    def _create_websocket_client(self, tpc: TradePairCategory):
        """Create or reuse Databento websocket client wrapper for equities."""
        if tpc != TradePairCategory.EQUITIES:
            return

        # Reuse existing client - db.Live uses singleton thread that can't restart
        existing = self.WEBSOCKET_OBJECTS.get(tpc)
        if existing is not None:
            bt.logging.info(f"Reusing existing {self.provider_name} websocket client for {tpc}")
            return

        client = DatabentoWebSocketClient(api_key=self._api_key)
        self.WEBSOCKET_OBJECTS[tpc] = client
        bt.logging.info(f"Created {self.provider_name} websocket client for {tpc}")

    def _subscribe_websockets(self, tpc: TradePairCategory):
        """Subscribe to all equity symbols."""
        if tpc != TradePairCategory.EQUITIES:
            return

        symbols = self._get_equity_symbols()
        if not symbols:
            bt.logging.warning("No equity symbols to subscribe to")
            return

        client = self.WEBSOCKET_OBJECTS.get(tpc)
        if client is None:
            bt.logging.error(f"No client available for {tpc}")
            return

        for symbol in symbols:
            client.subscribe(symbol)
        bt.logging.info(f"{self.provider_name} queued {len(symbols)} symbols for subscription")

    async def handle_msg(self, msg):
        """Convert Databento BBO message to PriceSource and update state."""
        # Skip non-BBO messages
        if not isinstance(msg, db.BBOMsg):
            return

        # Get resolved symbol from wrapper (attached during iteration)
        symbol = getattr(msg, '_resolved_symbol', None)
        if symbol is None:
            return

        tp = self.trade_pair_lookup.get(symbol)
        if tp is None:
            return

        # Convert nanoseconds to milliseconds
        timestamp_ms = msg.ts_event // 1_000_000

        # Databento uses INT64_MAX as null sentinel for missing prices
        UNDEF_PRICE = 9_223_372_036_854_775_807
        raw_bid = msg.levels[0].bid_px
        raw_ask = msg.levels[0].ask_px
        if raw_bid == UNDEF_PRICE or raw_ask == UNDEF_PRICE:
            return

        # Get bid/ask from first level (prices are in fixed-point, divide by 1e9)
        bid = raw_bid / 1e9
        ask = raw_ask / 1e9
        mid = (bid + ask) / 2

        ps = PriceSource(
            source=f"{DATABENTO_PROVIDER_NAME}_ws",
            timespan_ms=1000,  # 1 second interval for BBO-1s
            open=mid,
            close=mid,
            vwap=mid,
            high=mid,
            low=mid,
            start_ms=timestamp_ms,
            websocket=True,
            lag_ms=0,
            bid=bid,
            ask=ask,
        )

        # Update state
        self.latest_websocket_events[symbol] = ps
        self.trade_pair_to_recent_events[symbol].add_event(ps)
        self.tpc_to_n_events[TradePairCategory.EQUITIES] += 1
        self.tpc_to_last_event_time[TradePairCategory.EQUITIES] = time.time()

        # Reset closed market price
        self.closed_market_prices[tp] = None

    async def _cleanup_websocket(self, tpc: TradePairCategory):
        """Clean up websocket but keep client for reuse."""
        client = self.WEBSOCKET_OBJECTS.get(tpc)
        if client:
            try:
                client.unsubscribe_all()
                client.stop()
                bt.logging.info(f"Cleaned up {self.provider_name}[{tpc}] websocket (keeping client)")
            except Exception as e:
                bt.logging.error(f"Cleanup error for {tpc}: {e}")
            # Don't set to None - we want to reuse the client

    def instantiate_not_pickleable_objects(self):
        """Initialize non-pickleable clients after unpickling."""
        # Live client will be created in _create_websocket_client
        pass

    @staticmethod
    def _add_days(date_str: str, days: int) -> str:
        """Return a date string offset by the given number of days."""
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return (d + timedelta(days=days)).strftime("%Y-%m-%d")

    @staticmethod
    def _to_date_str(value) -> str:
        """
        Databento returns ISO 8601 dates expressed in the local time of the
        listing exchange. The value may arrive as a str, datetime.date, or
        pd.Timestamp depending on the client version.
        """
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        # Already a string — take first 10 chars to strip any time component
        return str(value)[:10]

    def get_closes_rest(self, trade_pairs: List[TradePair], time_ms: int, live: bool = False) -> dict[TradePair, PriceSource]:
        """Historical daily closes from Databento (equities only). Returns empty if live=True."""
        if live:
            return {}

        if self.running_unit_tests:
            from data_generator.polygon_data_service import PolygonDataService
            return {tp: PolygonDataService.DEFAULT_TESTING_FALLBACK_PRICE_SOURCE for tp in trade_pairs}

        equity_pairs = [tp for tp in trade_pairs if tp.trade_pair_category == TradePairCategory.EQUITIES]
        if not equity_pairs:
            return {}

        target_dt = datetime.fromtimestamp(time_ms / 1000, tz=ZoneInfo("UTC"))
        start_dt = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_dt = start_dt + timedelta(days=1)

        tp_to_price: dict[TradePair, PriceSource] = {}
        time_now_ms = int(time.time() * 1000)

        for tp in equity_pairs:
            try:
                store = self._hist_client.timeseries.get_range(
                    dataset="EQUS.MINI",
                    schema="ohlcv-1d",
                    symbols=[tp.trade_pair],
                    start=start_dt,
                    end=end_dt,
                )
                df = store.to_df(pretty_px=True)
                if df.empty:
                    bt.logging.warning(f"Databento: no ohlcv-1d data for {tp.trade_pair} on {start_dt.date()}")
                    continue

                row = df.iloc[-1]
                close = float(row["close"])
                open_ = float(row["open"])
                high = float(row["high"])
                low = float(row["low"])

                # ts_event is the index as a pandas Timestamp (nanoseconds -> Timestamp by pandas)
                ts_event = df.index[-1]
                bar_start_ms = int(ts_event.timestamp() * 1000)

                tp_to_price[tp] = PriceSource(
                    source=f"{DATABENTO_PROVIDER_NAME}_rest",
                    timespan_ms=86_400_000,
                    open=open_,
                    close=close,
                    vwap=close,
                    high=high,
                    low=low,
                    start_ms=bar_start_ms,
                    websocket=False,
                    lag_ms=time_now_ms - bar_start_ms,
                    bid=0,
                    ask=0,
                )
            except Exception as e:
                bt.logging.error(f"Databento historical REST failed for {tp.trade_pair}: {e}")

        return tp_to_price

    def get_corporate_actions(
        self,
        start_date_str: str,
        end_date_str: str | None = None,
    ) -> dict[str, CorporateActions]:
        """
        Fetch stock splits and dividends for all equity symbols over a date range.

        Args:
            start_date_str: Start date (inclusive) in YYYY-MM-DD format.
            end_date_str: End date (exclusive) in YYYY-MM-DD format.
                          Defaults to 3 days after start_date_str.

        Returns:
            dict mapping ex_date string to CorporateActions for that date.
        """
        if end_date_str is None:
            end_date_str = self._add_days(start_date_str, 3)

        # Determine which dates in the range are not yet cached
        dates_in_range = []
        d = start_date_str
        while d < end_date_str:
            if d not in self._corporate_actions_cache:
                dates_in_range.append(d)
            d = self._add_days(d, 1)

        if not dates_in_range:
            return {d: self._corporate_actions_cache[d] for d in self._corporate_actions_cache
                    if start_date_str <= d < end_date_str}

        symbols = self._get_corporate_action_symbols()
        result: dict[str, CorporateActions] = {}

        # Fetch splits
        try:
            df_splits = self._ref_client.corporate_actions.get_range(
                symbols=symbols,
                stype_in="nasdaq_symbol",
                start=start_date_str,
                end=end_date_str,
                index="ex_date",
                events=["FSPLT", "RSPLT"],
                countries=["US"]
            )
            if df_splits is not None and not df_splits.empty:
                for ex_date, row in df_splits.iterrows():
                    symbol = row.get("symbol")
                    ratio_old = row.get("ratio_old")
                    ratio_new = row.get("ratio_new")
                    if symbol and ratio_old and ratio_new and ratio_old != 0:
                        date_str = self._to_date_str(ex_date)
                        if date_str not in result:
                            result[date_str] = CorporateActions(splits={}, dividends={})
                        result[date_str].splits[symbol] = ratio_new / ratio_old
        except Exception as e:
            bt.logging.error(f"Failed to fetch stock splits from Databento: {e}")

        # Fetch dividends
        try:
            df_divs = self._ref_client.corporate_actions.get_range(
                symbols=symbols,
                stype_in="nasdaq_symbol",
                start=start_date_str,
                end=end_date_str,
                index="ex_date",
                events=["DIV"],
                countries=["US"]
            )
            if df_divs is not None and not df_divs.empty:
                for ex_date, row in df_divs.iterrows():
                    symbol = row.get("symbol")
                    gross_dividend = row.get("gross_dividend", 0)
                    payment_date = row.get("payment_date", "")
                    if symbol and gross_dividend is not None and gross_dividend > 0:
                        date_str = self._to_date_str(ex_date)
                        if date_str not in result:
                            result[date_str] = CorporateActions(splits={}, dividends={})
                        result[date_str].dividends[symbol] = DividendEvent(
                            gross_dividend=float(gross_dividend),
                            ex_date=date_str,
                            payment_date=str(payment_date) if payment_date else "",
                        )
        except Exception as e:
            bt.logging.error(f"Failed to fetch dividend events from Databento: {e}")

        # Cache each date individually (including empty dates so we don't re-fetch)
        for d in dates_in_range:
            self._corporate_actions_cache[d] = result.get(d, CorporateActions(splits={}, dividends={}))

        actions_in_range = {d: self._corporate_actions_cache[d] for d in self._corporate_actions_cache
                if start_date_str <= d < end_date_str}

        bt.logging.info(f"Databento corporate actions in range ({start_date_str} - {end_date_str}) {actions_in_range}")

        return actions_in_range

if __name__ == "__main__":
    import asyncio
    from vali_objects.utils.vali_utils import ValiUtils

    secrets = ValiUtils.get_secrets()
    api_key = secrets.get("databento_apikey")

    if not api_key:
        print("Error: databento_apikey not found in secrets")
        exit(1)

    print("Creating DatabentoDataService...")
    # Use disable_ws=True to prevent background thread from starting
    service = DatabentoDataService(api_key=api_key, disable_ws=True)

    symbols = service._get_equity_symbols()
    print(f"Equity symbols ({len(symbols)}): {symbols}")

    # Create client and subscribe directly using wrapper constants
    dataset = DatabentoWebSocketClient.DATASET
    schema = DatabentoWebSocketClient.SCHEMA
    print(f"\nConnecting to {dataset} with schema {schema}...")
    client = db.Live(key=api_key)
    client.subscribe(
        dataset=dataset,
        schema=schema,
        symbols=symbols,
    )
    print(f"Subscribed to {len(symbols)} symbols")

    # Run the websocket
    async def run():
        msg_count = 0
        symbol_map = {}  # instrument_id -> symbol

        async for msg in client:
            # Capture symbol mappings
            if isinstance(msg, db.SymbolMappingMsg):
                # Print first one to see attributes
                if not symbol_map:
                    print(f"SymbolMappingMsg attrs: {[a for a in dir(msg) if not a.startswith('_')]}")
                symbol_map[msg.instrument_id] = msg.stype_in_symbol
                print(f"Mapped {msg.instrument_id} -> {msg.stype_in_symbol}")
                continue

            if not isinstance(msg, db.BBOMsg):
                continue

            msg_count += 1
            instrument_id = msg.instrument_id
            symbol = symbol_map.get(instrument_id, f"unknown:{instrument_id}")
            UNDEF_PRICE = 9_223_372_036_854_775_807
            raw_bid = msg.levels[0].bid_px
            raw_ask = msg.levels[0].ask_px
            if raw_bid == UNDEF_PRICE or raw_ask == UNDEF_PRICE:
                print(f"[{msg_count}] {symbol}: skipping (null bid/ask sentinel)")
                continue
            bid = raw_bid / 1e9
            ask = raw_ask / 1e9
            price = msg.price / 1e9
            print(f"[{msg_count}] {symbol}: price={price:.2f} bid={bid:.2f} ask={ask:.2f}")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        # Close the client connection to free up the connection slot
        if client:
            try:
                client.stop()
                print("Client connection closed")
            except Exception as e:
                print(f"Error closing client: {e}")
