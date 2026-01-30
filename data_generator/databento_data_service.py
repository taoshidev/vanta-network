import threading
import time
import bittensor as bt
import databento as db

from data_generator.base_data_service import BaseDataService
from time_util.time_util import TimeUtil
from vali_objects.vali_config import TradePair, TradePairCategory
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
        # Reuse existing db.Live client - it uses a class-level singleton thread
        # that can only be started once. The client can be reused after stop().
        if self._client is None:
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
        """Stop the client connection but keep client for reuse."""
        if self._client:
            try:
                self._client.stop()
            except Exception as e:
                bt.logging.warning(f"Error stopping Databento client: {e}")
            # Don't set to None - db.Live can be reused after stop()

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

        # Start websocket manager thread (uses base class implementation)
        if disable_ws:
            self.websocket_manager_thread = None
        else:
            self.websocket_manager_thread = threading.Thread(target=self.websocket_manager, daemon=True)
            self.websocket_manager_thread.start()

    def _get_equity_symbols(self) -> list[str]:
        """Get all equity symbols from TradePair config."""
        symbols = []
        for tp in TradePair:
            if tp.is_equities and tp not in self.UNSUPPORTED_TRADE_PAIRS:
                symbols.append(tp.trade_pair)
        return symbols

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

        # Get bid/ask from first level (prices are in fixed-point, divide by 1e9)
        bid = msg.levels[0].bid_px / 1e9
        ask = msg.levels[0].ask_px / 1e9
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

    def get_stock_splits(self, time_ms) -> dict[str, float]:
        """
        Get stock splits for all equity symbols on a given date.

        Returns:
            dict mapping trade_pair_id to split ratio (ratio_new / ratio_old)
        """
        execution_date_str = TimeUtil.timestamp_ms_to_eastern_time_str(time_ms, short=True)

        try:
            df_raw = self._ref_client.corporate_actions.get_range(
                symbols=self._get_equity_symbols(),
                start=execution_date_str,
                end=execution_date_str,
                index="ex_date",
                events=["FSPLT", "RSPLT"],
                countries=["US"]
            )
        except Exception as e:
            bt.logging.error(f"Failed to fetch stock splits from Databento: {e}")
            return {}

        if df_raw is None or df_raw.empty:
            return {}

        result = {}
        for _, row in df_raw.iterrows():
            symbol = row.get("symbol")
            ratio_old = row.get("ratio_old")
            ratio_new = row.get("ratio_new")

            if symbol and ratio_old and ratio_new and ratio_old != 0:
                # trade_pair_id is the symbol for equities
                result[symbol] = ratio_new / ratio_old

        return result



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
            bid = msg.levels[0].bid_px / 1e9
            ask = msg.levels[0].ask_px / 1e9
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
